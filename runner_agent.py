"""Tool-loop worker for an approved GPT-OSS GitHub Actions task."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests


MODEL = os.environ.get("AGENT_MODEL", "gpt-oss:20b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
TASK = os.environ.get("AGENT_TASK", "").strip()
SESSION_ID = os.environ.get("AGENT_SESSION_ID", "dashboard")
ROOT = Path.cwd().resolve()
OUTPUT_DIR = ROOT / "agent_output"
MAX_STEPS = 24
MAX_FILE_BYTES = 512_000


def safe_path(value: str) -> Path:
    candidate = (ROOT / value).resolve()
    if candidate == ROOT or ROOT not in candidate.parents:
        raise ValueError("path is outside the checked-out repository")
    parts = candidate.relative_to(ROOT).parts
    if ".git" in parts or parts[0] == "agent_output" or parts[0] == ".env":
        raise ValueError("path is protected")
    if len(parts) >= 2 and parts[0] == ".github" and parts[1] == "workflows":
        raise ValueError("workflow files are protected")
    return candidate


def run_tool(action: dict) -> str:
    name = action.get("action")
    if name == "list_files":
        pattern = action.get("pattern", "*")
        return "\n".join(str(path.relative_to(ROOT)) for path in sorted(ROOT.glob(pattern)) if path.is_file())[:30000]
    if name == "read_file":
        path = safe_path(action["path"])
        return path.read_text(encoding="utf-8")[:50000]
    if name == "write_file":
        path = safe_path(action["path"])
        content = action.get("content", "")
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError("file exceeds worker size limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {path.relative_to(ROOT)}"
    if name == "run_check":
        command = action.get("command", "")
        allowed = [
            r"python3? -m unittest(?: .*)?",
            r"python3? -m pytest(?: .*)?",
            r"python3? -m compileall(?: .*)?",
            r"npm test(?: -- .*)?",
            r"npm run (?:test|build|lint)(?: -- .*)?",
            r"git (?:status|diff)(?: .*)?",
        ]
        if not any(re.fullmatch(pattern, command) for pattern in allowed):
            raise ValueError("command is outside the check allow-list")
        completed = subprocess.run(command.split(), cwd=ROOT, capture_output=True, text=True, timeout=300)
        return f"exit={completed.returncode}\n{completed.stdout[-20000:]}\n{completed.stderr[-10000:]}"
    raise ValueError(f"unsupported action: {name}")


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def main():
    if not TASK or len(TASK) > 30000:
        sys.exit("AGENT_TASK must contain 1 to 30,000 characters")
    plan = json.loads(TASK)
    plan_steps = len(plan.get("steps", []))
    worker_token_budget = min(2400, max(1000, 900 + plan_steps * 140))
    messages = [
        {
            "role": "system",
            "content": (
                "You are GPT-OSS 20B working on an already-approved repository plan. Work one step at a time. "
                "Respond with exactly one JSON object. Available actions: "
                "{\"action\":\"list_files\",\"pattern\":\"glob\"}, "
                "{\"action\":\"read_file\",\"path\":\"relative path\"}, "
                "{\"action\":\"write_file\",\"path\":\"relative path\",\"content\":\"full contents\"}, "
                "{\"action\":\"run_check\",\"command\":\"allow-listed command\"}, or "
                "{\"action\":\"finish\",\"summary\":\"...\",\"tests\":[\"...\"],\"risks\":[\"...\"]}. "
                "Stay within the approved objective, steps, constraints, and acceptance criteria. Never access secrets, "
                "the network, .git, or agent_output. Do not finish until you have inspected relevant files and run an appropriate check."
            ),
        },
        {"role": "user", "content": json.dumps(plan)},
    ]
    transcript = []
    started = time.time()
    final = None
    for step in range(1, MAX_STEPS + 1):
        response = requests.post(
            OLLAMA_URL.rstrip("/") + "/api/chat",
            json={"model": MODEL, "messages": messages, "stream": False, "options": {"num_predict": worker_token_budget, "temperature": 0.2}},
            timeout=900,
        )
        if not response.ok:
            raise RuntimeError(f"Ollama HTTP {response.status_code}: {response.text[:1000]}")
        content = response.json().get("message", {}).get("content", "")
        action = extract_json(content)
        transcript.append({"step": step, "model_action": action})
        messages.append({"role": "assistant", "content": json.dumps(action)})
        if action.get("action") == "finish":
            final = action
            break
        try:
            observation = run_tool(action)
        except Exception as error:
            observation = f"TOOL ERROR: {error}"
        transcript[-1]["observation"] = observation
        messages.append({"role": "user", "content": f"Tool result:\n{observation}"})
    if final is None:
        final = {"summary": "Worker reached its step limit.", "tests": [], "risks": ["Manual review required"]}
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = {"session_id": SESSION_ID, "model": MODEL, "plan": plan, "final": final, "steps_used": len(transcript), "duration_seconds": round(time.time() - started, 2)}
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "transcript.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    body = ["## Summary", final.get("summary", ""), "", "## Checks"]
    body.extend(f"- {item}" for item in final.get("tests", []) or ["No checks reported"])
    body.extend(["", "## Risks"])
    body.extend(f"- {item}" for item in final.get("risks", []) or ["None reported; review is still required"])
    (OUTPUT_DIR / "pull_request.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    print(final.get("summary", "Worker finished"))


if __name__ == "__main__":
    main()
