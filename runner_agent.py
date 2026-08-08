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


# Defaults to the 8192-ctx base tag if the workflow didn't override it, but
# agent_worker.yml sets this to gpt-oss:20b-max (num_ctx=131072 baked in via
# Modelfile) plus AGENT_NUM_CTX below, sent per-request too.
MODEL = os.environ.get("AGENT_MODEL", "gpt-oss:20b")
NUM_CTX = int(os.environ.get("AGENT_NUM_CTX", "131072"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
TASK = os.environ.get("AGENT_TASK", "").strip()
SESSION_ID = os.environ.get("AGENT_SESSION_ID", "dashboard")
ROOT = Path.cwd().resolve()
OUTPUT_DIR = ROOT / "agent_output"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
CONTINUE_MARKER = OUTPUT_DIR / "CONTINUE"
# Was a hard 24-step ceiling meant to bound worst-case cost/latency on a
# metered path. That's no longer the real constraint -- the time budget
# below is. Steps are cheap (one JSON action each); let a genuinely large
# task use as many as it needs within its time budget instead of getting cut
# off arbitrarily early.
MAX_STEPS = 100000
# Wall-clock budget for a single run of this script, independent of
# timeout-minutes in the workflow (which is GitHub's hard, non-adjustable
# cap). Leaving a run before the hard cap gives the "Publish changes, or
# checkpoint and continue" workflow step time to actually commit/push/
# dispatch before a force-kill would just drop that work on the floor.
TIME_BUDGET_S = int(os.environ.get("AGENT_TIME_BUDGET_S", str(300 * 60)))
RESUME_REF = os.environ.get("AGENT_RESUME_REF", "").strip()
# Was 512KB, sized against a small per-completion token budget that couldn't
# usefully write much more than that in one step anyway. Raised so writing a
# genuinely large generated file isn't blocked by an arbitrary size cap.
MAX_FILE_BYTES = 8_000_000
GITHUB_API = "https://api.github.com"
# The workflow's default GITHUB_TOKEN is repo-scoped and can never create a
# new repository under the account -- that's an account-level permission.
# REPO_ADMIN_TOKEN is a separate PAT (classic, `repo` scope, or fine-grained
# with account "Administration: write") passed in only if you've added it as
# a repo secret. If it's absent, create_repository fails loudly instead of
# silently doing nothing.
REPO_ADMIN_TOKEN = os.environ.get("REPO_ADMIN_TOKEN", "")
ALLOWED_OWNERS = [o.strip() for o in os.environ.get("AGENT_ALLOWED_OWNERS", "").split(",") if o.strip()]
# Map of token_env-name -> value for MCP servers whose config uses
# auth: "static_token" (mirrors control_plane.py's MCPRegistry). Populate as
# a repo secret shaped like {"NOTION_TOKEN": "..."} and pass it through as
# MCP_STATIC_TOKENS in agent_worker.yml. github_oauth-authed servers reuse
# REPO_ADMIN_TOKEN instead, same as the dashboard reuses its GitHub session.
try:
    MCP_STATIC_TOKENS = json.loads(os.environ.get("MCP_STATIC_TOKENS", "{}") or "{}")
except json.JSONDecodeError:
    MCP_STATIC_TOKENS = {}
_mcp_sessions: dict[str, tuple[requests.Session, dict, str]] = {}


def _mcp_session(server: str, servers_cfg: dict) -> tuple[requests.Session, dict, str]:
    if server in _mcp_sessions:
        return _mcp_sessions[server]
    cfg = servers_cfg.get(server)
    if not cfg:
        raise ValueError(f"MCP server {server!r} was not included in this task's briefing")
    if cfg.get("transport") != "streamable_http":
        raise ValueError("only streamable_http MCP servers are supported")
    auth_mode = cfg.get("auth")
    token = None
    if auth_mode == "github_oauth":
        token = REPO_ADMIN_TOKEN
    elif auth_mode == "static_token":
        token = MCP_STATIC_TOKENS.get(cfg.get("token_env") or "")
    if auth_mode and not token:
        raise ValueError(f"MCP server {server!r} needs a token but none was provided for this run")
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "MCP-Protocol-Version": "2025-03-26",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session = requests.Session()
    init = session.post(
        cfg["url"], headers=headers,
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "runner-agent", "version": "1.0"}}},
        timeout=30,
    )
    if not init.ok:
        raise ValueError(f"MCP initialize failed with HTTP {init.status_code}: {init.text[:500]}")
    session_id = init.headers.get("Mcp-Session-Id")
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    session.post(cfg["url"], headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, timeout=15)
    _mcp_sessions[server] = (session, headers, cfg["url"])
    return _mcp_sessions[server]


def _decode_mcp_response(response: requests.Response) -> dict:
    if not response.content:
        return {}
    if "text/event-stream" in response.headers.get("Content-Type", ""):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        return {}
    return response.json()


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


def run_tool(action: dict, plan: dict) -> str:
    name = action.get("action")
    if name == "list_files":
        pattern = action.get("pattern", "*")
        return "\n".join(str(path.relative_to(ROOT)) for path in sorted(ROOT.glob(pattern)) if path.is_file())[:150000]
    if name == "read_file":
        path = safe_path(action["path"])
        return path.read_text(encoding="utf-8")[:300000]
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
        completed = subprocess.run(command.split(), cwd=ROOT, capture_output=True, text=True, timeout=1800)
        return f"exit={completed.returncode}\n{completed.stdout[-100000:]}\n{completed.stderr[-50000:]}"
    if name == "create_repository":
        if not REPO_ADMIN_TOKEN:
            raise ValueError(
                "create_repository requires the REPO_ADMIN_TOKEN secret to be configured "
                "(the default GITHUB_TOKEN cannot create repositories; see agent_worker.yml)"
            )
        repo_name = action.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", repo_name):
            raise ValueError("invalid repository name")
        body = {
            "name": repo_name,
            "description": action.get("description", ""),
            "private": bool(action.get("private", True)),
            "auto_init": bool(action.get("auto_init", True)),
        }
        response = requests.post(
            f"{GITHUB_API}/user/repos",
            headers={"Authorization": f"Bearer {REPO_ADMIN_TOKEN}", "Accept": "application/vnd.github+json"},
            json=body,
            timeout=30,
        )
        if not response.ok:
            raise ValueError(f"GitHub returned HTTP {response.status_code}: {response.text[:500]}")
        created = response.json()
        owner = created.get("owner", {}).get("login")
        if ALLOWED_OWNERS and owner not in ALLOWED_OWNERS:
            raise ValueError(f"created repository owner {owner!r} is outside AGENT_ALLOWED_OWNERS -- refusing to report it as success")
        return f"created {created.get('full_name')}: {created.get('html_url')}"
    if name == "mcp_call":
        qualified_name = action.get("tool", "")
        parts = qualified_name.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            raise ValueError("tool must be a qualified name like mcp__server__tool_name, taken from this task's mcp_tools list")
        _, server, tool_name = parts
        known = {t["qualified_name"] for t in plan.get("mcp_tools", [])}
        if qualified_name not in known:
            raise ValueError("that MCP tool wasn't included in this task's briefing -- only call tools listed in mcp_tools")
        denied = ("delete", "remove", "destroy", "force", "secret", "token", "transfer", "archive")
        if any(word in tool_name.lower() for word in denied):
            raise ValueError("this MCP tool is blocked by the destructive-action policy")
        session, headers, url = _mcp_session(server, plan.get("mcp_servers", {}))
        response = session.post(url, headers=headers, json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": tool_name, "arguments": action.get("arguments", {})}}, timeout=120)
        if not response.ok:
            raise ValueError(f"MCP tools/call failed with HTTP {response.status_code}: {response.text[:500]}")
        payload = _decode_mcp_response(response)
        if payload.get("error"):
            raise ValueError(f"MCP tool failed: {payload['error']}")
        return json.dumps(payload.get("result", {}))[:20000]
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
    # 55,000 chars was a conservative guess against a small context window.
    # gpt-oss:20b's real ~128k token (~500k char) context leaves much more
    # room for a genuinely large, detailed plan -- capped well under that so
    # there's still room left for messages/tool results in the same context.
    if not TASK or len(TASK) > 300000:
        sys.exit("AGENT_TASK must contain 1 to 300,000 characters")
    plan = json.loads(TASK)
    plan_steps = len(plan.get("steps", []))
    # Was capped at 2400 completion tokens per step to bound cost/latency
    # under a metered path. Raised substantially now that the constraint is
    # CPU generation speed against a time budget, not a token quota -- a
    # step that legitimately needs to write a large file or reason through a
    # complex diff can use the room.
    worker_token_budget = min(24000, max(4000, 1500 * max(plan_steps, 1)))

    skill_text = "\n\n".join(
        f"### Skill: {s.get('title') or s.get('name')}\n{s.get('instructions', '')}"
        for s in plan.get("skills", []) if s.get("instructions")
    ) or "None matched this task."
    mcp_tools = plan.get("mcp_tools", [])
    mcp_text = "\n".join(
        f"- {t['qualified_name']} ({'read-only' if t.get('read_only') else 'WRITE'}): {t.get('description', '')}"
        for t in mcp_tools
    ) or "None available for this task."

    messages = [
        {
            "role": "system",
            "content": (
                "You are GPT-OSS 20B working on an already-approved repository plan. Work one step at a time. "
                "Respond with exactly one JSON object. Available actions: "
                "{\"action\":\"list_files\",\"pattern\":\"glob\"}, "
                "{\"action\":\"read_file\",\"path\":\"relative path\"}, "
                "{\"action\":\"write_file\",\"path\":\"relative path\",\"content\":\"full contents\"}, "
                "{\"action\":\"run_check\",\"command\":\"allow-listed command\"}, "
                "{\"action\":\"create_repository\",\"name\":\"...\",\"description\":\"...\",\"private\":true,\"auto_init\":true} "
                "(only if the plan explicitly calls for a new repository), "
                "{\"action\":\"mcp_call\",\"tool\":\"mcp__server__tool_name\",\"arguments\":{...}} "
                "(tool must be one of the qualified names listed below in Available MCP tools -- do not invent one), or "
                "{\"action\":\"finish\",\"summary\":\"...\",\"tests\":[\"...\"],\"risks\":[\"...\"]}. "
                "Stay within the approved objective, steps, constraints, and acceptance criteria. File and check tools "
                "are restricted to this checked-out repository and cannot touch .git, agent_output, or workflow files. "
                "Network access is limited to the GitHub API (for create_repository) and the MCP servers listed below "
                "(for mcp_call) -- nothing else. Do not finish until you have inspected relevant files and run an "
                "appropriate check.\n\n"
                "## Skills relevant to this task\n" + skill_text + "\n\n"
                "## Available MCP tools (call only these, exactly as named)\n" + mcp_text
            ),
        },
        {"role": "user", "content": json.dumps({k: v for k, v in plan.items() if k not in ("skills", "mcp_tools", "mcp_servers")})},
    ]
    transcript = []
    step_offset = 0

    # Resume from a previous run's checkpoint if agent_worker.yml checked one
    # out (resume_ref set + checkpoint.json present on that branch). Replaces
    # the freshly-built messages/transcript above with where the prior run
    # left off, so this run continues the same conversation instead of
    # restarting the task from scratch.
    if RESUME_REF and CHECKPOINT_FILE.exists():
        checkpoint = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        messages = checkpoint["messages"]
        transcript = checkpoint["transcript"]
        step_offset = checkpoint["steps_used"]
        print(f"[resume] loaded checkpoint from {RESUME_REF}: {step_offset} steps already used", flush=True)

    started = time.time()
    final = None
    ran_out_of_time = False
    for step in range(step_offset + 1, step_offset + MAX_STEPS + 1):
        if time.time() - started > TIME_BUDGET_S:
            print(f"[checkpoint] time budget ({TIME_BUDGET_S}s) reached after {step - 1} steps this run; "
                  f"checkpointing for continuation instead of pushing further.", flush=True)
            ran_out_of_time = True
            break
        response = requests.post(
            OLLAMA_URL.rstrip("/") + "/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": worker_token_budget, "temperature": 0.2, "num_ctx": NUM_CTX},
            },
            # Raised from 900s: at full 128k context with a large
            # worker_token_budget, CPU-only generation can fall to ~9 tok/s,
            # so a single step's completion can legitimately take a while.
            timeout=3600,
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
            observation = run_tool(action, plan)
        except Exception as error:
            observation = f"TOOL ERROR: {error}"
        transcript[-1]["observation"] = observation
        messages.append({"role": "user", "content": f"Tool result:\n{observation}"})

    OUTPUT_DIR.mkdir(exist_ok=True)

    if ran_out_of_time and final is None:
        # Not done -- write a checkpoint instead of a result/PR body.
        # agent_worker.yml's publish step looks for CONTINUE_MARKER and, if
        # present, commits this checkpoint to a resume branch and dispatches
        # a fresh run of this workflow pointed at it instead of opening a PR.
        checkpoint = {
            "session_id": SESSION_ID,
            "messages": messages,
            "transcript": transcript,
            "steps_used": len(transcript),
        }
        CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        CONTINUE_MARKER.write_text("continue\n", encoding="utf-8")
        print(f"Checkpointed after {len(transcript)} total steps; a continuation run will be dispatched.")
        return

    if final is None:
        final = {"summary": "Worker reached its step limit.", "tests": [], "risks": ["Manual review required"]}
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