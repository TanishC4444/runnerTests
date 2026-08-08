"""
Runs on the Actions runner itself, for the whole session now -- this is the
primary coordinator (tool-calling routing decisions AND plain conversational
replies), not a rare Groq-failure backup. Groq is gone entirely.

Protocol: control_plane.py / ollama_relay.py queue one JSON object per chat
completion into chat/Log 1/completions.json (via the GitHub Contents API,
from the local machine) and block waiting for a matching object with a
"response" key to appear. This script's only job is to notice unanswered
entries, run them through the locally-running GPT-OSS 20B (via Ollama), and
push the answer back -- one request in, one response out, no debounce
needed (unlike the old chunks.json, every entry here already represents one
complete, ready-to-answer job, not a partial speech chunk).

On startup, once Ollama/the model are actually confirmed reachable, it
pushes a short "ready" signal to status.json exactly as the old Qwen
watcher did, so run_all.py knows the coordinator is genuinely up before it
starts the microphone.

Stops only when the job is cancelled (a new session's trigger always wins,
see the workflow's concurrency group) or the workflow's own timeout is hit.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time

import requests

# GitHub Actions doesn't attach a TTY to step output, so Python buffers
# stdout by default -- prints can sit invisible for a long time instead
# of showing up as they happen. Force line buffering so the log is live.
sys.stdout.reconfigure(line_buffering=True)

CPU_COUNT = os.cpu_count() or 4
print(f"[deps] requests {requests.__version__}, {CPU_COUNT} CPU cores visible", flush=True)

TOKEN = os.environ["GH_TOKEN"]
REPO = os.environ["GITHUB_REPOSITORY"]  # "owner/repo"
FILE_PATH = "chat/Log 1/completions.json"
STATUS_FILE = "chat/Log 1/status.json"
API_URL = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
STATUS_API_URL = f"https://api.github.com/repos/{REPO}/contents/{STATUS_FILE}"

# The OpenAI-compatible endpoint (/v1/chat/completions) does not accept
# num_ctx, so a request here could never actually use more than whatever
# context Ollama defaults to (8192 for gpt-oss, per an Ollama bug/floor --
# see github.com/ollama/ollama/issues/11711). Ollama's native /api/chat
# endpoint accepts an "options" object (num_ctx, num_predict, temperature)
# and returns an equivalent message/tool_calls shape, so use that instead.
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
# gptoss_watcher.yml bakes a `gpt-oss:20b-max` tag with num_ctx=131072
# (gpt-oss's real max) via a Modelfile, so the model's own default context
# is already large; NUM_CTX below is sent per-request too as a second line
# of defense (and to make it overridable without editing the workflow).
MODEL = os.environ.get("GPTOSS_MODEL", "gpt-oss:20b-max")
NUM_CTX = int(os.environ.get("GPTOSS_NUM_CTX", "131072"))

# 280s was sized for small, low-context Groq-era completions. Large
# completions at gpt-oss:20b's full 128k context on a CPU-only runner can
# fall to roughly 9 tok/s, so a several-thousand-token reply can genuinely
# take 10+ minutes to generate. Sized generously rather than tightly --
# ollama_relay.py's own timeout_s (the local machine's patience) was raised
# to match, so a slow generation here doesn't just get abandoned as a
# relay timeout on the other end.
GENERATE_TIMEOUT_S = int(os.environ.get("GPTOSS_GENERATE_TIMEOUT_S", "3600"))
POLL_SECONDS = 1.5        # git fetch is a plain subprocess call, not API-rate-limited

# --- Self-relaunch: GitHub's 360-minute hard cap on hosted-runner jobs
# cannot be raised by any workflow setting. What can be done instead is have
# this job trigger a fresh run of itself shortly before its own
# timeout-minutes budget runs out. gptoss_watcher.yml's concurrency group
# (cancel-in-progress: true) already exists so a new session's dispatch
# always wins over a stale watcher -- this reuses exactly that mechanism:
# once the new run actually starts, GitHub cancels this one automatically,
# so no explicit self-exit is needed here. The net effect is one unbroken
# chain of watchers for as long as a session runs, with no manual restart.
JOB_TIMEOUT_MINUTES = int(os.environ.get("JOB_TIMEOUT_MINUTES", "350"))  # match timeout-minutes in gptoss_watcher.yml
RELAUNCH_MARGIN_MINUTES = 15  # dispatch the next watcher this long before the hard cutoff
RELAUNCH_AT_S = max(60, (JOB_TIMEOUT_MINUTES - RELAUNCH_MARGIN_MINUTES) * 60)
WORKFLOW_FILE = "gptoss_watcher.yml"
GITHUB_TIMEOUT_S = 30
PUSH_RETRIES = 5
HEARTBEAT_SECONDS = 15
BRANCH = os.environ.get("GITHUB_REF_NAME", "main")
REMOTE_REF = f"refs/remotes/origin/{BRANCH}"


def gh_headers():
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        # The watcher needs the moving branch head, not a cached representation
        # from when this Actions job started.
        "Cache-Control": "no-cache",
    }


def fetch_file():
    """Fetch the moving remote branch and return entries plus blob SHA.

    Deliberately avoids the Contents API for polling. actions/checkout
    leaves authenticated Git credentials on the runner, and passing the
    `branch:path` expression as one subprocess argument handles spaces in the
    path without any shell quoting ambiguity.
    """
    fetch = subprocess.run(
        ["git", "fetch", "--quiet", "--no-tags", "origin", f"+refs/heads/{BRANCH}:{REMOTE_REF}"],
        capture_output=True, text=True, timeout=GITHUB_TIMEOUT_S,
    )
    if fetch.returncode != 0:
        detail = (fetch.stderr or fetch.stdout).strip()
        raise RuntimeError(f"git fetch origin/{BRANCH} failed: {detail}")

    object_name = f"{REMOTE_REF}:{FILE_PATH}"
    show = subprocess.run(["git", "show", object_name], capture_output=True, text=True, timeout=GITHUB_TIMEOUT_S)
    if show.returncode != 0:
        # File doesn't exist yet -- normal on a brand new session before the
        # local side has queued its first request.
        return [], None

    blob = subprocess.run(["git", "rev-parse", object_name], capture_output=True, text=True, timeout=GITHUB_TIMEOUT_S)
    if blob.returncode != 0:
        detail = (blob.stderr or blob.stdout).strip()
        raise RuntimeError(f"could not resolve {object_name}: {detail}")

    entries = json.loads(show.stdout) if show.stdout.strip() else []
    return entries, blob.stdout.strip()


def push_file(entries, sha, message):
    content_b64 = base64.b64encode(json.dumps(entries, indent=2).encode("utf-8")).decode("utf-8")
    body = {"message": message, "content": content_b64, "branch": BRANCH}
    if sha:
        body["sha"] = sha
    resp = requests.put(API_URL, headers=gh_headers(), json=body, timeout=GITHUB_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()["content"]["sha"]


def push_response(request_id: str, message: dict, usage: dict):
    """Merge one response into the newest file and push it immediately.

    The local side can queue another request while GPT-OSS is generating.
    Re-fetching here prevents that concurrent commit from being overwritten;
    retrying a conflict covers a request that lands between our fetch and PUT.
    """
    for attempt in range(1, PUSH_RETRIES + 1):
        entries, sha = fetch_file()
        target = next((entry for entry in entries if entry.get("request_id") == request_id), None)
        if target is None:
            raise RuntimeError("request disappeared before its response could be saved")
        if "response" in target:
            return

        target["response"] = {"message": message, "usage": usage}

        try:
            push_file(entries, sha, "Add GPT-OSS completion response")
            return
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code not in (409, 422):
                raise
            print(f"  completions file changed during response push; retrying ({attempt}/{PUSH_RETRIES})", flush=True)

    raise RuntimeError("could not save response after repeated completions-file conflicts")


_MARKDOWN_STRIP_RE = re.compile(r"[*_`#]+|^\s*[-•]\s+", re.MULTILINE)


def clean_content(text: str) -> str:
    """Belt-and-suspenders cleanup in case GPT-OSS ignores the system
    prompt's "no markdown" instruction anyway -- strips the literal symbols
    so they never get spoken aloud as "asterisk" etc. on the local TTS side.
    Only applied to plain-text replies; tool-call messages have no content
    to clean."""
    if not text:
        return text
    text = _MARKDOWN_STRIP_RE.sub("", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def run_completion(messages: list, tools: list | None, reasoning_effort: str, max_tokens: int, temperature: float, num_ctx: int = NUM_CTX) -> tuple[dict, dict]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        # reasoning_effort isn't a native /api/chat field; gpt-oss reads its
        # reasoning level from the system/template convention Ollama wires
        # up for this model family, so pass it through options as well for
        # forward-compat, and keep sending it at the top level too in case a
        # future Ollama version reads it from there directly.
        "reasoning_effort": reasoning_effort,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    if tools:
        payload["tools"] = tools
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=GENERATE_TIMEOUT_S)
    if not resp.ok:
        detail = resp.text.strip() or "<empty response body>"
        raise RuntimeError(f"Ollama chat completion failed with HTTP {resp.status_code}: {detail[:4000]}")
    data = resp.json()
    message = data["message"]
    if message.get("content"):
        message["content"] = clean_content(message["content"])
    # Native /api/chat reports usage as prompt_eval_count/eval_count instead
    # of an OpenAI-shaped usage object -- normalize so every downstream
    # caller (control_plane.py's UsageTracker etc.) keeps working unchanged.
    usage = {
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
    }
    return message, usage


def fetch_status():
    resp = requests.get(STATUS_API_URL, headers=gh_headers(), params={"ref": BRANCH}, timeout=GITHUB_TIMEOUT_S)
    if resp.status_code == 404:
        return None, None
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    payload = json.loads(content) if content.strip() else {}
    return payload, data["sha"]


def push_status(payload: dict, message: str):
    _, sha = fetch_status()
    body = {
        "message": message,
        "content": base64.b64encode(json.dumps(payload, indent=2).encode("utf-8")).decode("utf-8"),
        "branch": BRANCH,
    }
    if sha:
        body["sha"] = sha
    resp = requests.put(STATUS_API_URL, headers=gh_headers(), json=body, timeout=GITHUB_TIMEOUT_S)
    resp.raise_for_status()


def announce_ready():
    """A tiny capped completion proves the model is loaded and generating
    in a few seconds instead of however long a full-length reply takes on a
    CPU-only runner -- a free-length generation here would make startup look
    stuck even though it's just slow."""
    print("[ready] sending a capped warm-up prompt to confirm GPT-OSS is loaded...", flush=True)
    t0 = time.time()
    try:
        run_completion([{"role": "user", "content": "Say 'ready'."}], None, "low", 5, 0.2)
    except Exception as e:
        error = f"GPT-OSS warm-up failed: {e}"
        print(f"[ready] {error}", flush=True)
        try:
            push_status({"ready": False, "error": error, "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, "GPT-OSS watcher failed to start")
        except Exception as status_error:
            print(f"[ready] failed to publish startup error: {status_error}", flush=True)
        # A broken model cannot answer real requests either. Fail the job
        # instead of advertising a ready watcher that silently does nothing.
        raise

    print(f"[ready] warm-up call answered in {time.time() - t0:.1f}s", flush=True)

    try:
        push_status({"ready": True, "ready_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, "GPT-OSS watcher ready")
        print("[ready] pushed ready signal to status.json.", flush=True)
    except Exception as e:
        print(f"[ready] failed to push status.json (continuing anyway): {e}", flush=True)


def relaunch_watcher():
    """Dispatch a fresh gptoss_watcher.yml run so a session survives past
    this job's own timeout-minutes budget. Best-effort: if this fails, the
    current watcher just keeps running until its own hard timeout -- the
    session ends there rather than never having a chance to continue, but
    generation isn't interrupted just because a relaunch attempt failed."""
    url = f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    body = {"ref": BRANCH, "inputs": {"session_id": os.environ.get("RUNNER_SESSION_ID", "")}}
    resp = requests.post(url, headers=gh_headers(), json=body, timeout=GITHUB_TIMEOUT_S)
    resp.raise_for_status()


def main():
    print(f"Watching {FILE_PATH} in {REPO}, polling every {POLL_SECONDS}s...", flush=True)

    announce_ready()

    handled: set[str] = set()
    last_heartbeat = time.time()
    job_started = time.time()
    relaunched = False

    while True:
        if not relaunched and (time.time() - job_started) >= RELAUNCH_AT_S:
            # gptoss_watcher.yml's concurrency group (cancel-in-progress:
            # true) cancels this run automatically once the new one starts --
            # no need to exit here, just keep serving requests until it does.
            print(f"[relaunch] approaching {JOB_TIMEOUT_MINUTES}min job timeout, "
                  f"dispatching a fresh watcher run...", flush=True)
            try:
                relaunch_watcher()
                relaunched = True
                print("[relaunch] dispatched successfully; this run keeps serving until superseded.", flush=True)
            except Exception as e:
                print(f"[relaunch] dispatch failed, will retry next poll: {e}", flush=True)
        try:
            entries, _sha = fetch_file()
        except Exception as e:
            print(f"[poll] remote Git read failed; retrying: {e}", flush=True)
            time.sleep(POLL_SECONDS)
            continue

        now = time.time()
        if now - last_heartbeat >= HEARTBEAT_SECONDS:
            pending = sum(1 for e in entries if e.get("request_id") not in handled and "response" not in e)
            print(f"[poll] alive branch={BRANCH} pending={pending}", flush=True)
            last_heartbeat = now

        for entry in entries:
            request_id = entry.get("request_id")
            if not request_id or request_id in handled or "response" in entry:
                continue

            print(f"[job {request_id[:8]}] {len(entry.get('messages', []))} messages, "
                  f"{len(entry.get('tools') or [])} tools offered", flush=True)
            try:
                t0 = time.time()
                message, usage = run_completion(
                    entry.get("messages", []),
                    entry.get("tools"),
                    entry.get("reasoning_effort", "low"),
                    entry.get("max_tokens", 4000),
                    entry.get("temperature", 0.2),
                    entry.get("num_ctx", NUM_CTX),
                )
                print(f"  [timing] generate={time.time() - t0:.2f}s -> "
                      f"{'tool_call' if message.get('tool_calls') else (message.get('content') or '')[:60]!r}", flush=True)
            except Exception as e:
                print(f"  generation failed: {e}", flush=True)
                handled.add(request_id)
                continue

            try:
                t0 = time.time()
                push_response(request_id, message, usage)
                print(f"  [timing] git push={time.time() - t0:.2f}s", flush=True)
            except Exception as e:
                print(f"  failed to save response: {e}", flush=True)
            finally:
                handled.add(request_id)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()