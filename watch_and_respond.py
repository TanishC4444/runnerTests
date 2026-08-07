"""
Runs on the Actions runner itself. Fetches the remote branch and reads
chunks.json directly from that fetched Git commit (using the file's blob SHA
to detect changes), and for every new entry created during the current session:
  1. waits until QUIET_SECONDS pass with no further change (so it doesn't
     respond mid-thought if you're still talking / more chunks are still
     landing)
  2. sends the text to the locally-running Qwen 2.5 3B (via Ollama)
  3. commits the reply text back to the repo

TEXT ONLY. No TTS runs here anymore -- speech happens locally on run_all.py's
side instead (see that file). That removes edge_tts as a runner dependency
entirely (pip install is now just `requests`) and removes the audio-file
push this file used to do per response.

On startup, once Ollama/the model are actually confirmed reachable, it
pushes a short "ready" signal to a separate status.json (see STATUS_FILE
below) so the local side (run_all.py) knows Qwen is genuinely ready --
rather than guessing. status.json is kept separate from chunks.json on
purpose: chunks.json is already being written independently by the local
listener, and mixing a second, differently-shaped writer into that file
would just recreate the same race condition run_all.py/live_split_on_pauses.py
had to work around.

Stops only when the job is cancelled (by you, via the cancel-run API) or
the workflow's own timeout is hit.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import threading
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
FILE_PATH = "chat/Log 1/chunks.json"
STATUS_FILE = "chat/Log 1/status.json"
API_URL = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
STATUS_API_URL = f"https://api.github.com/repos/{REPO}/contents/{STATUS_FILE}"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

# Told explicitly not to use markdown -- this used to come out as literal
# asterisks/hashes/backticks when spoken aloud on the local side. Cheaper
# to stop it at the source than to strip it after the fact every time.
SYSTEM_PREFIX = (
    "You are a voice assistant. Reply in plain spoken English only: full "
    "sentences, normal punctuation, no markdown, no asterisks, no bullet "
    "points, no headers, no code fences. Someone is going to hear this "
    "read aloud, not read it on a screen.\n\n"
)

# Ollama defaults num_predict (max generated tokens) to 128 unless told
# otherwise, which is why replies felt clipped -- this gives real replies
# room to actually finish a thought. The warm-up ping stays separately
# capped (see announce_ready) so it doesn't slow down startup.
REPLY_MAX_TOKENS = 200
GENERATE_TIMEOUT_S = 180  # CPU-only runner; keep live conversation responsive

POLL_SECONDS = 3
QUIET_SECONDS = 2  # listener already emits only completed pause-delimited chunks
GITHUB_TIMEOUT_S = 30
PUSH_RETRIES = 5
HEARTBEAT_SECONDS = 15
BRANCH = os.environ.get("GITHUB_REF_NAME", "main")
REMOTE_REF = f"refs/remotes/origin/{BRANCH}"


def gh_headers(etag=None):
    h = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        # The watcher needs the moving branch head, not a cached representation
        # from when this Actions job started.
        "Cache-Control": "no-cache",
    }
    if etag:
        h["If-None-Match"] = etag
    return h


def fetch_file():
    """Fetch the moving remote branch and return entries plus blob SHA.

    This deliberately avoids the Contents API for polling. actions/checkout
    leaves authenticated Git credentials on the runner, and passing the
    `branch:path` expression as one subprocess argument handles spaces in the
    path without any shell quoting ambiguity.
    """
    fetch = subprocess.run(
        [
            "git",
            "fetch",
            "--quiet",
            "--no-tags",
            "origin",
            f"+refs/heads/{BRANCH}:{REMOTE_REF}",
        ],
        capture_output=True,
        text=True,
        timeout=GITHUB_TIMEOUT_S,
    )
    if fetch.returncode != 0:
        detail = (fetch.stderr or fetch.stdout).strip()
        raise RuntimeError(f"git fetch origin/{BRANCH} failed: {detail}")

    object_name = f"{REMOTE_REF}:{FILE_PATH}"
    show = subprocess.run(
        ["git", "show", object_name],
        capture_output=True,
        text=True,
        timeout=GITHUB_TIMEOUT_S,
    )
    if show.returncode != 0:
        detail = (show.stderr or show.stdout).strip()
        raise RuntimeError(f"could not read {object_name}: {detail}")

    blob = subprocess.run(
        ["git", "rev-parse", object_name],
        capture_output=True,
        text=True,
        timeout=GITHUB_TIMEOUT_S,
    )
    if blob.returncode != 0:
        detail = (blob.stderr or blob.stdout).strip()
        raise RuntimeError(f"could not resolve {object_name}: {detail}")

    entries = json.loads(show.stdout) if show.stdout.strip() else []
    return entries, blob.stdout.strip()


def push_file(entries, sha, message):
    content_b64 = base64.b64encode(
        json.dumps(entries, indent=2).encode("utf-8")
    ).decode("utf-8")
    resp = requests.put(
        API_URL,
        headers=gh_headers(),
        json={
            "message": message,
            "content": content_b64,
            "sha": sha,
            "branch": BRANCH,
        },
        timeout=GITHUB_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["content"]["sha"]


def entry_key(entry: dict) -> tuple:
    """Stable-enough identity for chunks created by the local listener."""
    return (
        entry.get("datetime"),
        entry.get("raw_text"),
        entry.get("talk_seconds"),
    )


def push_response(key: tuple, reply: str):
    """Merge one response into the newest file and push it immediately.

    The microphone can add another chunk while Qwen is generating. Re-fetching
    here prevents that concurrent commit from being overwritten; retrying a
    conflict covers a chunk that lands between our fetch and PUT.
    """
    for attempt in range(1, PUSH_RETRIES + 1):
        entries, sha = fetch_file()
        target = next((entry for entry in entries if entry_key(entry) == key), None)
        if target is None:
            raise RuntimeError("chunk disappeared before its response could be saved")
        if "response" in target:
            return

        target["response"] = reply

        try:
            push_file(entries, sha, "Add Qwen response")
            return
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code not in (409, 422):
                raise
            print(
                f"  chunk file changed during response push; retrying "
                f"({attempt}/{PUSH_RETRIES})",
                flush=True,
            )

    raise RuntimeError("could not save response after repeated chunk-file conflicts")


_MARKDOWN_STRIP_RE = re.compile(r"[*_`#]+|^\s*[-•]\s+", re.MULTILINE)


def clean_reply(text: str) -> str:
    """Belt-and-suspenders cleanup in case Qwen ignores SYSTEM_PREFIX and
    emits markdown anyway -- strips the literal symbols so they never get
    spoken aloud as "asterisk" etc. on the local TTS side."""
    text = _MARKDOWN_STRIP_RE.sub("", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def ask_qwen(text: str, max_tokens: int | None = None, use_system_prefix: bool = True) -> str:
    prompt = (SYSTEM_PREFIX + text) if use_system_prefix else text
    options = {
        # Ollama's own core auto-detection is inconsistent on GitHub-hosted
        # runners (containerized cgroup CPU limits vs. host nproc can
        # disagree) -- pin it explicitly so llama.cpp actually spreads
        # across every visible core instead of guessing low.
        "num_thread": CPU_COUNT,
    }
    if max_tokens is not None:
        # caps generation length -- keeps the readiness warm-up fast on
        # a CPU-only runner; real replies pass REPLY_MAX_TOKENS instead
        options["num_predict"] = max_tokens
    payload = {"model": MODEL, "prompt": prompt, "stream": False, "options": options}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=GENERATE_TIMEOUT_S)
    if not resp.ok:
        # requests' default HTTPError only includes the status line. Ollama
        # puts the useful cause (OOM, corrupt model blob, runner crash, etc.)
        # in the response body, so preserve it in the Actions log.
        detail = resp.text.strip() or "<empty response body>"
        raise RuntimeError(
            f"Ollama generation failed with HTTP {resp.status_code}: {detail[:4000]}"
        )

    reply = resp.json().get("response", "").strip()
    if not reply:
        raise RuntimeError("Ollama returned HTTP 200 but an empty response")
    return clean_reply(reply)


def fetch_status():
    """Returns (payload_dict, sha). payload/sha are (None, None) if the
    file doesn't exist yet (first run in this repo)."""
    resp = requests.get(
        STATUS_API_URL,
        headers=gh_headers(),
        params={"ref": BRANCH},
        timeout=GITHUB_TIMEOUT_S,
    )
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
    resp = requests.put(
        STATUS_API_URL,
        headers=gh_headers(),
        json=body,
        timeout=GITHUB_TIMEOUT_S,
    )
    resp.raise_for_status()


def announce_ready():
    """Text-only ready signal now -- run_all.py speaks it locally itself.

    The warm-up call caps output at a handful of tokens (num_predict) --
    a full free-length generation on a CPU-only Actions runner can take
    well over a minute, which made this look "stuck" even though it was
    just slow. A capped call still proves the model is loaded and
    generating, in a few seconds instead of a minute-plus."""
    print("[ready] sending a capped warm-up prompt to confirm Qwen is loaded...", flush=True)
    t0 = time.time()
    try:
        ask_qwen("Say 'ready'.", max_tokens=5, use_system_prefix=False)
    except Exception as e:
        error = f"Qwen warm-up failed: {e}"
        print(f"[ready] {error}", flush=True)
        try:
            push_status(
                {
                    "ready": False,
                    "error": error,
                    "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                "Qwen watcher failed to start",
            )
        except Exception as status_error:
            print(f"[ready] failed to publish startup error: {status_error}", flush=True)
        # A broken model cannot answer real chunks either. Fail the job
        # instead of advertising a ready watcher that silently does nothing.
        raise

    print(f"[ready] warm-up call answered in {time.time() - t0:.1f}s", flush=True)

    try:
        push_status(
            {
                "ready": True,
                "ready_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "Qwen watcher ready",
        )
        print("[ready] pushed ready signal to status.json.", flush=True)
    except Exception as e:
        # Don't let a failed ready-ping take down the whole watcher --
        # chunk responses can still work even if this push fails.
        print(f"[ready] failed to push status.json (continuing anyway): {e}", flush=True)


def main():
    print(f"Watching {FILE_PATH} in {REPO}, polling every {POLL_SECONDS}s...")

    # Anything already present belongs to an earlier session. Mark it before
    # announcing readiness so this run answers only chunks recorded after the
    # user hears the ready cue instead of draining an old backlog first.
    initial_entries, last_sha = fetch_file()
    handled = {entry_key(entry) for entry in initial_entries}
    print(
        f"[startup] branch={BRANCH} chunks_sha={last_sha[:10]} "
        f"ignoring {len(handled)} pre-existing chunks",
        flush=True,
    )

    announce_ready()

    last_change_time = None
    pending_entries = None
    last_heartbeat = time.time()

    while True:
        try:
            entries, sha = fetch_file()
        except Exception as e:
            print(f"[poll] remote Git read failed; retrying: {e}", flush=True)
            time.sleep(POLL_SECONDS)
            continue

        if sha != last_sha:
            # file changed since last check
            new_count = sum(
                entry_key(entry) not in handled and "response" not in entry
                for entry in entries
            )
            print(
                f"[poll] chunks changed {last_sha[:10]} -> {sha[:10]}; "
                f"{new_count} new chunk(s)",
                flush=True,
            )
            pending_entries = entries
            last_sha = sha
            last_change_time = time.time()

        now = time.time()
        if now - last_heartbeat >= HEARTBEAT_SECONDS:
            pending_count = 0
            if pending_entries is not None:
                pending_count = sum(
                    entry_key(entry) not in handled and "response" not in entry
                    for entry in pending_entries
                )
            print(
                f"[poll] alive branch={BRANCH} chunks_sha={last_sha[:10]} "
                f"pending={pending_count}",
                flush=True,
            )
            last_heartbeat = now

        # once quiet, process anything unprocessed
        if (
            pending_entries is not None
            and last_change_time is not None
            and time.time() - last_change_time >= QUIET_SECONDS
        ):
            for entry in pending_entries:
                key = entry_key(entry)
                if key in handled or "response" in entry:
                    continue
                text = entry.get("text", "").strip()
                if not text:
                    handled.add(key)
                    continue

                t_start = time.time()
                print(f"Responding to: {text[:60]!r}", flush=True)
                try:
                    t0 = time.time()
                    reply = ask_qwen(text, max_tokens=REPLY_MAX_TOKENS)
                    generate_s = time.time() - t0
                    print(f"  [timing] generate={generate_s:.2f}s -> {reply[:60]!r}", flush=True)
                except Exception as e:
                    print(f"  generation failed: {e}", flush=True)
                    handled.add(key)
                    continue

                try:
                    t0 = time.time()
                    push_response(key, reply)
                    print(f"  [timing] git push={time.time()-t0:.2f}s "
                          f"total={time.time()-t_start:.2f}s", flush=True)
                except Exception as e:
                    print(f"  failed to save response: {e}", flush=True)
                finally:
                    handled.add(key)

            # Refresh after the batch. A new chunk may have landed while Qwen
            # was generating; keep it pending instead of accidentally treating
            # its SHA as already handled.
            latest_entries, last_sha = fetch_file()
            still_pending = any(
                entry_key(entry) not in handled
                and "response" not in entry
                and entry.get("text", "").strip()
                for entry in latest_entries
            )
            if still_pending:
                pending_entries = latest_entries
                last_change_time = time.time()
            else:
                pending_entries = None
                last_change_time = None

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()