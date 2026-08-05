"""
Runs on the Actions runner itself. Polls the repo's chunks.json every
second via the GitHub Contents API (using ETags so unchanged polls don't
cost API-rate-limit budget), and for every new entry:
  1. waits until 5s pass with no further change (so it doesn't respond
     mid-thought if you're still talking / more chunks are still landing)
  2. sends the text to the locally-running Qwen 2.5 3B (via Ollama)
  3. converts the reply to speech (gTTS, saved as mp3, base64-embedded
     in the JSON so it survives as one file — no binary blobs to juggle)
  4. commits the updated file back to the repo

On startup, once Ollama/the model are actually confirmed reachable, it
also pushes a short verbal "ready" cue to a separate status.json (see
STATUS_FILE below) so the local side (run_all.py) can play it and only
start listening once Qwen is genuinely ready -- rather than guessing.
status.json is kept separate from chunks.json on purpose: chunks.json is
already being written independently by the local listener, and mixing a
second, differently-shaped writer into that file would just recreate the
same race condition run_all.py/live_split_on_pauses.py had to work around.

Stops only when the job is cancelled (by you, via the cancel-run API) or
the workflow's own timeout is hit.
"""

import base64
import json
import os
import subprocess
import sys
import time

import requests
from gtts import gTTS

# GitHub Actions doesn't attach a TTY to step output, so Python buffers
# stdout by default -- prints can sit invisible for a long time instead
# of showing up as they happen. Force line buffering so the log is live.
sys.stdout.reconfigure(line_buffering=True)

print(f"[deps] requests {requests.__version__}, gTTS import OK", flush=True)

TOKEN = os.environ["GH_TOKEN"]
REPO = os.environ["GITHUB_REPOSITORY"]  # "owner/repo"
FILE_PATH = "chat/Log 1/chunks.json"
STATUS_FILE = "chat/Log 1/status.json"
API_URL = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
STATUS_API_URL = f"https://api.github.com/repos/{REPO}/contents/{STATUS_FILE}"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

POLL_SECONDS = 1
QUIET_SECONDS = 5  # required pause before we respond


def gh_headers(etag=None):
    h = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    if etag:
        h["If-None-Match"] = etag
    return h


def fetch_file(etag=None):
    """Returns (entries, sha, etag) or (None, None, etag) if unchanged (304)."""
    resp = requests.get(API_URL, headers=gh_headers(etag))
    if resp.status_code == 304:
        return None, None, etag
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    entries = json.loads(content) if content.strip() else []
    return entries, data["sha"], resp.headers.get("ETag")


def push_file(entries, sha, message):
    content_b64 = base64.b64encode(
        json.dumps(entries, indent=2).encode("utf-8")
    ).decode("utf-8")
    resp = requests.put(
        API_URL,
        headers=gh_headers(),
        json={"message": message, "content": content_b64, "sha": sha},
    )
    resp.raise_for_status()
    return resp.json()["content"]["sha"]


def ask_qwen(text: str, max_tokens: int | None = None) -> str:
    payload = {"model": MODEL, "prompt": text, "stream": False}
    if max_tokens is not None:
        # caps generation length -- keeps the readiness warm-up fast on
        # a CPU-only runner; real replies still run uncapped
        payload["options"] = {"num_predict": max_tokens}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def tts_base64(text: str) -> str:
    tmp_path = "/tmp/reply.mp3"
    gTTS(text=text).save(tmp_path)
    with open(tmp_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def fetch_status():
    """Returns (payload_dict, sha). payload/sha are (None, None) if the
    file doesn't exist yet (first run in this repo)."""
    resp = requests.get(STATUS_API_URL, headers=gh_headers())
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
    }
    if sha:
        body["sha"] = sha
    resp = requests.put(STATUS_API_URL, headers=gh_headers(), json=body)
    resp.raise_for_status()


def announce_ready():
    """Verbal cue: TTS a short line and push it to status.json.

    The warm-up call caps output at a handful of tokens (num_predict) --
    a full free-length generation on a CPU-only Actions runner can take
    well over a minute, which made this look "stuck" even though it was
    just slow. A capped call still proves the model is loaded and
    generating, in a few seconds instead of a minute-plus."""
    print("[ready] sending a capped warm-up prompt to confirm Qwen is loaded...", flush=True)
    try:
        t0 = time.time()
        ask_qwen("Say 'ready'.", max_tokens=5)
        print(f"[ready] warm-up call answered in {time.time() - t0:.1f}s", flush=True)
        audio_b64 = tts_base64("I'm ready, go ahead.")
        print("[ready] TTS generated.", flush=True)
    except Exception as e:
        print(f"[ready] warm-up call failed, pushing text-only ready signal: {e}", flush=True)
        audio_b64 = None

    try:
        push_status(
            {
                "ready": True,
                "ready_audio_b64": audio_b64,
                "ready_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "Qwen watcher ready",
        )
        print("[ready] pushed verbal ready signal to status.json.", flush=True)
    except Exception as e:
        # Don't let a failed ready-ping take down the whole watcher --
        # chunk responses can still work even if this push fails.
        print(f"[ready] failed to push status.json (continuing anyway): {e}", flush=True)


def main():
    print(f"Watching {FILE_PATH} in {REPO}, polling every {POLL_SECONDS}s...")
    announce_ready()

    etag = None
    last_seen_count = None
    last_change_time = None
    pending_entries = None
    pending_sha = None

    while True:
        entries, sha, etag = fetch_file(etag)

        if entries is not None:
            # file changed since last check
            pending_entries = entries
            pending_sha = sha
            last_change_time = time.time()
            if last_seen_count is None:
                last_seen_count = len(entries)

        # once quiet, process anything unprocessed
        if (
            pending_entries is not None
            and last_change_time is not None
            and time.time() - last_change_time >= QUIET_SECONDS
        ):
            changed = False
            for entry in pending_entries:
                if "response" in entry:
                    continue
                text = entry.get("text", "").strip()
                if not text:
                    continue

                print(f"Responding to: {text[:60]!r}", flush=True)
                try:
                    t0 = time.time()
                    reply = ask_qwen(text)
                    print(f"  generated in {time.time() - t0:.1f}s: {reply[:60]!r}", flush=True)
                    audio_b64 = tts_base64(reply)
                except Exception as e:
                    print(f"Failed on entry: {e}", flush=True)
                    continue

                entry["response"] = reply
                entry["response_audio_b64"] = audio_b64
                changed = True

            if changed:
                pending_sha = push_file(
                    pending_entries, pending_sha, "Add Qwen response + TTS audio"
                )
                # re-fetch etag/sha baseline after our own push
                entries, sha, etag = fetch_file(None)

            pending_entries = None
            last_change_time = None

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()