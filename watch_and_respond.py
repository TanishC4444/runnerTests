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

Stops only when the job is cancelled (by you, via the cancel-run API) or
the workflow's own timeout is hit.
"""

import base64
import json
import os
import subprocess
import time

import requests
from gtts import gTTS

TOKEN = os.environ["GH_TOKEN"]
REPO = os.environ["GITHUB_REPOSITORY"]  # "owner/repo"
FILE_PATH = "chat/Log 1/chunks.json"
API_URL = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"

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


def ask_qwen(text: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": text, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def tts_base64(text: str) -> str:
    tmp_path = "/tmp/reply.mp3"
    gTTS(text=text).save(tmp_path)
    with open(tmp_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    print(f"Watching {FILE_PATH} in {REPO}, polling every {POLL_SECONDS}s...")

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

                print(f"Responding to: {text[:60]!r}")
                try:
                    reply = ask_qwen(text)
                    audio_b64 = tts_base64(reply)
                except Exception as e:
                    print(f"Failed on entry: {e}")
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