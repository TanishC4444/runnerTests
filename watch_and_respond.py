"""
Runs on the Actions runner itself. Fetches the remote branch and reads
chunks.json directly from that fetched Git commit (using the file's blob SHA
to detect changes), and for every new entry created during the current session:
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

import asyncio
import base64
import json
import os
import subprocess
import sys
import time

import requests
import edge_tts

# GitHub Actions doesn't attach a TTY to step output, so Python buffers
# stdout by default -- prints can sit invisible for a long time instead
# of showing up as they happen. Force line buffering so the log is live.
sys.stdout.reconfigure(line_buffering=True)

print(f"[deps] requests {requests.__version__}, edge_tts import OK", flush=True)

# Jarvis-ish: calm, deep, British male voice. Full voice list: `edge-tts --list-voices`.
# Other reasonable picks: "en-GB-ThomasNeural" (younger/brisker, more clipped RP),
# "en-GB-RyanNeural" (previous default, deeper/slower delivery).
TTS_VOICE = "en-GB-ThomasNeural"
TTS_RATE = "+32%"   # bumped further; drop back toward +18% if it starts sounding rushed/clipped

TOKEN = os.environ["GH_TOKEN"]
REPO = os.environ["GITHUB_REPOSITORY"]  # "owner/repo"
FILE_PATH = "chat/Log 1/chunks.json"
STATUS_FILE = "chat/Log 1/status.json"
AUDIO_DIR = "chat/Log 1/audio"  # one small file per response, NOT embedded in chunks.json
API_URL = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
STATUS_API_URL = f"https://api.github.com/repos/{REPO}/contents/{STATUS_FILE}"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

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


def entry_id(key: tuple) -> str:
    """Filesystem-safe id derived from entry_key, used as the audio filename."""
    import hashlib
    return hashlib.sha1("|".join(str(k) for k in key).encode("utf-8")).hexdigest()[:16]


def push_audio_file(filename: str, audio_bytes: bytes) -> str:
    """Uploads ONE small audio file (not the whole chunks.json array).

    This is the fix for chunks.json growing unbounded: previously every
    response embedded its full base64 mp3 INSIDE the shared entries array,
    so every subsequent push (new chunk OR new response) re-serialized and
    re-uploaded the ENTIRE conversation's audio history every single time --
    strictly growing payload size per exchange. Each response's audio now
    gets its own tiny file; chunks.json only ever stores a path string.
    Returns the relative path stored in the entry.
    """
    path = f"{AUDIO_DIR}/{filename}"
    url = f"https://api.github.com/repos/{REPO}/contents/{path}"
    resp = requests.put(
        url,
        headers=gh_headers(),
        json={
            "message": "Add response audio",
            "content": base64.b64encode(audio_bytes).decode("utf-8"),
            "branch": BRANCH,
        },
        timeout=GITHUB_TIMEOUT_S,
    )
    resp.raise_for_status()
    return path


def push_response(key: tuple, reply: str, audio_path: str | None):
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
        if audio_path:
            # path only -- the actual bytes already live in their own small
            # file (see push_audio_file), not embedded here.
            target["response_audio_path"] = audio_path

        try:
            push_file(entries, sha, "Add Qwen response + TTS audio")
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


def ask_qwen(text: str, max_tokens: int | None = None) -> str:
    payload = {"model": MODEL, "prompt": text, "stream": False}
    if max_tokens is not None:
        # caps generation length -- keeps the readiness warm-up fast on
        # a CPU-only runner; real replies pass REPLY_MAX_TOKENS instead
        payload["options"] = {"num_predict": max_tokens}
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
    return reply


async def _edge_tts_save(text: str, path: str) -> None:
    communicate = edge_tts.Communicate(text, voice=TTS_VOICE, rate=TTS_RATE)
    await communicate.save(path)


def tts_bytes(text: str, lang: str = "en") -> bytes:
    """lang kept in the signature so call sites don't need to change;
    voice/rate/accent are controlled by TTS_VOICE/TTS_RATE above instead."""
    tmp_path = f"/tmp/reply_{lang}.mp3"
    asyncio.run(_edge_tts_save(text, tmp_path))
    with open(tmp_path, "rb") as f:
        return f.read()


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
    """Verbal cue: TTS a short line and push it to status.json.

    The warm-up call caps output at a handful of tokens (num_predict) --
    a full free-length generation on a CPU-only Actions runner can take
    well over a minute, which made this look "stuck" even though it was
    just slow. A capped call still proves the model is loaded and
    generating, in a few seconds instead of a minute-plus."""
    print("[ready] sending a capped warm-up prompt to confirm Qwen is loaded...", flush=True)
    t0 = time.time()
    try:
        ask_qwen("Say 'ready'.", max_tokens=5)
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
        audio_b64 = base64.b64encode(tts_bytes("I'm ready, go ahead.", lang="en")).decode("utf-8")
        print("[ready] TTS generated.", flush=True)
    except Exception as e:
        # Speech is optional; a TTS outage should not disguise a healthy
        # local model as a failed model startup.
        print(f"[ready] TTS failed, using text-only ready signal: {e}", flush=True)
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

                print(f"Responding to: {text[:60]!r}", flush=True)
                try:
                    t0 = time.time()
                    reply = ask_qwen(text, max_tokens=REPLY_MAX_TOKENS)
                    print(f"  generated in {time.time() - t0:.1f}s: {reply[:60]!r}", flush=True)
                except Exception as e:
                    print(f"  generation failed: {e}", flush=True)
                    handled.add(key)
                    continue

                audio_path = None
                try:
                    audio_bytes = tts_bytes(reply, lang="en")
                    filename = f"{entry_id(key)}.mp3"
                    audio_path = push_audio_file(filename, audio_bytes)
                    print(f"  audio pushed separately: {audio_path}", flush=True)
                except Exception as e:
                    print(f"  TTS/audio push failed; saving text response only: {e}", flush=True)

                try:
                    push_response(key, reply, audio_path)
                    print("  response pushed", flush=True)
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