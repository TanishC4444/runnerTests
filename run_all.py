"""
Run this one script. It:
  1. Resets the "ready" status from any previous session.
  2. Triggers the GitHub Actions workflow (Qwen watcher, cloud side).
  3. Waits for Qwen to actually announce it's ready (a real warmed-up
     call, not just "the process started"), and plays that verbal cue
     through your speakers.
  4. Only then starts the local mic listener as a subprocess.
  5. Sits idle — if you haven't made a sound, nothing happens on either
     side, both just wait.
  6. The moment you speak and pause, the listener writes + pushes a
     chunk, the cloud watcher picks it up, and eventually writes the
     response + audio back into chunks.json.
  7. Ctrl+C here cancels the cloud run AND kills the local listener.

Setup:
    export GH_TOKEN="your_new_token"   # repo + workflow scope
"""

import base64
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time

import requests

TOKEN = os.environ.get("GH_TOKEN")
if not TOKEN:
    sys.exit("Set GH_TOKEN first (export GH_TOKEN=...)")

REPO = "TanishC4444/runnerTests"
WORKFLOW_FILE = "qwen_watcher.yml"
BASE = f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW_FILE}"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}

STATUS_FILE = "chat/Log 1/status.json"
STATUS_API_URL = f"https://api.github.com/repos/{REPO}/contents/{STATUS_FILE}"

CHUNKS_LOG_FILE = os.path.join("chat", "Log 1", "chunks.json")

LISTENER_SCRIPT = "live_split_on_pauses.py"
READY_TIMEOUT_S = 240  # generous: cold model pull + Ollama boot can be slow
RESPONSE_POLL_SECONDS = 3


def trigger_workflow():
    resp = requests.post(f"{BASE}/dispatches", headers=HEADERS, json={"ref": "main"})
    resp.raise_for_status()
    print("[cloud] workflow triggered.")


def get_latest_run_id():
    resp = requests.get(f"{BASE}/runs?per_page=1", headers=HEADERS)
    resp.raise_for_status()
    runs = resp.json()["workflow_runs"]
    return runs[0]["id"] if runs else None


def cancel_run(run_id):
    url = f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/cancel"
    resp = requests.post(url, headers=HEADERS)
    print(f"[cloud] cancel requested (status {resp.status_code}).")


def fetch_status():
    """Returns (payload_dict, sha), or (None, None) if status.json
    doesn't exist yet in the repo."""
    resp = requests.get(STATUS_API_URL, headers=HEADERS)
    if resp.status_code == 404:
        return None, None
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    payload = json.loads(content) if content.strip() else {}
    return payload, data["sha"]


def reset_status():
    """Clear any stale ready-flag left over from a previous session
    before triggering a new run, so we don't mistake an old signal for
    this one."""
    payload, sha = fetch_status()
    if payload is None:
        return  # nothing to reset yet
    body = {
        "message": "Reset ready status for new session",
        "content": base64.b64encode(json.dumps({"ready": False}, indent=2).encode("utf-8")).decode("utf-8"),
        "sha": sha,
    }
    resp = requests.put(STATUS_API_URL, headers=HEADERS, json=body)
    resp.raise_for_status()


def play_audio(path: str):
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", path], check=True)
        elif system == "Linux":
            for player in (["mpg123", path], ["ffplay", "-nodisp", "-autoexit", path], ["aplay", path]):
                try:
                    subprocess.run(player, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            print("[audio] no player found (tried mpg123/ffplay/aplay), skipping playback")
        elif system == "Windows":
            os.startfile(path)  # opens default player
        else:
            print(f"[audio] unrecognized platform {system!r}, skipping playback")
    except Exception as e:
        print(f"[audio] playback failed: {e}")


def wait_for_ready(timeout_s: int = READY_TIMEOUT_S):
    """Poll status.json until the cloud watcher announces it's ready,
    then play its verbal cue. Raises if startup fails or times out so the
    microphone never records against a watcher that cannot answer.

    Prints an elapsed-time heartbeat every 15s -- the wait can legitimately
    take a couple minutes (Ollama install + model load on a CPU-only
    runner), and a silent wait is indistinguishable from a stuck one."""
    print("[cloud] waiting for Qwen to finish loading...")
    start = time.time()
    deadline = start + timeout_s
    last_heartbeat = start
    while time.time() < deadline:
        payload, _ = fetch_status()
        if payload and payload.get("error"):
            raise RuntimeError(payload["error"])
        if payload and payload.get("ready"):
            audio_b64 = payload.get("ready_audio_b64")
            if audio_b64:
                tmp_path = os.path.join(tempfile.gettempdir(), "qwen_ready.mp3")
                with open(tmp_path, "wb") as f:
                    f.write(base64.b64decode(audio_b64))
                play_audio(tmp_path)
            else:
                print("[cloud] Qwen is ready (no audio cue was included).")
            return
        now = time.time()
        if now - last_heartbeat >= 15:
            print(f"[cloud] ...still waiting ({int(now - start)}s elapsed). "
                  f"Check the 'Run watcher loop' step's live log if this runs long.")
            last_heartbeat = now
        time.sleep(2)
    raise RuntimeError(
        "Timed out waiting for Qwen to pass its model warm-up; the microphone was not started."
    )


def load_chunks():
    if os.path.exists(CHUNKS_LOG_FILE):
        with open(CHUNKS_LOG_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def git_pull_quiet():
    """Best-effort pull -- if it fails (e.g. a lock held by the listener's
    own concurrent pull), just skip this cycle and try again shortly."""
    try:
        subprocess.run(
            ["git", "pull", "--rebase", "--autostash"],
            capture_output=True, timeout=20,
        )
    except Exception:
        pass


def response_watcher_loop(stop_event: threading.Event):
    """Background thread: periodically checks chunks.json for replies
    that have landed from the cloud watcher and plays them out loud --
    this is what actually makes Qwen "talk back" instead of you having
    to go open the file yourself. Runs independently of the listener's
    own git activity, which only happens when you're the one speaking."""
    # seed with anything already answered from a previous session so we
    # don't replay old responses on startup
    played = {e.get("datetime") for e in load_chunks() if "response" in e}

    while not stop_event.is_set():
        git_pull_quiet()
        for entry in load_chunks():
            key = entry.get("datetime")
            if key in played or "response" not in entry:
                continue
            played.add(key)

            reply = entry.get("response", "")
            print(f"[talkback] {reply[:60]!r}")

            audio_b64 = entry.get("response_audio_b64")
            if audio_b64:
                tmp_path = os.path.join(tempfile.gettempdir(), "qwen_reply.mp3")
                with open(tmp_path, "wb") as f:
                    f.write(base64.b64decode(audio_b64))
                play_audio(tmp_path)

        stop_event.wait(RESPONSE_POLL_SECONDS)


def main():
    reset_status()
    trigger_workflow()

    run_id = None
    for _ in range(10):
        time.sleep(2)
        run_id = get_latest_run_id()
        if run_id:
            break

    if not run_id:
        sys.exit("[cloud] could not find the triggered run.")
    print(f"[cloud] run id {run_id} is live.")

    try:
        wait_for_ready()
    except RuntimeError as e:
        print(f"[cloud] startup failed: {e}")
        cancel_run(run_id)
        return

    print("[local] starting mic listener...")
    listener = subprocess.Popen([sys.executable, LISTENER_SCRIPT])

    stop_event = threading.Event()
    responder = threading.Thread(target=response_watcher_loop, args=(stop_event,), daemon=True)
    responder.start()

    print("\nReady. Idle until you speak. Ctrl+C to stop everything.\n")

    try:
        while True:
            time.sleep(1)
            if listener.poll() is not None:
                print("[local] listener exited on its own, stopping.")
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        if listener.poll() is None:
            listener.terminate()
            try:
                listener.wait(timeout=5)
            except subprocess.TimeoutExpired:
                listener.kill()
        print("[local] listener stopped.")
        cancel_run(run_id)


if __name__ == "__main__":
    main()
