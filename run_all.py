"""
Run this one script. It:
  1. Triggers the GitHub Actions workflow (Qwen watcher, cloud side).
  2. Starts the local mic listener as a subprocess.
  3. Sits idle — if you haven't made a sound, nothing happens on either
     side, both just wait.
  4. The moment you speak and pause, the listener writes + pushes a
     chunk, the cloud watcher picks it up, and eventually writes the
     response + audio back into chunks.json.
  5. Ctrl+C here cancels the cloud run AND kills the local listener.

Setup:
    export GH_TOKEN="your_new_token"   # repo + workflow scope
"""

import os
import subprocess
import sys
import time

import requests

TOKEN = os.environ.get("GH_TOKEN")
if not TOKEN:
    sys.exit("Set GH_TOKEN first (export GH_TOKEN=...)")

REPO = "TanishC4444/runnerTests"
WORKFLOW_FILE = "qwen_watcher.yml"
BASE = f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW_FILE}"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}

LISTENER_SCRIPT = "live_split_on_pauses.py"


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


def main():
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

    print("[local] starting mic listener...")
    listener = subprocess.Popen([sys.executable, LISTENER_SCRIPT])

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