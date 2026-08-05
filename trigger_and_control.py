"""
Local-side control script. Run this alongside your mic listener.

Triggers the qwen_watcher.yml workflow, then waits. On Ctrl+C, it cancels
the in-progress workflow run via the GitHub API before exiting.

Setup:
    export GH_TOKEN="ghp_..."       # your PAT, scoped to repo + workflow
    pip install requests --break-system-packages

The token is read from an environment variable ONLY. Never hardcode it
here or anywhere else that could end up committed.
"""

import os
import sys
import time

import requests

TOKEN = os.environ.get("GH_TOKEN")
if not TOKEN:
    sys.exit("Set GH_TOKEN as an environment variable first (export GH_TOKEN=...)")

REPO = "TanishC4444/runnerTests"
WORKFLOW_FILE = "qwen_watcher.yml"

BASE = f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW_FILE}"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
}


def trigger_workflow():
    resp = requests.post(
        f"{BASE}/dispatches",
        headers=HEADERS,
        json={"ref": "main"},
    )
    resp.raise_for_status()
    print("Workflow triggered.")


def get_latest_run_id():
    resp = requests.get(f"{BASE}/runs?per_page=1", headers=HEADERS)
    resp.raise_for_status()
    runs = resp.json()["workflow_runs"]
    return runs[0]["id"] if runs else None


def cancel_run(run_id):
    url = f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/cancel"
    resp = requests.post(url, headers=HEADERS)
    if resp.status_code == 202:
        print(f"Cancel requested for run {run_id}.")
    else:
        print(f"Cancel request returned {resp.status_code}: {resp.text}")


def main():
    trigger_workflow()

    # give GitHub a moment to register the run before we can find its ID
    time.sleep(5)
    run_id = None
    for _ in range(10):
        run_id = get_latest_run_id()
        if run_id:
            break
        time.sleep(2)

    if not run_id:
        sys.exit("Could not find the triggered run. Check the Actions tab manually.")

    print(f"Workflow running (run id {run_id}). Ctrl+C to cancel it.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCancelling workflow run...")
        cancel_run(run_id)


if __name__ == "__main__":
    main()