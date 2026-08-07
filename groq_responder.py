"""
Primary responder: Groq-hosted Qwen 3.6 27B. Fast enough to answer in-process,
no round trip through GitHub/Actions -- the cloud Qwen watcher
(qwen_watcher.yml) becomes pure backup.

Called from live_split_on_pauses.py's worker thread, right where a chunk's
final text becomes available (before it's pushed to chunks.json). On
success, the caller attaches our reply to the chunk BEFORE pushing it, so
run_all.py's existing playback watcher (which just looks for a "response"
field it hasn't spoken yet) plays it out with zero changes on that side.

Fallback behavior:
  - HTTP 429 (rate limited) -> raise GroqRateLimited. Caller pushes the
    chunk WITHOUT a response and calls ensure_qwen_fallback(), which
    triggers qwen_watcher.yml (same call trigger_and_control.py makes) --
    only once per session, since the watcher stays alive and keeps
    answering everything unanswered once it's up.
  - Any other failure after retries -> raise. Caller treats it the same
    way (fall back for this chunk) so you always get an answer.
  - watch_and_respond.py already skips any entry that already has a
    "response" key, so there's no double-answer risk even if Groq recovers
    mid-session while the Qwen watcher is still up.

The model can still be overridden with GROQ_MODEL, but the first/default
connection uses Groq's qwen/qwen3.6-27b model.
"""

import os
import time

import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT_S = 20
GROQ_MAX_RETRIES = 2   # transient network hiccups only -- never retried on 429
GROQ_MAX_TOKENS = 200

SYSTEM_PROMPT = (
    "You are a voice assistant. Reply in plain spoken English only: full "
    "sentences, normal punctuation, no markdown, no asterisks, no bullet "
    "points, no headers, no code fences. Someone is going to hear this "
    "read aloud, not read it on a screen. Keep replies conversational and "
    "reasonably brief."
)

# Trim so the prompt doesn't grow unbounded over a long session.
MAX_HISTORY_MESSAGES = 20

# GitHub side, for triggering the Qwen backup workflow -- same repo/workflow
# trigger_and_control.py uses.
GH_TOKEN = os.environ.get("GH_TOKEN")
SESSION_ID = os.environ.get("RUNNER_SESSION_ID")
REPO = os.environ.get("GITHUB_REPOSITORY", "TanishC4444/runnerTests")
WORKFLOW_FILE = "qwen_watcher.yml"
WORKFLOW_BASE = f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW_FILE}"
GH_HEADERS = {
    "Authorization": f"Bearer {GH_TOKEN}" if GH_TOKEN else "",
    "Accept": "application/vnd.github+json",
}


class GroqRateLimited(Exception):
    """Specifically an HTTP 429 -- tells the caller to stop hammering
    Groq and fall back, as opposed to a one-off network blip."""


class _History:
    def __init__(self):
        self.messages = []

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        overflow = len(self.messages) - MAX_HISTORY_MESSAGES
        if overflow > 0:
            del self.messages[:overflow]

    def drop_last(self):
        if self.messages:
            self.messages.pop()

    def as_payload(self):
        return [{"role": "system", "content": SYSTEM_PROMPT}] + self.messages


history = _History()


def ask_groq(user_text: str) -> str:
    """Returns the reply text, or raises GroqRateLimited / requests
    exceptions on failure. Maintains conversation history across calls
    within this process so it's a real back-and-forth, not one-shot Q&A."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    history.add("user", user_text)
    payload = {
        "model": GROQ_MODEL,
        "messages": history.as_payload(),
        "max_tokens": GROQ_MAX_TOKENS,
        "temperature": 0.7,
    }
    # This is a real-time voice path, so use Qwen's non-thinking mode. It
    # avoids spending latency/tokens on hidden reasoning and keeps the reply
    # in message.content for immediate playback. Preserve GROQ_MODEL as a
    # useful generic override by only sending the Qwen-specific option when
    # a Qwen model is selected.
    if GROQ_MODEL.startswith("qwen/"):
        payload["reasoning_effort"] = "none"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                GROQ_URL, headers=headers, json=payload, timeout=GROQ_TIMEOUT_S
            )
        except requests.RequestException as e:
            last_error = e
            time.sleep(1.5 * attempt)
            continue

        if resp.status_code == 429:
            history.drop_last()  # don't leave a dangling unanswered turn
            raise GroqRateLimited(resp.text[:500])

        if not resp.ok:
            last_error = RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:500]}")
            time.sleep(1.5 * attempt)
            continue

        data = resp.json()
        reply = data["choices"][0]["message"]["content"].strip()
        if not reply:
            last_error = RuntimeError("Groq returned an empty message")
            time.sleep(1.5 * attempt)
            continue

        history.add("assistant", reply)
        return reply

    history.drop_last()
    raise last_error or RuntimeError("Groq request failed with no captured error")


# --- Qwen fallback trigger -------------------------------------------------

_fallback_state = {"triggered": False}


def _get_latest_qwen_run():
    resp = requests.get(f"{WORKFLOW_BASE}/runs?per_page=1", headers=GH_HEADERS, timeout=15)
    resp.raise_for_status()
    runs = resp.json().get("workflow_runs", [])
    return runs[0] if runs else None


def _trigger_qwen_workflow():
    dispatch = {"ref": "main"}
    if SESSION_ID:
        dispatch["inputs"] = {"session_id": SESSION_ID}
    resp = requests.post(
        f"{WORKFLOW_BASE}/dispatches", headers=GH_HEADERS, json=dispatch, timeout=15
    )
    resp.raise_for_status()


def ensure_qwen_fallback(reason: str):
    """Idempotent per process: fires qwen_watcher.yml at most once per
    session. Safe to call every time Groq fails -- after the first call
    it just no-ops, since the watcher stays alive and keeps answering
    whatever's unanswered on its own poll loop."""
    if _fallback_state["triggered"]:
        return True
    if not GH_TOKEN:
        print(f"[fallback] Groq unavailable ({reason}) and GH_TOKEN is not set "
              f"-- cannot trigger the Qwen backup. This chunk will go unanswered.")
        return False
    try:
        run = _get_latest_qwen_run()
        if run and run.get("status") in ("queued", "in_progress"):
            print(f"[fallback] Groq unavailable ({reason}); Qwen watcher already "
                  f"running (run {run['id']}) -- it'll pick this chunk up.")
        else:
            _trigger_qwen_workflow()
            print(f"[fallback] Groq unavailable ({reason}); triggered qwen_watcher.yml as backup.")
        _fallback_state["triggered"] = True
        return True
    except Exception as e:
        print(f"[fallback] failed to trigger Qwen backup: {e}")
        # Leave this false so a later failed primary request can retry the
        # backup launch instead of permanently disabling fallback for the
        # remainder of the session after one transient GitHub/API error.
        return False
