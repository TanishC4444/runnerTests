"""
Secondary responder: reaches the same GPT-OSS 20B coordinator control_plane.py
uses (via ollama_relay.py -> the gptoss_watcher.yml GitHub Actions runner),
but directly from this process instead of through the local control plane's
HTTP server.

Called from live_split_on_pauses.py's worker thread only when the primary
path -- POSTing to the local control plane's /api/voice-command endpoint --
fails, e.g. the dashboard process isn't running, crashed mid-session, or is
unreachable for some other local reason. It is not a different brain, just a
different way to reach the same one: both paths end up going through the
exact same relay file and the exact same remote watcher, so this exists for
resilience against the control-plane *process* being unavailable, not
against the model/watcher itself being slow or down. If the watcher is what's
slow or down, this path will be exactly as slow or down as the primary one.

Maintains its own short conversation history across calls within this
process, independent of whatever history the control plane keeps.
"""

import os

import ollama_relay

MODEL = os.environ.get("LOCAL_RESPONDER_MODEL", "gpt-oss:20b")
MAX_TOKENS = 200

SYSTEM_PROMPT = (
    "You are a voice assistant. Reply in plain spoken English only: full "
    "sentences, normal punctuation, no markdown, no asterisks, no bullet "
    "points, no headers, no code fences. Someone is going to hear this "
    "read aloud, not read it on a screen. Keep replies conversational and "
    "reasonably brief."
)

# Trim so the prompt doesn't grow unbounded over a long session.
MAX_HISTORY_MESSAGES = 20


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
last_usage = {}


def ask_local_model(user_text: str) -> str:
    """Returns the reply text, or raises on failure (ollama_relay.RelayError
    if GH_TOKEN is missing or the watcher never answers, plus whatever the
    underlying request can raise). Maintains conversation history across
    calls within this process so it's a real back-and-forth, not one-shot
    Q&A."""
    history.add("user", user_text)
    lowered = user_text.lower()
    adaptive_max_tokens = 120
    if any(word in lowered for word in ("explain", "compare", "why", "how", "steps")):
        adaptive_max_tokens = MAX_TOKENS
    elif len(user_text.split()) <= 8:
        adaptive_max_tokens = 90

    try:
        completion = ollama_relay.request_completion(
            model=MODEL,
            messages=history.as_payload(),
            reasoning_effort="low",
            max_tokens=adaptive_max_tokens,
            temperature=0.7,
        )
    except Exception:
        history.drop_last()  # don't leave a dangling unanswered turn
        raise

    global last_usage
    last_usage = completion.get("usage") or {}
    reply = (completion["message"].get("content") or "").strip()
    if not reply:
        history.drop_last()
        raise RuntimeError("GPT-OSS returned an empty message")

    history.add("assistant", reply)
    return reply
