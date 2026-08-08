"""Shared relay between this machine and the always-on GitHub Actions
gpt-oss:20b watcher (gptoss_watcher.yml / watch_and_respond.py).

Groq is gone. Every chat completion this app needs -- tool-calling routing
decisions, plain conversational replies, tool-result summaries -- is queued
here and answered by that remote Ollama instance. Communication is one JSON
file in the repo (chat/Log 1/completions.json), read and written through the
GitHub Contents API, exactly the mechanism chunks.json already proved out.
Only the payload shape is new: messages + tool schemas instead of just
spoken text, because the watcher is now the coordinator, not a plain
Q&A backup.

request_completion() blocks (poll-sleep) until the watcher answers or the
timeout elapses. That is a real, felt latency cost compared to a direct
Groq call -- there is no way around a git-relay round trip taking multiple
seconds at minimum, and gpt-oss:20b generating on a GitHub-hosted CPU
runner (no GPU) on top of that. This trade removes Groq's per-minute token
ceiling entirely; it does not make a single turn faster.
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid

import requests

REPO = os.environ.get("GITHUB_REPOSITORY", "TanishC4444/runnerTests")
BRANCH = os.environ.get("GITHUB_REF_NAME", "main")
FILE_PATH = "chat/Log 1/completions.json"
API_URL = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
GITHUB_TIMEOUT_S = 30
PUSH_RETRIES = 8


class RelayError(RuntimeError):
    """Raised for anything relay-specific: no token, no watcher, timed out."""


def _token() -> str:
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RelayError("GH_TOKEN is not set -- required to relay requests to the gpt-oss watcher")
    return token


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Cache-Control": "no-cache",
    }


def _fetch() -> tuple[list[dict], str | None]:
    resp = requests.get(API_URL, headers=_headers(), params={"ref": BRANCH}, timeout=GITHUB_TIMEOUT_S)
    if resp.status_code == 404:
        return [], None
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    entries = json.loads(content) if content.strip() else []
    return entries, data["sha"]


def _push(entries: list[dict], sha: str | None, message: str) -> None:
    body = {
        "message": message,
        "content": base64.b64encode(json.dumps(entries, indent=2).encode("utf-8")).decode("utf-8"),
        "branch": BRANCH,
    }
    if sha:
        body["sha"] = sha
    resp = requests.put(API_URL, headers=_headers(), json=body, timeout=GITHUB_TIMEOUT_S)
    resp.raise_for_status()


def _append_request(entry: dict) -> None:
    """Merge-and-retry against concurrent writers, same pattern
    live_split_on_pauses.py already uses for chunks.json."""
    for attempt in range(1, PUSH_RETRIES + 1):
        entries, sha = _fetch()
        entries.append(entry)
        try:
            _push(entries, sha, "Queue completion request for gpt-oss watcher")
            return
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code not in (409, 422):
                raise
            time.sleep(0.5 * attempt)
    raise RelayError("could not queue completion request after repeated conflicts")


def request_completion(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    reasoning_effort: str = "low",
    max_tokens: int = 300,
    temperature: float = 0.2,
    timeout_s: int = 240,
    poll_s: float = 2.0,
) -> dict:
    """Queue one chat-completion job for the gpt-oss watcher and block until
    it answers. Returns {"message": {...}, "usage": {...}} -- the same shape
    an OpenAI-compatible HTTP call already returned, so every caller that
    used to POST to Groq directly can call this instead with no change to
    how it reads the result."""
    request_id = uuid.uuid4().hex
    entry = {
        "request_id": request_id,
        "created_at": time.time(),
        "model": model,
        "messages": messages,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        entry["tools"] = tools
    _append_request(entry)

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(poll_s)
        try:
            entries, _ = _fetch()
        except requests.RequestException:
            continue  # transient read failure -- just retry on the next tick
        match = next((item for item in entries if item.get("request_id") == request_id), None)
        if match and "response" in match:
            return match["response"]
    raise RelayError(
        f"The gpt-oss watcher did not answer within {timeout_s}s. "
        "Check that gptoss_watcher.yml is running (Actions tab) and that it pulled gpt-oss:20b successfully."
    )
