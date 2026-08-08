"""
Run this one script. It:
  1. Starts the local mic listener (live_split_on_pauses.py) immediately --
     no waiting on the cloud. Groq (Qwen 3.6 27B) is the primary brain now and
     answers in-process on that side; the GitHub Actions Qwen watcher is
     backup only, and live_split_on_pauses.py triggers it itself, lazily,
     the first time Groq comes back rate-limited.
  2. Sits idle — if you haven't made a sound, nothing happens.
  3. The moment you speak and pause, the listener answers (via Groq, or
     via the Qwen backup if Groq is down) and writes text + response into
     chunks.json in one push.
  4. This script's response watcher just plays back whatever "response"
     shows up in chunks.json, same as before -- it doesn't care which
     brain produced it.
  5. Ctrl+C here kills the local listener and cancels the Qwen run only
     if the fallback ever actually triggered one.

Setup:
    export GH_TOKEN="your_new_token"     # repo + workflow scope
    export GROQ_API_KEY="your_groq_key"  # primary responder, read by live_split_on_pauses.py
"""

from __future__ import annotations

import asyncio
import base64
import http.server
import json
import mimetypes
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    import fcntl
except ImportError:  # Windows fallback; fcntl is available on macOS/Linux.
    fcntl = None

import requests
import edge_tts
from control_plane import ControlPlane, ControlPlaneError


def load_local_env(path: Path | None = None):
    """Load simple KEY=VALUE entries without overwriting exported values."""
    env_path = path or (Path(__file__).resolve().parent / ".env")
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_local_env()

# Speech now happens HERE, locally, not on the Actions runner. The runner
# only ever sends text; this machine turns it into audio and plays it. That
# cuts out the whole audio-file-over-git round trip for every single reply.
# Voice list: `edge-tts --list-voices`. Alternatives: "en-GB-RyanNeural"
# (deeper/slower), "en-GB-SoniaNeural" (female RP).
TTS_VOICE = "en-GB-ThomasNeural"
TTS_RATE = "+20%"   # dial toward +30% for snappier, back toward +10% if it sounds rushed

_MARKDOWN_STRIP_RE = re.compile(r"[*_`#]+|^\s*[-•]\s+", re.MULTILINE)


def clean_for_speech(text: str) -> str:
    """Strip markdown symbols before they get spoken as literal words
    ("asterisk", "pound sign", etc.). watch_and_respond.py already asks
    Qwen not to produce markdown and does its own cleanup server-side --
    this is the second line of defense, cheap insurance either way."""
    text = _MARKDOWN_STRIP_RE.sub("", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


async def _edge_tts_save(text: str, path: str) -> None:
    communicate = edge_tts.Communicate(text, voice=TTS_VOICE, rate=TTS_RATE)
    await communicate.save(path)


def speak_locally(text: str, tmp_name: str) -> str | None:
    """Synthesizes text to a local temp mp3 and returns its path, or None
    if TTS failed (caller should treat speech as optional, not fatal)."""
    cleaned = clean_for_speech(text)
    if not cleaned:
        return None
    tmp_path = os.path.join(tempfile.gettempdir(), tmp_name)
    try:
        asyncio.run(_edge_tts_save(cleaned, tmp_path))
        return tmp_path
    except Exception as e:
        print(f"[tts] local synthesis failed: {e}")
        return None


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def speak_locally_pipelined(text: str, session_start_ts: float, tmp_prefix: str = "qwen_reply"):
    """Splits the reply into sentences and pipelines synth + playback:
    while sentence N is playing, sentence N+1 is already being synthesized
    on a background thread, instead of "synthesize the whole reply, then
    play the whole reply" -- cuts time-to-first-sound on longer, multi-
    sentence replies (which is most of them, per your logs).

    Falls back to one-shot synthesis for single-sentence replies, since
    pipelining a single chunk has no benefit and just adds complexity.
    """
    cleaned = clean_for_speech(text)
    if not cleaned:
        return
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    if len(sentences) <= 1:
        t0 = time.time()
        tmp_path = speak_locally(text, f"{tmp_prefix}.mp3")
        print(f"  [timing] local TTS synth={time.time()-t0:.2f}s")
        if tmp_path:
            play_audio_interruptible(tmp_path, session_start_ts=session_start_ts)
        return

    # Synthesize sentence 0 synchronously (nothing to overlap it with yet),
    # then kick off sentence 1's synthesis in the background immediately
    # -- it renders while sentence 0 plays, so it's ready the instant
    # playback catches up instead of causing a gap.
    next_audio: dict[int, str | None] = {}
    next_ready = threading.Event()

    def _synth_ahead(idx: int, sentence: str):
        next_audio[idx] = speak_locally(sentence, f"{tmp_prefix}_{idx}.mp3")
        next_ready.set()

    t0 = time.time()
    current_path = speak_locally(sentences[0], f"{tmp_prefix}_0.mp3")
    print(f"  [timing] first-sentence TTS synth={time.time()-t0:.2f}s (of {len(sentences)} sentences)")

    for i, sentence in enumerate(sentences[1:], start=1):
        next_ready.clear()
        threading.Thread(target=_synth_ahead, args=(i, sentence), daemon=True).start()

        if current_path:
            finished = play_audio_interruptible(current_path, session_start_ts=session_start_ts)
            if not finished:
                return  # user barged in -- don't keep queuing more sentences

        next_ready.wait(timeout=10)  # sentence i's synth should already be done or close to it
        current_path = next_audio.get(i)

    if current_path:
        play_audio_interruptible(current_path, session_start_ts=session_start_ts)

TOKEN = os.environ.get("GH_TOKEN")

REPO = "TanishC4444/runnerTests"
WORKFLOW_FILE = "qwen_watcher.yml"
BASE = f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW_FILE}"
HEADERS = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

STATUS_FILE = "chat/Log 1/status.json"
STATUS_API_URL = f"https://api.github.com/repos/{REPO}/contents/{STATUS_FILE}"

CHUNKS_LOG_FILE = os.path.join("chat", "Log 1", "chunks.json")

LISTENER_SCRIPT = "live_split_on_pauses.py"
READY_TIMEOUT_S = 240  # generous: cold model pull + Ollama boot can be slow
RESPONSE_POLL_SECONDS = 1.5  # git pull, not an API call -- safe to poll tighter
LOCAL_RESPONSE_POLL_SECONDS = 0.1
GIT_LOCK_FILE = os.path.join(tempfile.gettempdir(), "runnerTests_git_sync.lock")

# Same path live_split_on_pauses.py writes to. Local-only IPC, never touches
# Git -- this is what makes barge-in and the dashboard possible without
# changing anything about how the cloud side works.
LOCAL_STATUS_FILE = os.path.join(tempfile.gettempdir(), "jarvis_local_status.json")
LOCAL_RESPONSE_FILE = os.path.join(tempfile.gettempdir(), "jarvis_local_responses.jsonl")
DASHBOARD_PORT = 8765
DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
# How aggressively the playback watcher checks "did the user start talking
# yet". 80ms feels effectively instant without busy-looping the CPU.
BARGE_IN_POLL_S = 0.08


def _read_local_status() -> dict:
    try:
        with open(LOCAL_STATUS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_local_status(**fields):
    data = _read_local_status()
    data.update(fields)
    try:
        tmp_path = LOCAL_STATUS_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, LOCAL_STATUS_FILE)
    except Exception:
        pass


@contextmanager
def git_sync_lock():
    """Serialize Git operations with the microphone subprocess."""
    with open(GIT_LOCK_FILE, "a+") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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


def _players_for_platform(path: str):
    system = platform.system()
    if system == "Darwin":
        return [["afplay", path]]
    if system == "Linux":
        return [["mpg123", path], ["ffplay", "-nodisp", "-autoexit", path], ["aplay", path]]
    if system == "Windows":
        # ffplay is the only one of these that gives us a killable child
        # process on Windows; os.startfile() opens a detached default
        # player we have no handle to, so barge-in can't stop it.
        return [["ffplay", "-nodisp", "-autoexit", path]]
    return []


def play_audio_interruptible(path: str, session_start_ts: float) -> bool:
    """Plays audio locally, but kills it the instant the mic hears you speak.

    Returns True if playback finished on its own, False if it was cut short
    by barge-in. `session_start_ts` filters out stale/old speech-onset
    events already sitting in the status file from before this reply started.
    """
    proc = None
    for player_cmd in _players_for_platform(path):
        try:
            proc = subprocess.Popen(
                player_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            break
        except FileNotFoundError:
            continue

    if proc is None:
        print("[audio] no usable player found, skipping playback")
        return True

    try:
        while proc.poll() is None:
            status = _read_local_status()
            if (
                status.get("mic_state") == "speech"
                and status.get("mic_state_ts", 0) > session_start_ts
            ):
                print("[barge-in] you started talking -- cutting playback")
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return False
            time.sleep(BARGE_IN_POLL_S)
        return True
    except Exception as e:
        print(f"[audio] playback failed: {e}")
        try:
            proc.kill()
        except Exception:
            pass
        return True


# --- Dashboard: a small local HTTP server so you can watch Jarvis's state
# (listening / thinking / speaking) live in a browser tab. Pure read-only
# view of LOCAL_STATUS_FILE -- no Git, no polling the cloud.

DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Jarvis</title>
<style>
  body { background:#05070c; color:#cfe8ff; font-family:'SF Mono',Consolas,monospace;
         display:flex; flex-direction:column; align-items:center; padding-top:8vh; }
  #orb { width:140px; height:140px; border-radius:50%; margin-bottom:28px;
         background:radial-gradient(circle at 35% 30%, #8fd8ff, #0a84ff 55%, #002347);
         box-shadow:0 0 40px 10px rgba(10,132,255,0.35); transition:box-shadow .25s, transform .25s; }
  #orb.thinking { background:radial-gradient(circle at 35% 30%, #ffe28f, #ff9d0a 55%, #4a2900);
                  box-shadow:0 0 40px 10px rgba(255,157,10,0.4); animation:pulse 1s infinite; }
  #orb.speaking { animation:pulse .5s infinite; box-shadow:0 0 55px 16px rgba(10,132,255,0.55); }
  #orb.muted { background:#2a2f3a; box-shadow:none; }
  @keyframes pulse { 0%,100%{transform:scale(1);} 50%{transform:scale(1.08);} }
  #state { font-size:1.3em; letter-spacing:.15em; text-transform:uppercase; color:#7fb8ff; margin-bottom:36px; }
  .row { width:min(640px,88vw); margin-bottom:18px; }
  .label { font-size:.75em; letter-spacing:.1em; text-transform:uppercase; color:#4a6a8f; margin-bottom:6px; }
  .text { font-size:1.05em; line-height:1.5; min-height:1.5em; color:#e6f2ff; }
</style></head>
<body>
  <div id="orb"></div>
  <div id="state">idle</div>
  <div class="row"><div class="label">You said</div><div class="text" id="user_text">&mdash;</div></div>
  <div class="row"><div class="label">Jarvis</div><div class="text" id="reply_text">&mdash;</div></div>
<script>
async function tick() {
  try {
    const r = await fetch('/status', {cache: 'no-store'});
    const s = await r.json();
    const orb = document.getElementById('orb');
    const stateEl = document.getElementById('state');
    let label = s.conv_state || 'idle';
    if (s.mic_state === 'muted') label = 'muted';
    orb.className = label;
    stateEl.textContent = label;
    document.getElementById('user_text').textContent = s.last_user_text || '\\u2014';
    document.getElementById('reply_text').textContent = s.last_reply_text || '\\u2014';
  } catch (e) {}
  setTimeout(tick, 250);
}
tick();
</script>
</body></html>"""


class _DashboardHandler(http.server.BaseHTTPRequestHandler):
    control_plane: ControlPlane | None = None

    def log_message(self, *args):
        pass  # keep the console clean; the dashboard doesn't need per-request logs

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 1_000_000:
            raise ControlPlaneError("Request body is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def _serve_asset(self, request_path: str):
        relative = "index.html" if request_path in ("", "/") else request_path.lstrip("/")
        candidate = (DASHBOARD_DIR / relative).resolve()
        if DASHBOARD_DIR.resolve() not in candidate.parents and candidate != DASHBOARD_DIR.resolve():
            self.send_error(404)
            return
        if not candidate.is_file():
            candidate = DASHBOARD_DIR / "index.html"
        body = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/status":
                return self._send_json(_read_local_status())
            if parsed.path == "/api/snapshot":
                return self._send_json(self.control_plane.snapshot(_read_local_status()))
            if parsed.path.startswith("/api/sessions/") and parsed.path.endswith("/messages"):
                session_id = parsed.path.split("/")[3]
                return self._send_json({"messages": self.control_plane.sessions.messages(session_id)})
            if parsed.path == "/api/github/overview":
                query = parse_qs(parsed.query)
                return self._send_json(self.control_plane.github_overview((query.get("owner") or [None])[0], (query.get("repo") or [None])[0]))
            if parsed.path.startswith("/api/mcp/") and parsed.path.endswith("/tools"):
                name = parsed.path.split("/")[3]
                return self._send_json({"tools": self.control_plane.mcp.list_tools(name, refresh=True)})
            return self._serve_asset(parsed.path)
        except (ControlPlaneError, requests.RequestException, ValueError) as e:
            return self._send_json({"error": str(e)}, 400)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/sessions":
                return self._send_json(self.control_plane.create_session(payload.get("title", "New session")), 201)
            if parsed.path == "/api/commands":
                return self._send_json(self.control_plane.submit(payload.get("session_id", ""), payload.get("text", "")))
            if parsed.path == "/api/voice-command":
                result = self.control_plane.submit_voice(payload.get("text", ""))
                return self._send_json(result)
            if parsed.path.startswith("/api/approvals/"):
                approval_id = parsed.path.split("/")[3]
                return self._send_json(self.control_plane.resolve_approval(approval_id, bool(payload.get("approve"))))
            if parsed.path == "/api/github/oauth/start":
                return self._send_json(self.control_plane.auth.start_device_flow(payload.get("client_id", "")))
            if parsed.path == "/api/github/oauth/poll":
                return self._send_json(self.control_plane.auth.poll_device_flow())
            if parsed.path == "/api/github/oauth/disconnect":
                self.control_plane.auth.disconnect()
                return self._send_json({"status": "disconnected"})
            if parsed.path == "/api/input-mode":
                mode = payload.get("mode")
                if mode not in ("voice", "text"):
                    raise ControlPlaneError("Input mode must be voice or text")
                _write_local_status(
                    input_mode=mode,
                    mic_state="muted" if mode == "text" else "listening",
                    mic_state_ts=time.time(),
                )
                return self._send_json({"mode": mode})
            if parsed.path.startswith("/api/mcp/") and parsed.path.endswith("/toggle"):
                name = parsed.path.split("/")[3]
                return self._send_json(self.control_plane.set_mcp_enabled(name, bool(payload.get("enabled"))))
            if parsed.path.startswith("/api/skills/") and parsed.path.endswith("/toggle"):
                name = parsed.path.split("/")[3]
                return self._send_json(self.control_plane.set_skill_enabled(name, bool(payload.get("enabled"))))
            if parsed.path == "/api/skills":
                return self._send_json(self.control_plane.save_skill(payload), 201)
            if parsed.path.startswith("/api/skills/") and parsed.path.endswith("/save"):
                name = parsed.path.split("/")[3]
                return self._send_json(self.control_plane.save_skill(payload, original_name=name))
            if parsed.path.startswith("/api/skills/") and parsed.path.endswith("/delete"):
                name = parsed.path.split("/")[3]
                self.control_plane.delete_skill(name)
                return self._send_json({"status": "deleted"})
            if parsed.path == "/api/mcp":
                return self._send_json(self.control_plane.save_mcp_server(payload.get("name", ""), payload), 201)
            if parsed.path.startswith("/api/mcp/") and parsed.path.endswith("/save"):
                name = parsed.path.split("/")[3]
                return self._send_json(self.control_plane.save_mcp_server(payload.get("name") or name, payload, original_name=name))
            if parsed.path.startswith("/api/mcp/") and parsed.path.endswith("/delete"):
                name = parsed.path.split("/")[3]
                self.control_plane.delete_mcp_server(name)
                return self._send_json({"status": "deleted"})
            return self._send_json({"error": "Unknown endpoint"}, 404)
        except (ControlPlaneError, requests.RequestException, ValueError, KeyError) as e:
            return self._send_json({"error": str(e)}, 400)


def start_dashboard_server(control_plane: ControlPlane) -> http.server.ThreadingHTTPServer:
    _DashboardHandler.control_plane = control_plane
    server = http.server.ThreadingHTTPServer(("127.0.0.1", DASHBOARD_PORT), _DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[dashboard] http://localhost:{DASHBOARD_PORT}")
    return server


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
            tmp_path = speak_locally("I'm ready, go ahead.", "qwen_ready.mp3")
            if tmp_path:
                play_audio_interruptible(tmp_path, session_start_ts=time.time())
            else:
                print("[cloud] Qwen is ready (local TTS unavailable, text-only).")
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


def load_local_responses() -> list[dict]:
    """Read primary replies queued directly by the microphone process.

    This local path makes Groq speech independent of Git synchronization.
    The queue is session-scoped and response_watcher_loop de-duplicates it
    against the durable chunks log using the chunk datetime/response ID.
    """
    try:
        with open(LOCAL_RESPONSE_FILE, "r", encoding="utf-8") as response_file:
            events = []
            for line in response_file:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("response_id") and event.get("text"):
                    events.append(event)
            return events
    except FileNotFoundError:
        return []


def _queue_local_response(response_id: str, text: str, user_text: str = "", usage: dict | None = None):
    """Write to the same local IPC file live_split_on_pauses.py's own
    _queue_local_response uses. response_watcher_loop below already polls
    this file and speaks anything new in it -- so this one write is all it
    takes to make a GPT-OSS completion talk out loud, with zero changes to
    the playback side. `logged=True` and an empty user_text keep it out of
    record_voice_turn -- this isn't a conversational Q&A pair."""
    event = {"response_id": response_id, "text": text, "user_text": user_text, "usage": usage or {}, "logged": True}
    try:
        with open(LOCAL_RESPONSE_FILE, "a", encoding="utf-8") as response_file:
            response_file.write(json.dumps(event) + "\n")
            response_file.flush()
    except OSError as e:
        print(f"[gptoss] could not queue completion speech: {e}")


GPTOSS_POLL_SECONDS = 20


def gptoss_completion_watcher(stop_event: threading.Event, control_plane: ControlPlane):
    """Polls GitHub Actions for every GPT-OSS run this session has dispatched
    (single or fanned-out parallel subtasks) and, the moment one finishes,
    drops a message into its dashboard session and queues a short spoken
    line -- so approving a delegation doesn't mean going to check GitHub
    yourself later to find out what happened.

    `dispatch_runs[].reported` (set here) is also what the dashboard itself
    polls via /api/snapshot to fire its own toast notification -- so this is
    the one place completion actually gets detected, for both surfaces."""
    while not stop_event.is_set():
        with control_plane.lock:
            approvals = list(control_plane.approvals.values())
        for approval in approvals:
            if approval.get("tool") != "delegate_to_gptoss" or approval.get("status") != "completed":
                continue
            pending_runs = [r for r in (approval.get("dispatch_runs") or []) if not r.get("reported")]
            if not pending_runs:
                continue
            owner, repo = approval.get("dispatch_owner"), approval.get("dispatch_repo")
            if not owner or not repo:
                continue
            try:
                workflow_runs = control_plane.github.execute("github_list_workflow_runs", {"owner": owner, "repo": repo})
            except Exception as e:
                print(f"[gptoss] could not poll workflow runs: {e}")
                continue
            for run_entry in pending_runs:
                match = next((wr for wr in workflow_runs if run_entry["dispatch_session_id"] in (wr.get("display_title") or "")), None)
                if not match or match.get("status") != "completed":
                    continue
                run_entry["reported"] = True
                conclusion = match.get("conclusion") or "unknown"
                ok = conclusion == "success"
                detail = (
                    f"GPT-OSS finished \"{run_entry['objective']}\" -- {conclusion}. "
                    f"Check {owner}/{repo} for the pull request. {match.get('html_url', '')}"
                )
                control_plane.sessions.append(approval["session_id"], {"kind": "message", "role": "assistant", "source": "gptoss", "content": detail})
                spoken = f"GPT-OSS {'finished' if ok else 'stopped, with a problem, on'} the task: {run_entry['objective']}."
                _queue_local_response(f"gptoss-{approval['id']}-{run_entry['dispatch_session_id']}", spoken)
                print(f"[gptoss] {detail}")
        stop_event.wait(GPTOSS_POLL_SECONDS)


def git_pull_quiet():
    """Best-effort pull -- if it fails (e.g. a lock held by the listener's
    own concurrent pull), just skip this cycle and try again shortly."""
    try:
        with git_sync_lock():
            subprocess.run(
                ["git", "pull", "--rebase", "--autostash"],
                capture_output=True, timeout=20,
            )
    except Exception:
        pass


def response_watcher_loop(stop_event: threading.Event, control_plane: ControlPlane):
    """Speak primary local replies and backup replies from chunks.json.

    Both paths use speak_locally_pipelined, whose player checks microphone
    state every BARGE_IN_POLL_S, so either model can be interrupted. Primary
    replies arrive through a local queue and do not wait for Git; the response
    ID prevents the durable remote copy from being spoken a second time.
    """
    # seed with anything already answered from a previous session so we
    # don't replay old responses on startup
    played = {e.get("datetime") for e in load_chunks() if "response" in e}
    next_remote_poll = 0.0

    while not stop_event.is_set():
        # Primary Groq replies take the direct local path for minimum latency.
        for event in load_local_responses():
            key = event["response_id"]
            if key in played:
                continue
            played.add(key)

            reply = event["text"]
            print(f"[talkback:groq] {reply[:60]!r}")
            control_plane.usage.add("qwen/qwen3.6-27b", event.get("usage"))
            _write_local_status(
                conv_state="speaking", conv_state_ts=time.time(), last_reply_text=reply
            )
            speak_locally_pipelined(reply, session_start_ts=time.time(), tmp_prefix="groq_reply")
            _write_local_status(conv_state="idle", conv_state_ts=time.time())
            if event.get("user_text") and not event.get("logged"):
                control_plane.record_voice_turn(event["user_text"], reply)

        # Backup responses still arrive through the durable remote log, but
        # keep its slower Git polling separate from the 100ms local queue.
        if time.monotonic() >= next_remote_poll:
            git_pull_quiet()
            for entry in load_chunks():
                key = entry.get("datetime")
                if key in played or "response" not in entry:
                    continue
                played.add(key)

                reply = entry.get("response", "")
                print(f"[talkback:qwen-backup] {reply[:60]!r}")
                _write_local_status(
                    conv_state="speaking", conv_state_ts=time.time(), last_reply_text=reply
                )

                speak_locally_pipelined(reply, session_start_ts=time.time())

                _write_local_status(conv_state="idle", conv_state_ts=time.time())
                if entry.get("text"):
                    control_plane.record_voice_turn(entry["text"], reply, model="qwen2.5:3b-backup")
            next_remote_poll = time.monotonic() + RESPONSE_POLL_SECONDS

        stop_event.wait(LOCAL_RESPONSE_POLL_SECONDS)


def main():
    # Fresh session: clear any stale mic/conv state left from a previous run
    # before the dashboard starts reading it.
    for session_file in (LOCAL_STATUS_FILE, LOCAL_RESPONSE_FILE):
        try:
            os.remove(session_file)
        except FileNotFoundError:
            pass
    control_plane = ControlPlane()
    start_dashboard_server(control_plane)

    # Reset any stale "ready" flag from a previous session so that if the
    # Qwen fallback does get triggered later, wait_for_ready-style checks
    # elsewhere don't mistake an old signal for a fresh one. Best-effort --
    # a failure here shouldn't block starting the mic.
    try:
        reset_status()
    except Exception as e:
        print(f"[cloud] could not reset status.json (continuing anyway): {e}")

    # Tag every chunk and any lazily-dispatched backup run with the same
    # session ID. This lets a newly booted watcher distinguish the unanswered
    # chunk that launched it from stale, pre-existing conversation history.
    session_id = str(time.time_ns())
    _write_local_status(session_id=session_id)

    # No pre-trigger, no wait-for-ready: Groq answers directly, so the mic
    # starts immediately. live_split_on_pauses.py triggers qwen_watcher.yml
    # itself, only if/when Groq actually fails.
    print("[local] starting mic listener...")
    listener_env = os.environ.copy()
    listener_env["RUNNER_SESSION_ID"] = session_id
    listener = subprocess.Popen([sys.executable, LISTENER_SCRIPT], env=listener_env)

    stop_event = threading.Event()
    responder = threading.Thread(target=response_watcher_loop, args=(stop_event, control_plane), daemon=True)
    responder.start()
    completion_watcher = threading.Thread(target=gptoss_completion_watcher, args=(stop_event, control_plane), daemon=True)
    completion_watcher.start()

    print(f"\nReady. Idle until you speak. Dashboard: http://localhost:{DASHBOARD_PORT}")
    print("Ctrl+C to stop everything.\n")

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

        # Only cancel a watcher that this session actually used. Looking up
        # and cancelling the latest workflow unconditionally can stop an
        # unrelated/manual backup run when Groq never failed here.
        if _read_local_status().get("qwen_fallback_triggered"):
            try:
                run_id = get_latest_run_id()
                if run_id:
                    cancel_run(run_id)
            except Exception as e:
                print(f"[cloud] could not check/cancel Qwen run: {e}")


if __name__ == "__main__":
    main()
