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

import asyncio
import base64
import http.server
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # Windows fallback; fcntl is available on macOS/Linux.
    fcntl = None

import requests
import edge_tts

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
GIT_LOCK_FILE = os.path.join(tempfile.gettempdir(), "runnerTests_git_sync.lock")

# Same path live_split_on_pauses.py writes to. Local-only IPC, never touches
# Git -- this is what makes barge-in and the dashboard possible without
# changing anything about how the cloud side works.
LOCAL_STATUS_FILE = os.path.join(tempfile.gettempdir(), "jarvis_local_status.json")
DASHBOARD_PORT = 8765
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


def get_job_id(run_id) -> int | None:
    """The job-level logs endpoint (unlike the run-level one) returns raw
    text for an in_progress job, which is what makes fast log-polling
    possible at all -- but it needs a job_id, not just a run_id."""
    url = f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/jobs"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    for job in resp.json().get("jobs", []):
        if job.get("status") in ("in_progress", "queued"):
            return job["id"]
    return None


_RESULT_RE = re.compile(r"^RESULT::(\{.*\})\s*$", re.MULTILINE)


def fetch_new_log_text(job_id: int, already_seen_len: int) -> tuple[str, int]:
    """Returns (new_text_suffix, new_total_len). Job logs are append-only
    while the job runs, so tracking the previously-seen length and slicing
    is enough to get "what's new" without re-parsing the whole thing."""
    url = f"https://api.github.com/repos/{REPO}/actions/jobs/{job_id}/logs"
    resp = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=15)
    if resp.status_code == 404:
        # job hasn't produced a log blob yet, or already rotated past this id
        return "", already_seen_len
    resp.raise_for_status()
    full_text = resp.text
    if len(full_text) <= already_seen_len:
        return "", already_seen_len
    return full_text[already_seen_len:], len(full_text)


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
    def log_message(self, *args):
        pass  # keep the console clean; the dashboard doesn't need per-request logs

    def do_GET(self):
        if self.path.startswith("/status"):
            body = json.dumps(_read_local_status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def start_dashboard_server() -> http.server.ThreadingHTTPServer:
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


_played_lock = threading.Lock()
_played: set[str] = set()  # datetime strings already spoken, shared across both watcher threads


def _speak_reply_once(reply: str, entry_datetime: str, source: str, t_reply_ready: float | None = None):
    """Guards against speaking the same reply twice -- the fast log-based
    path and the slower git-fallback path both funnel through here."""
    with _played_lock:
        if entry_datetime in _played:
            return
        _played.add(entry_datetime)

    latency_note = ""
    if t_reply_ready is not None:
        latency_note = f" (seen {time.time()-t_reply_ready:.2f}s after generated, clock skew included)"
    print(f"[talkback:{source}] {reply[:60]!r}{latency_note}")
    _write_local_status(conv_state="speaking", conv_state_ts=time.time(), last_reply_text=reply)

    t0 = time.time()
    tmp_path = speak_locally(reply, "qwen_reply.mp3")
    print(f"  [timing] local TTS synth={time.time()-t0:.2f}s")
    if tmp_path:
        play_audio_interruptible(tmp_path, session_start_ts=time.time())

    _write_local_status(conv_state="idle", conv_state_ts=time.time())


def log_watcher_loop(job_id: int, stop_event: threading.Event):
    """FAST path: polls the runner's own live job log for RESULT:: lines
    instead of waiting on a git commit round trip. This is what actually
    cuts latency -- the git-based response_watcher_loop below still runs,
    but now purely as a slower durable-record fallback (e.g. if a log line
    is missed due to GitHub's internal log-buffering lag or this poll
    catching a job right as it transitions/rotates)."""
    seen_len = 0
    LOG_POLL_SECONDS = 1.5
    while not stop_event.is_set():
        try:
            new_text, seen_len = fetch_new_log_text(job_id, seen_len)
        except Exception as e:
            print(f"[log-watch] poll failed, will retry: {e}")
            stop_event.wait(LOG_POLL_SECONDS)
            continue

        for match in _RESULT_RE.finditer(new_text):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            _speak_reply_once(
                payload.get("text", ""),
                payload.get("datetime", ""),
                source="log",
            )

        stop_event.wait(LOG_POLL_SECONDS)


def response_watcher_loop(stop_event: threading.Event):
    """SLOW fallback path: still reads chunks.json via git pull, same as
    before -- but now only needed as a durable-record catch for whatever
    the fast log_watcher_loop misses (buffering lag, a job restart, etc).
    _speak_reply_once's dedup means it's a no-op for anything already
    spoken via the log path, so this just quietly confirms the record."""
    played_at_startup = {e.get("datetime") for e in load_chunks() if "response" in e}
    with _played_lock:
        _played.update(played_at_startup)

    GIT_FALLBACK_POLL_SECONDS = 8  # no longer time-critical; log path handles latency

    while not stop_event.is_set():
        git_pull_quiet()
        for entry in load_chunks():
            key = entry.get("datetime")
            if "response" not in entry:
                continue
            _speak_reply_once(entry.get("response", ""), key, source="git-fallback")

        stop_event.wait(GIT_FALLBACK_POLL_SECONDS)


def main():
    # Fresh session: clear any stale mic/conv state left from a previous run
    # before the dashboard starts reading it.
    try:
        os.remove(LOCAL_STATUS_FILE)
    except FileNotFoundError:
        pass
    start_dashboard_server()

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

    job_id = get_job_id(run_id)
    stop_event = threading.Event()
    if job_id:
        print(f"[cloud] job id {job_id} -- fast log-based responses enabled.")
        log_thread = threading.Thread(target=log_watcher_loop, args=(job_id, stop_event), daemon=True)
        log_thread.start()
    else:
        print("[cloud] could not resolve job id; falling back to git-only response timing.")

    responder = threading.Thread(target=response_watcher_loop, args=(stop_event,), daemon=True)
    responder.start()

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
        cancel_run(run_id)


if __name__ == "__main__":
    main()