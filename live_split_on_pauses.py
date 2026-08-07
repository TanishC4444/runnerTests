"""
Live version: listens continuously, prints a chunk each time you pause, and
publishes each chunk as a JSON entry in chat/Log 1/chunks.json as it happens.

Pause detection uses webrtcvad (voice activity detection) with a tunable
pause length (PAUSE_MS), not Vosk's own built-in endpointer.

Correction step: ASR sometimes mishears a word as a real-but-wrong word.
language_tool_python catches spelling AND grammar/context issues.

Response step: each chunk is sent to Groq (Qwen 3.6 27B, see groq_responder.py)
right here, in-process -- that's the primary "brain" now. The cloud Qwen
watcher (qwen_watcher.yml / watch_and_respond.py) only gets triggered as a
backup, automatically, the first time Groq comes back rate-limited or
otherwise fails.

Setup (run on your own machine, needs a mic):
    pip install vosk pyaudio webrtcvad language_tool_python requests --break-system-packages
    export GROQ_API_KEY="gsk_..."   # Groq console -- required for the primary path
    export GH_TOKEN="ghp_..."       # repo + workflow scope -- needed for the Qwen fallback trigger

language_tool_python needs Java (JRE 17+):
    brew install openjdk
    sudo ln -sfn $(brew --prefix openjdk)/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk
    java -version   # confirm it works

First run downloads ~200MB LanguageTool package (one-time, then offline).

Download a small offline speech model (one-time, ~40MB):
    curl -L -o model.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip model.zip
    mv vosk-model-small-en-us-0.15 model

Each printed chunk shows:
  t=            total time since the script started listening
  since last    time since the previous chunk was printed

Each chunk is appended through the GitHub API to:
  chat/Log 1/chunks.json
as {"datetime": ..., "talk_seconds": ..., "text": ..., "raw_text": ...}

NOTE ON SYNCING: both this listener and the cloud watcher update the same
JSON file through GitHub's Contents API. Each update re-fetches the latest
blob and retries optimistic-lock conflicts, so neither side performs local
Git commits/rebases or overwrites the other's fields.
"""

import base64
import json
import os
import queue
import sys
import tempfile
import threading
import time
from datetime import datetime

import pyaudio
import requests
import webrtcvad
from vosk import Model, KaldiRecognizer
import language_tool_python

import groq_responder

SESSION_ID = os.environ.get("RUNNER_SESSION_ID")

# Local-only status file, shared with run_all.py so it can (a) show a live
# dashboard and (b) know the instant you start talking again, so it can kill
# playback for barge-in. This never touches Git/GitHub -- it's pure local
# IPC between the two processes on your machine, written to on VAD state
# *transitions* only (not every audio frame) so it stays cheap.
LOCAL_STATUS_FILE = os.path.join(tempfile.gettempdir(), "jarvis_local_status.json")


def _write_local_status(**fields):
    """Atomic read-modify-write against the shared local status file."""
    try:
        try:
            with open(LOCAL_STATUS_FILE, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        data.update(fields)
        tmp_path = LOCAL_STATUS_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, LOCAL_STATUS_FILE)
    except Exception:
        # The dashboard/barge-in is a nice-to-have; never let it take down
        # the actual mic pipeline.
        pass

MODEL_PATH = "model"
SAMPLE_RATE = 16000

FRAME_MS = 30                     # webrtcvad requires 10/20/30ms frames
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit samples
PAUSE_MS = 600                     # how much silence = "you paused"
PAUSE_FRAMES = PAUSE_MS // FRAME_MS
VAD_AGGRESSIVENESS = 3             # 0 (lenient) - 3 (strict about what counts as speech)
BARGE_IN_CONFIRM_FRAMES = 5        # ~150ms of sustained speech before we call it a real barge-in
                                    # (was 1 frame / 30ms -- that's why a cough or click triggered it)
SHOW_PARTIAL_PREVIEW = True        # set False if your terminal/log doesn't handle \r

LOG_DIR = os.path.join("chat", "Log 1")
LOG_FILE = os.path.join(LOG_DIR, "chunks.json")

TOKEN = os.environ.get("GH_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY", "TanishC4444/runnerTests")
BRANCH = os.environ.get("GITHUB_REF_NAME", "main")
FILE_PATH = "chat/Log 1/chunks.json"
API_URL = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
API_TIMEOUT_S = 30
PUSH_RETRIES = 8
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Cache-Control": "no-cache",
}


def fmt(seconds: float) -> str:
    return f"{seconds:6.2f}s"


def correct(text: str, tool: language_tool_python.LanguageTool) -> str:
    if not text.strip():
        return text
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)


def load_log() -> list:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def entry_key(entry: dict) -> tuple:
    return entry.get("datetime"), entry.get("raw_text"), entry.get("talk_seconds")


def fetch_remote_log():
    response = requests.get(
        API_URL,
        headers=HEADERS,
        params={"ref": BRANCH},
        timeout=API_TIMEOUT_S,
    )
    response.raise_for_status()
    data = response.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return json.loads(content) if content.strip() else [], data["sha"]


def append_remote_chunk(entry: dict):
    """Append one chunk without rebasing or overwriting cloud responses."""
    if not TOKEN:
        raise RuntimeError("GH_TOKEN is not set")

    key = entry_key(entry)
    for attempt in range(1, PUSH_RETRIES + 1):
        entries, sha = fetch_remote_log()
        if any(entry_key(existing) == key for existing in entries):
            return

        entries.append(entry)
        content = base64.b64encode(
            json.dumps(entries, indent=2).encode("utf-8")
        ).decode("utf-8")
        response = requests.put(
            API_URL,
            headers=HEADERS,
            json={
                "message": "New chunk",
                "content": content,
                "sha": sha,
                "branch": BRANCH,
            },
            timeout=API_TIMEOUT_S,
        )
        if response.ok:
            return
        if response.status_code not in (409, 422):
            detail = response.text.strip() or "<empty response body>"
            raise RuntimeError(
                f"GitHub chunk update failed with HTTP {response.status_code}: "
                f"{detail[:2000]}"
            )
        print(f"[sync] file changed; retrying append ({attempt}/{PUSH_RETRIES})")

    raise RuntimeError("chunk append kept conflicting after all retries")


def worker_loop(work_q: "queue.Queue", tool, t_start: float):
    """Runs on a background thread: does the slow stuff (grammar correction,
    remote append, and printing) so the mic-reading loop never waits on it."""
    pending = []
    while True:
        item = work_q.get()
        if item is None:
            return

        raw_text, talk_seconds, now, since_last = item
        elapsed = now - t_start

        fixed = correct(raw_text, tool)
        tag = "  (corrected)" if fixed != raw_text else ""

        if SHOW_PARTIAL_PREVIEW:
            sys.stdout.write("\r" + " " * 80 + "\r")
        print(f"[t={fmt(elapsed)} | +{fmt(since_last)} since last]  {fixed}{tag}")

        entry = {
            "datetime": datetime.now().isoformat(timespec="microseconds"),
            "talk_seconds": round(talk_seconds, 2),
            "text": fixed,
            "raw_text": raw_text,
        }
        if SESSION_ID:
            entry["session_id"] = SESSION_ID
        _write_local_status(last_user_text=fixed, last_user_ts=time.time())

        # Try Groq (Qwen 3.6 27B) first -- fast enough to answer right here,
        # no round trip through GitHub/Actions. Attach the reply to the
        # entry BEFORE it's pushed: run_all.py's playback watcher already
        # speaks any chunk that arrives with a "response" field, so this
        # needs no changes on that side.
        fallback_reason = None
        try:
            reply = groq_responder.ask_groq(fixed)
            entry["response"] = reply
            print(f"  [groq] {reply[:80]!r}")
            _write_local_status(
                conv_state="speaking", conv_state_ts=time.time(), last_reply_text=reply
            )
        except groq_responder.GroqRateLimited:
            print("  [groq] rate limited -- falling back to Qwen for this chunk")
            fallback_reason = "rate limited"
        except Exception as e:
            print(f"  [groq] request failed ({e}) -- falling back to Qwen for this chunk")
            fallback_reason = str(e)

        pending.append((entry, fallback_reason))

        # Retain failed entries in memory and flush them in order on the next
        # chunk. append_remote_chunk is idempotent, so ambiguous retries are safe.
        while pending:
            pending_entry, pending_fallback_reason = pending[0]
            try:
                append_remote_chunk(pending_entry)
                print("[sync] chunk appended")
                pending.pop(0)
                # Store the unanswered chunk before starting a new watcher.
                # Otherwise a slow Actions startup can see the chunk in its
                # initial snapshot, classify it as pre-session history, and
                # never answer the request that caused the fallback.
                if pending_fallback_reason is not None:
                    fallback_ready = groq_responder.ensure_qwen_fallback(
                        pending_fallback_reason
                    )
                    if fallback_ready:
                        _write_local_status(qwen_fallback_triggered=True)
                # Cloud watcher has the chunk now and will start generating --
                # this is what makes the dashboard show "thinking".
                _write_local_status(conv_state="thinking", conv_state_ts=time.time())
            except Exception as e:
                print(f"[sync] chunk append failed; will retry: {e}")
                break


def live_split():
    print("Loading grammar/spelling checker (first run downloads it, be patient)...")
    tool = language_tool_python.LanguageTool("en-US")

    print("Loading speech model...")
    vosk_model = Model(MODEL_PATH)
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_BYTES // 2,
    )
    stream.start_stream()

    entries = load_log()
    _write_local_status(mic_state="listening", mic_state_ts=time.time(), conv_state="idle")
    print(f"Publishing to {LOG_FILE} ({len(entries)} entries in local mirror)")
    print(f"Listening... a {PAUSE_MS}ms pause ends a chunk. Ctrl+C to stop.")
    print("Type 'm' + Enter at any time to mute/unmute.\n")

    t_start = time.perf_counter()
    t_last_chunk = t_start

    work_q: "queue.Queue" = queue.Queue()
    worker = threading.Thread(
        target=worker_loop,
        args=(work_q, tool, t_start),
        daemon=True,
    )
    worker.start()

    muted = threading.Event()

    def mute_control_loop():
        while True:
            try:
                line = input()
            except EOFError:
                return
            if line.strip().lower() in ("m", "mute"):
                if muted.is_set():
                    muted.clear()
                    print("[mic] UNMUTED")
                    _write_local_status(mic_state="listening", mic_state_ts=time.time())
                else:
                    muted.set()
                    print("[mic] MUTED — not listening")
                    _write_local_status(mic_state="muted", mic_state_ts=time.time())

    control_thread = threading.Thread(target=mute_control_loop, daemon=True)
    control_thread.start()

    silence_run = 0
    in_speech = False
    speech_run = 0       # frames flagged as actual speech, i.e. talk time
    last_partial = ""
    barge_in_confirm_run = 0   # consecutive speech frames -- separate from
                               # speech_run so debouncing barge-in never
                               # affects chunk-boundary timing/accuracy
    barge_in_signaled = False

    try:
        while True:
            frame = stream.read(FRAME_BYTES // 2, exception_on_overflow=False)

            if muted.is_set():
                # keep pulling from the stream so the buffer doesn't
                # overflow, but don't process anything while muted —
                # also clear any in-progress chunk state so unmuting
                # doesn't resume a stale half-finished chunk
                in_speech = False
                silence_run = 0
                speech_run = 0
                last_partial = ""
                barge_in_confirm_run = 0
                barge_in_signaled = False
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            rec.AcceptWaveform(frame)  # feed regardless, keeps partials live

            if is_speech:
                in_speech = True
                silence_run = 0
                speech_run += 1

                # Always check the partial now (not just when the preview is
                # on) -- barge-in confirmation needs it.
                partial = json.loads(rec.PartialResult()).get("partial", "")

                # Debounced separately from chunk-boundary tracking above --
                # a single noisy frame (cough, click, chair creak) no longer
                # fires barge-in. Needs BARGE_IN_CONFIRM_FRAMES in a row
                # AND Vosk must have actually recognized a real word by
                # then -- webrtcvad alone can't tell "cough" from "speech",
                # but Vosk producing a partial transcript means it heard
                # something word-shaped, not just noise-shaped.
                barge_in_confirm_run += 1
                has_recognized_word = bool(partial.strip())
                if (
                    not barge_in_signaled
                    and barge_in_confirm_run >= BARGE_IN_CONFIRM_FRAMES
                    and has_recognized_word
                ):
                    barge_in_signaled = True
                    _write_local_status(mic_state="speech", mic_state_ts=time.time())

                if SHOW_PARTIAL_PREVIEW:
                    if partial and partial != last_partial:
                        last_partial = partial
                        sys.stdout.write(f"\r...{partial}" + " " * 10)
                        sys.stdout.flush()

            elif in_speech:
                silence_run += 1
                barge_in_confirm_run = 0
                barge_in_signaled = False

            should_finalize = in_speech and (not is_speech) and silence_run >= PAUSE_FRAMES

            if should_finalize:
                result = json.loads(rec.FinalResult())
                raw_text = result.get("text", "").strip()
                talk_seconds = speech_run * FRAME_MS / 1000

                # reset recognizer for the next chunk right away so the mic
                # loop is never blocked waiting on correction/logging
                rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)

                in_speech = False
                silence_run = 0
                speech_run = 0
                last_partial = ""
                barge_in_confirm_run = 0
                barge_in_signaled = False
                _write_local_status(mic_state="listening", mic_state_ts=time.time())

                if raw_text:
                    now = time.perf_counter()
                    since_last = now - t_last_chunk
                    t_last_chunk = now
                    work_q.put((raw_text, talk_seconds, now, since_last))

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        work_q.put(None)
        worker.join(timeout=5)
        tool.close()


if __name__ == "__main__":
    live_split()
