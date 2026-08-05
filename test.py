"""
Live version: listens continuously, prints a chunk each time you pause, and
writes each chunk as a JSON entry to chat/Log 1/chunks.json as it happens.

Pause detection uses webrtcvad (voice activity detection) with a tunable
pause length (PAUSE_MS), not Vosk's own built-in endpointer.

Correction step: ASR sometimes mishears a word as a real-but-wrong word.
language_tool_python catches spelling AND grammar/context issues.

Setup (run on your own machine, needs a mic):
    pip install vosk pyaudio webrtcvad language_tool_python --break-system-packages

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

Each chunk is also appended live to:
  chat/Log 1/chunks.json
as {"datetime": ..., "talk_seconds": ..., "text": ..., "raw_text": ...}
"""

import json
import os
import queue
import sys
import threading
import time
from datetime import datetime

import pyaudio
import webrtcvad
from vosk import Model, KaldiRecognizer
import language_tool_python

MODEL_PATH = "model"
SAMPLE_RATE = 16000

FRAME_MS = 30                     # webrtcvad requires 10/20/30ms frames
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit samples
PAUSE_MS = 600                     # how much silence = "you paused"
PAUSE_FRAMES = PAUSE_MS // FRAME_MS
VAD_AGGRESSIVENESS = 2             # 0 (lenient) - 3 (strict about what counts as speech)
SHOW_PARTIAL_PREVIEW = True        # set False if your terminal/log doesn't handle \r

LOG_DIR = os.path.join("chat", "Log 1")
LOG_FILE = os.path.join(LOG_DIR, "chunks.json")


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


def save_log(entries: list):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def worker_loop(work_q: "queue.Queue", entries: list, tool, t_start: float, state: dict):
    """Runs on a background thread: does the slow stuff (grammar correction,
    disk write, printing) so the mic-reading loop never has to wait on it."""
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
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "talk_seconds": round(talk_seconds, 2),
            "text": fixed,
            "raw_text": raw_text,
        }
        entries.append(entry)
        save_log(entries)


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
    print(f"Logging to {LOG_FILE} ({len(entries)} existing entries)")
    print(f"Listening... a {PAUSE_MS}ms pause ends a chunk. Ctrl+C to stop.\n")

    t_start = time.perf_counter()
    t_last_chunk = t_start

    work_q: "queue.Queue" = queue.Queue()
    worker = threading.Thread(
        target=worker_loop,
        args=(work_q, entries, tool, t_start, {}),
        daemon=True,
    )
    worker.start()

    silence_run = 0
    in_speech = False
    speech_run = 0       # frames flagged as actual speech, i.e. talk time
    last_partial = ""

    try:
        while True:
            frame = stream.read(FRAME_BYTES // 2, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            rec.AcceptWaveform(frame)  # feed regardless, keeps partials live

            if is_speech:
                in_speech = True
                silence_run = 0
                speech_run += 1

                if SHOW_PARTIAL_PREVIEW:
                    partial = json.loads(rec.PartialResult()).get("partial", "")
                    if partial and partial != last_partial:
                        last_partial = partial
                        sys.stdout.write(f"\r...{partial}" + " " * 10)
                        sys.stdout.flush()

            elif in_speech:
                silence_run += 1

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