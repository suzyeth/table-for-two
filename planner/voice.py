"""Speech to text with the Speechmatics batch API.

Used for spoken instructions ("hold the cup with your left arm and pour").
The API key is read from the environment variable SPEECHMATICS_API_KEY, or
from a local ``.env`` file (see ``.env.example``). The key is never printed.

Run:  .venv\\Scripts\\python.exe -m planner.voice path\\to\\instruction.wav
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://asr.api.speechmatics.com/v2"
POLL_INTERVAL_S = 1.0
REQUEST_TIMEOUT_S = 60
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


class TranscriptionError(RuntimeError):
    pass


def _api_key():
    key = os.environ.get("SPEECHMATICS_API_KEY")
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "SPEECHMATICS_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise TranscriptionError("SPEECHMATICS_API_KEY is not set. Add it to .env (see .env.example) "
                                 "or set the environment variable.")
    return key


def transcribe(audio_path, language="en", operating_point="enhanced", timeout_s=120):
    """Upload ``audio_path``, wait for the job and return the plain-text transcript."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise TranscriptionError(f"audio file not found: {audio_path}")
    if audio_path.suffix.lower() not in AUDIO_SUFFIXES:
        raise TranscriptionError(f"unsupported audio type {audio_path.suffix}; use one of {sorted(AUDIO_SUFFIXES)}")

    headers = {"Authorization": f"Bearer {_api_key()}"}
    config = {"type": "transcription",
              "transcription_config": {"language": language, "operating_point": operating_point}}
    with audio_path.open("rb") as audio:
        response = requests.post(f"{API_URL}/jobs", headers=headers, files={"data_file": audio},
                                 data={"config": json.dumps(config)}, timeout=REQUEST_TIMEOUT_S)
    if response.status_code == 401:
        raise TranscriptionError("Speechmatics rejected the API key (401); check SPEECHMATICS_API_KEY.")
    if not response.ok:
        raise TranscriptionError(f"job submission failed ({response.status_code}): {response.text[:200]}")
    job_id = response.json()["id"]

    deadline = time.monotonic() + timeout_s
    while True:
        status = requests.get(f"{API_URL}/jobs/{job_id}", headers=headers, timeout=REQUEST_TIMEOUT_S)
        status.raise_for_status()
        state = status.json()["job"]["status"]
        if state == "done":
            break
        if state in ("rejected", "deleted", "expired"):
            raise TranscriptionError(f"transcription job {job_id} ended with status '{state}'")
        if time.monotonic() > deadline:
            raise TranscriptionError(f"transcription job {job_id} not done after {timeout_s}s")
        time.sleep(POLL_INTERVAL_S)

    transcript = requests.get(f"{API_URL}/jobs/{job_id}/transcript", headers=headers,
                              params={"format": "txt"}, timeout=REQUEST_TIMEOUT_S)
    transcript.raise_for_status()
    return transcript.text.strip()


def main():
    parser = argparse.ArgumentParser(description="Transcribe a spoken instruction with Speechmatics.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    started = time.perf_counter()
    text = transcribe(args.audio, language=args.language)
    print(text)
    print(f"transcribed in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
