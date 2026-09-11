"""Generate the English voice-over offline with Piper, one WAV per video segment.

Text comes from docs/narration.json; {placeholders} are filled from
docs/slides/deck_data.json and written the way a narrator would say them
("four out of ten", "2.7 times"). Nothing is sent to any online service.

Run:  .venv\\Scripts\\python.exe -m tools.voiceover
Then: .venv\\Scripts\\python.exe -m tools.make_video --voice-dir out/voice
"""
import argparse
import json
import wave
from pathlib import Path

from piper import PiperVoice

ROOT = Path(__file__).resolve().parent.parent
NARRATION = ROOT / "docs" / "narration.json"
DECK_DATA = ROOT / "docs" / "slides" / "deck_data.json"
VOICE_MODEL = ROOT / "models" / "tts" / "en_US-lessac-medium.onnx"
OUT_DIR = ROOT / "out" / "voice"
SEGMENTS = ("title", "plan", "demo", "grid", "results", "latency", "closing")
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def out_of_ten(value):
    return "a number still being measured" if value is None else f"{NUMBER_WORDS[value]} out of ten"


def fill(text, deck):
    int8_ms = next(row["ms"] for row in deck["latency"] if row["label"].startswith("INT8"))
    values = {
        "policy_full_task": out_of_ten(deck.get("policy_full_task")),
        "hybrid_full_task": out_of_ten(deck.get("hybrid_full_task")),
        "int8_cpu_ms": f"{int8_ms:.1f}",
        "int8_speedup_words": deck["int8_speedup"].replace("×", " times"),
    }
    return text.format(**values)


def synthesize(voice, text, path):
    with wave.open(str(path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def main():
    parser = argparse.ArgumentParser(description="Synthesize the voice-over with Piper.")
    parser.add_argument("--model", type=Path, default=VOICE_MODEL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"{args.model} not found; run: python -m piper.download_voices en_US-lessac-medium "
                         f"--data-dir {args.model.parent}")
    narration = json.loads(NARRATION.read_text(encoding="utf-8"))
    deck = json.loads(DECK_DATA.read_text(encoding="utf-8"))
    voice = PiperVoice.load(str(args.model))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    durations, script = {}, {}
    for segment in SEGMENTS:
        text = fill(narration[segment], deck)
        durations[segment] = round(synthesize(voice, text, args.out_dir / f"{segment}.wav"), 2)
        script[segment] = text
        print(f"{segment:8s} {durations[segment]:5.1f}s  {text}")
    (args.out_dir / "durations.json").write_text(json.dumps(durations, indent=1), encoding="utf-8")
    (args.out_dir / "script.json").write_text(json.dumps(script, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
