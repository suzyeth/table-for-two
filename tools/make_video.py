"""Assemble the captioned submission video from the rendered assets, optionally with voice-over.

Sequence (1280x720, 10 fps):
  title card -> instruction + plan card -> learned-policy demo -> 10-seed grid
  -> results card -> latency card -> closing card
All numbers come from docs/slides/deck_data.json and the demo's JSON log, so
re-running after new results keeps the video in step with the slides.

With --voice-dir (output of tools/voiceover.py) every card is held at least as
long as its narration, each segment's WAV is placed at that segment's start,
and the audio track is muxed in with the ffmpeg bundled in imageio-ffmpeg.

Run:  .venv\\Scripts\\python.exe -m tools.make_video --voice-dir out/voice
"""
import argparse
import json
import subprocess
import textwrap
import wave
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DECK_DATA = ROOT / "docs" / "slides" / "deck_data.json"
SIZE = (1280, 720)
FPS = 10
NARRATION_TAIL_S = 0.8  # breathing room after each narrated card
DARK = (27, 34, 32)
LIGHT = (242, 244, 243)
INK = (27, 34, 32)
MUTED = (91, 104, 100)
ACCENT = (229, 178, 15)
BLUE = (44, 90, 120)
GOOD = (47, 122, 79)
CARD_SECONDS = {"title": 4, "plan": 7, "results": 7, "latency": 6, "closing": 5}


def font(size, bold=False):
    names = ("arialbd.ttf", "segoeuib.ttf") if bold else ("arial.ttf", "segoeui.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def letterbox(frame, background=DARK):
    """Fit an HxWx3 frame into SIZE without distortion."""
    image = Image.fromarray(np.asarray(frame))
    scale = min(SIZE[0] / image.width, SIZE[1] / image.height)
    image = image.resize((int(image.width * scale), int(image.height * scale)))
    canvas = Image.new("RGB", SIZE, background)
    canvas.paste(image, ((SIZE[0] - image.width) // 2, (SIZE[1] - image.height) // 2))
    return canvas


def hold(image, seconds):
    frame = np.asarray(image)
    return [frame] * int(round(seconds * FPS))


def caption(image, text):
    """Bottom caption bar over an existing frame."""
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, SIZE[1] - 56, SIZE[0], SIZE[1]], fill=DARK)
    draw.text((32, SIZE[1] - 42), text, font=font(24), fill=(230, 233, 231))
    return image


def title_card(data):
    image = letterbox(Image.open(OUT / "cover.png").convert("RGB"))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 560, SIZE[1]], fill=DARK)
    draw.text((48, 250), data["title"], font=font(64, bold=True), fill=(255, 255, 255))
    for i, line in enumerate(textwrap.wrap(data["subtitle"], 36)):
        draw.text((48, 345 + 34 * i), line, font=font(26), fill=(228, 233, 231))
    draw.text((48, 640), "AI Infra Summit Hackathon · Intel Physical AI Online Challenge", font=font(18), fill=ACCENT)
    return image


def plan_card(demo_log):
    image = Image.new("RGB", SIZE, DARK)
    draw = ImageDraw.Draw(image)
    draw.text((48, 40), "1 · Instruction", font=font(26, bold=True), fill=ACCENT)
    for i, line in enumerate(textwrap.wrap(f'"{demo_log["instruction"]}"', 80)):
        draw.text((48, 84 + 32 * i), line, font=font(24), fill=(230, 233, 231))
    planner = demo_log["planner"]
    draw.text((48, 250), f"2 · Plan  (Qwen2.5-1.5B INT4 on OpenVINO {planner['device']} · {planner['latency_s']:.1f} s · validated)",
              font=font(26, bold=True), fill=ACCENT)
    y = 296
    for stage in demo_log["plan"]:
        parts = []
        for sub in stage:
            if sub["skill"] == "handoff":
                parts.append(f"handoff {sub['object']} {sub['giver']} -> {sub['receiver']}")
            else:
                parts.append(" ".join(str(sub.get(k, "")) for k in ("skill", "arm", "object", "into") if sub.get(k)))
        draw.text((72, y), "  ;  ".join(parts), font=font(22), fill=(200, 208, 205))
        y += 38
    return image


def results_card(data):
    image = Image.new("RGB", SIZE, LIGHT)
    draw = ImageDraw.Draw(image)
    draw.text((48, 36), "Results on 10 held-out randomised seeds", font=font(36, bold=True), fill=INK)
    columns = [("Sub-goal", 48), ("Scripted", 560), ("Learned policy", 760), ("Hybrid", 1020)]
    for name, x in columns:
        draw.text((x, 120), name, font=font(24, bold=True), fill=MUTED)
    rows = list(zip(data["subgoals"], data["scripted_10_seeds"], data["policy_10_seeds"] or [None] * 6,
                    data["hybrid_10_seeds"] or [None] * 6))
    rows.append(("Full task", 10, data.get("policy_full_task"), data.get("hybrid_full_task")))
    for index, (name, scripted, policy, hybrid) in enumerate(rows):
        y = 170 + index * 58
        last = index == len(rows) - 1
        if last:
            draw.rectangle([36, y - 10, 1240, y + 44], fill=(251, 241, 207))
        style = font(26, bold=last)
        draw.text((48, y), name, font=style, fill=INK)
        for value, x, colour in ((scripted, 560, GOOD), (policy, 760, INK), (hybrid, 1020, INK)):
            draw.text((x, y), "pending" if value is None else f"{value}/10", font=style, fill=colour)
    if data.get("failure_note"):
        for i, line in enumerate(textwrap.wrap(data["failure_note"], 95)):
            draw.text((48, 640 + 28 * i), line, font=font(20), fill=BLUE)
    return image


def latency_card(data):
    image = Image.new("RGB", SIZE, LIGHT)
    draw = ImageDraw.Draw(image)
    draw.text((48, 36), "OpenVINO on Intel: ACT policy latency per action chunk", font=font(34, bold=True), fill=INK)
    rows = data["latency"]
    top = max(row["ms"] for row in rows)
    for index, row in enumerate(rows):
        y = 150 + index * 100
        width = int(700 * row["ms"] / top)
        colour = ACCENT if "INT8" in row["label"] else BLUE
        draw.text((48, y + 14), row["label"], font=font(26), fill=INK)
        draw.rectangle([340, y, 340 + width, y + 60], fill=colour)
        draw.text((350 + width, y + 14), f"{row['ms']:.1f} ms", font=font(26, bold=True), fill=INK)
    draw.text((48, 580), f"INT8 (NNCF) is {data['int8_speedup']} faster than FP32 on the same CPU.", font=font(26), fill=INK)
    draw.text((48, 630), data["hardware_note"], font=font(18), fill=MUTED)
    return image


def closing_card(data):
    image = Image.new("RGB", SIZE, DARK)
    draw = ImageDraw.Draw(image)
    draw.text((48, 60), "How it works", font=font(40, bold=True), fill=(255, 255, 255))
    steps = ["Speech or text", "OpenVINO LLM planner", "Plan validation", "ACT policy (OpenVINO INT8)", "MuJoCo · 2 × SO-101"]
    for index, step in enumerate(steps):
        x = 48 + index * 240
        draw.rounded_rectangle([x, 200, x + 220, 320], radius=12, fill=(40, 48, 45),
                               outline=ACCENT if "ACT" in step else (70, 80, 76), width=3)
        for i, line in enumerate(textwrap.wrap(step, 16)):
            draw.text((x + 16, 225 + 32 * i), line, font=font(24, bold=True), fill=(240, 240, 240))
    draw.text((48, 420), "Grasps are finger contact only; hybrid stages are reported separately from policy wins.",
              font=font(24), fill=(200, 208, 205))
    draw.text((48, 600), f"{data['title']} · everything shown is reproducible from the repository",
              font=font(24), fill=ACCENT)
    return image


def card_seconds(name, durations):
    spoken = durations.get(name, 0.0)
    return max(CARD_SECONDS[name], spoken + NARRATION_TAIL_S if spoken else 0.0)


def read_wav(path):
    with wave.open(str(path), "rb") as wav_file:
        rate = wav_file.getframerate()
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
    return rate, samples


def build_audio(voice_dir, starts, total_s, path):
    """Mix each segment's narration in at its start time; returns the sample rate."""
    rate = None
    track = None
    for name, start in starts.items():
        wav_path = voice_dir / f"{name}.wav"
        if not wav_path.exists():
            continue
        seg_rate, samples = read_wav(wav_path)
        if track is None:
            rate = seg_rate
            track = np.zeros(int(total_s * rate) + rate, dtype=np.int32)
        offset = int(start * rate)
        end = min(len(track), offset + len(samples))
        track[offset:end] += samples[: end - offset]
    clipped = np.clip(track, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(clipped.tobytes())


def mux(video_path, audio_path, out_path):
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video_path),
               "-i", str(audio_path), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out_path)]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Build the captioned submission video.")
    parser.add_argument("--demo", type=Path, default=OUT / "demo_policy_seed8.mp4")
    parser.add_argument("--grid", type=Path, default=OUT / "grid_10seeds.mp4")
    parser.add_argument("--voice-dir", type=Path, help="WAVs from tools/voiceover.py; adds narration")
    parser.add_argument("--out", type=Path, default=OUT / "submission_video.mp4")
    args = parser.parse_args()

    data = json.loads(DECK_DATA.read_text(encoding="utf-8"))
    demo_log = json.loads(args.demo.with_suffix(".json").read_text(encoding="utf-8"))
    durations = {}
    if args.voice_dir:
        durations = json.loads((args.voice_dir / "durations.json").read_text(encoding="utf-8"))

    frames, starts = [], {}

    def begin(name):
        starts[name] = len(frames) / FPS

    begin("title")
    frames += hold(title_card(data), card_seconds("title", durations))
    begin("plan")
    frames += hold(plan_card(demo_log), card_seconds("plan", durations))
    executor = "learned policy" if demo_log["executor"] == "policy" else "scripted skills"
    begin("demo")
    for frame in imageio.get_reader(args.demo):
        frames.append(np.asarray(caption(letterbox(frame), f"3 · Execution by the {executor} — seed {demo_log['seed']}")))
    begin("grid")
    for frame in imageio.get_reader(args.grid):
        frames.append(np.asarray(caption(letterbox(frame), "4 · Robustness: 10 randomised seeds (positions, mass, friction, light)")))
    begin("results")
    frames += hold(results_card(data), card_seconds("results", durations))
    begin("latency")
    frames += hold(latency_card(data), card_seconds("latency", durations))
    begin("closing")
    frames += hold(closing_card(data), card_seconds("closing", durations))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_s = len(frames) / FPS
    silent = args.out if not args.voice_dir else args.out.with_name(args.out.stem + "_silent.mp4")
    imageio.mimsave(silent, frames, fps=FPS, quality=8)
    if args.voice_dir:
        for name in ("demo", "grid"):
            segment_end = min((t for t in starts.values() if t > starts[name]), default=total_s)
            if durations.get(name, 0) > segment_end - starts[name]:
                print(f"warning: {name} narration ({durations[name]}s) is longer than its footage")
        audio_path = args.out.with_suffix(".wav")
        build_audio(args.voice_dir, starts, total_s, audio_path)
        mux(silent, audio_path, args.out)
    print(f"wrote {args.out}: {len(frames)} frames, {total_s:.0f} s; segment starts {starts}")


if __name__ == "__main__":
    main()
