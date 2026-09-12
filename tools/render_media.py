"""Render the still images and the 10-seed grid video used by the README, deck and submission video.

  cover      out/cover.png + docs/media/cover.png   both arms carrying the plate (seed 0, operator view)
  cover_alt  out/cover_alt.png                       the pour, seen from the front-high camera
  grid       out/grid_10seeds.mp4 + docs/media/grid_10_seeds.png
             seeds 0-9 side by side (2 x 5), each tile captioned with its seed and final result

Everything is the scripted contact pipeline; nothing here needs a trained policy.

Run:  python -m tools.render_media [cover|cover_alt|grid|all]
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim.env import DinnerTableEnv
from sim.task import DEFAULT_PLAN, Executor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
MEDIA = ROOT / "docs" / "media"
TILE = (192, 256)  # (height, width) of one grid tile
GRID = (2, 5)
FRAME_EVERY = 4  # control steps per grid frame (20 Hz -> 5 fps)


def font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def snapshot_at(env, executor, stage_index, fraction, camera, size):
    """Run the plan up to ``stage_index`` and ``fraction`` of the way through it, then render."""
    executor.run(DEFAULT_PLAN[:stage_index])
    frames = []
    executor.run([DEFAULT_PLAN[stage_index]], on_step=lambda *_: frames.append(None))
    total = len(frames)
    env.reset(0)
    executor.reset()
    executor.run(DEFAULT_PLAN[:stage_index])
    target = int(total * fraction)
    count = [0]

    def stop_at(*_):
        count[0] += 1
        if count[0] == target:
            raise StopIteration

    try:
        executor.run([DEFAULT_PLAN[stage_index]], on_step=stop_at)
    except StopIteration:
        pass
    return env.render(camera, size)


def render_cover(env, executor):
    env.reset(0)
    executor.reset()
    image = Image.fromarray(snapshot_at(env, executor, 0, 0.55, "operator", (720, 1280)))
    for path in (OUT / "cover.png", MEDIA / "cover.png"):
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    print("wrote cover.png (two-handed plate carry)")


def render_cover_alt(env, executor):
    env.reset(0)
    executor.reset()
    image = Image.fromarray(snapshot_at(env, executor, 3, 0.6, "front_high", (720, 1280)))
    image.save(OUT / "cover_alt.png")
    print("wrote cover_alt.png (the pour)")


def render_grid(env, executor, seeds):
    clips, results = [], []
    for seed in seeds:
        env.reset(seed)
        executor.reset()
        frames = []

        def grab(*_):
            if env.time_step % FRAME_EVERY == 0:
                frames.append(env.render("operator", TILE))

        executor.run(DEFAULT_PLAN, on_step=grab)
        env.hold(1.0)
        results.append(env.success()["all"])
        clips.append(frames)
        print(f"seed {seed}: {len(frames)} frames, success={results[-1]}")
    length = max(len(c) for c in clips)
    rows, cols = GRID
    writer = imageio.get_writer(OUT / "grid_10seeds.mp4", fps=20 // FRAME_EVERY)
    label = font(16)
    last = None
    for t in range(length + 10):
        sheet = Image.new("RGB", (cols * TILE[1], rows * TILE[0]), (18, 22, 20))
        for i, (seed, clip) in enumerate(zip(seeds, clips)):
            frame = clip[min(t, len(clip) - 1)]
            tile = Image.fromarray(frame)
            draw = ImageDraw.Draw(tile)
            done = t >= len(clip) - 1
            text = f"seed {seed}" + ("  ✓ all 7" if done and results[i] else ("  ✗" if done else ""))
            draw.rectangle([0, 0, TILE[1], 22], fill=(18, 22, 20))
            draw.text((6, 3), text, font=label, fill=(120, 220, 140) if done and results[i] else (230, 233, 231))
            sheet.paste(tile, ((i % cols) * TILE[1], (i // cols) * TILE[0]))
        last = sheet
        writer.append_data(np.asarray(sheet))
    writer.close()
    last.save(MEDIA / "grid_10_seeds.png")
    print(f"wrote grid_10seeds.mp4 ({length} frames) and grid_10_seeds.png; {sum(results)}/{len(seeds)} succeeded")


def main():
    parser = argparse.ArgumentParser(description="Render README/deck/video stills and the 10-seed grid.")
    parser.add_argument("what", nargs="?", default="all", choices=("cover", "cover_alt", "grid", "all"))
    args = parser.parse_args()
    env = DinnerTableEnv(obs_cameras=())
    executor = Executor(env)
    OUT.mkdir(exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    if args.what in ("cover", "all"):
        render_cover(env, executor)
    if args.what in ("cover_alt", "all"):
        render_cover_alt(env, executor)
    if args.what in ("grid", "all"):
        render_grid(env, executor, list(range(10)))
    env.close()


if __name__ == "__main__":
    main()
