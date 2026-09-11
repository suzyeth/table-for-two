"""Full-pipeline demo: instruction (typed or spoken) -> OpenVINO planner -> bimanual execution -> annotated video.

Examples:
  .venv\\Scripts\\python.exe demo.py --seed 3
  .venv\\Scripts\\python.exe demo.py --text "Carry the plate with both hands, pour a drink and pass the spoon to the right arm"
  .venv\\Scripts\\python.exe demo.py --audio recordings\\set_table.wav --executor policy

Writes an MP4 (operator view with an overhead inset, the instruction, the
current subtask and the final sub-goal checklist) plus a JSON log next to it.
"""
import argparse
import json
import textwrap
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from planner.planner import EXAMPLE_INSTRUCTION, Planner
from sim.env import DinnerTableEnv
from sim.task import Executor

ROOT = Path(__file__).resolve().parent
VIDEO_FPS = 10
FRAME_EVERY = 2  # control steps per video frame (20 Hz -> 10 fps)
MAIN_SIZE = (480, 640)
INSET_SIZE = (150, 200)
BANNER_H = 64
FOOTER_H = 34
END_HOLD_S = 2.5


def _font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = _font(16)
FONT_SMALL = _font(13)


def compose_frame(env, instruction, stage_label, footer, checklist=None, main_frame=None):
    """Banner + camera view + footer. ``main_frame`` reuses an already rendered operator frame."""
    if main_frame is None:
        main = Image.fromarray(env.render("operator", MAIN_SIZE))
        inset = Image.fromarray(env.render("overhead", INSET_SIZE))
        main.paste(inset, (MAIN_SIZE[1] - INSET_SIZE[1] - 8, 8))
    else:
        main = Image.fromarray(main_frame).resize((MAIN_SIZE[1], MAIN_SIZE[0]))
    width, height = MAIN_SIZE[1], MAIN_SIZE[0] + BANNER_H + FOOTER_H
    canvas = Image.new("RGB", (width, height), (18, 22, 20))
    canvas.paste(main, (0, BANNER_H))
    draw = ImageDraw.Draw(canvas)
    lines = textwrap.wrap(f'"{instruction}"', width=78)[:2]
    for i, line in enumerate(lines):
        draw.text((10, 6 + 18 * i), line, font=FONT_SMALL, fill=(230, 233, 231))
    draw.text((10, BANNER_H - 22), f"Subtask: {stage_label}", font=FONT, fill=(240, 194, 51))
    draw.text((10, height - FOOTER_H + 9), footer, font=FONT_SMALL, fill=(151, 163, 159))
    if checklist:
        y = BANNER_H + 10
        for name, ok in checklist.items():
            draw.text((10, y), f"{'OK ' if ok else 'X  '} {name}", font=FONT, fill=(108, 192, 143) if ok else (240, 138, 104))
            y += 20
    return np.asarray(canvas)


def stage_text(stage):
    parts = []
    for sub in stage:
        if sub["skill"] == "handoff":
            parts.append(f"hand {sub['object']} {sub['giver']} -> {sub['receiver']}")
        else:
            target = sub.get("object") or sub.get("into") or ""
            parts.append(f"{sub['arm']}: {sub['skill'].replace('_', ' ')} {target}".strip())
    return "  |  ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Instruction -> plan -> bimanual execution demo video.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="typed instruction")
    source.add_argument("--audio", type=Path, help="spoken instruction (transcribed with Speechmatics)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planner-device", default="CPU")
    parser.add_argument("--executor", choices=("scripted", "policy"), default="scripted")
    parser.add_argument("--policy", type=Path, default=ROOT / "models" / "policy" / "act_int8.xml")
    parser.add_argument("--policy-device", default="CPU")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "demo.mp4")
    args = parser.parse_args()

    transcription_s = None
    if args.audio:
        from planner.voice import transcribe

        started = time.perf_counter()
        instruction = transcribe(args.audio)
        transcription_s = time.perf_counter() - started
    else:
        instruction = args.text or EXAMPLE_INSTRUCTION

    planner = Planner(device=args.planner_device)
    plan, info = planner.plan(instruction)
    footer = (f"Planner: Qwen2.5-1.5B INT4 on OpenVINO {args.planner_device} · {info['latency_s']:.2f}s · "
              f"{info['source']}   Executor: {args.executor}   Seed {args.seed}")
    print(f"instruction: {instruction}\nplan ({info['source']}, {info['latency_s']:.2f}s): {json.dumps(plan)}")

    env = DinnerTableEnv(obs_cameras=())
    env.reset(args.seed)
    frames = []
    stage_log = []

    if args.executor == "scripted":
        executor = Executor(env)
        executor.reset()
        for stage in plan:
            label = stage_text(stage)

            def record(_action, _stage, label=label):
                if env.time_step % FRAME_EVERY == 0:
                    frames.append(compose_frame(env, instruction, label, footer))

            executor.run([stage], on_step=record)
            stage_log.append({"stage": label, "policy_ok": None, "assisted": False})
    else:
        from data.record import SUBTASK_VOCAB, stage_signature
        from policy.export_openvino import DEFAULT_CKPT
        from policy.ov_policy import OVActPolicy
        from policy.rollout import run_stage_policy, run_stage_scripted

        policy = OVActPolicy(args.policy, DEFAULT_CKPT, device=args.policy_device)
        executor = Executor(env)
        executor.reset()
        for stage in plan:
            label = stage_text(stage)
            raw = []
            ok = run_stage_policy(env, policy, stage, SUBTASK_VOCAB.index(stage_signature(stage)), raw)
            frames.extend(compose_frame(env, instruction, f"{label}  (learned policy)", footer, main_frame=frame)
                          for frame in raw)
            assisted = False
            if not ok:
                run_stage_scripted(env, executor, stage)
                assisted = True
            stage_log.append({"stage": label, "policy_ok": ok, "assisted": assisted})

    result = env.success()
    checklist = {k.replace("_", " "): v for k, v in result.items() if k != "all"}
    final = compose_frame(env, instruction, "done" if result["all"] else "finished with misses", footer, checklist)
    frames.extend([final] * int(END_HOLD_S * VIDEO_FPS))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=VIDEO_FPS)
    log = {"instruction": instruction, "transcription_s": transcription_s, "plan": plan,
           "planner": {k: v for k, v in info.items() if k != "raw"}, "executor": args.executor,
           "seed": args.seed, "variation": env.variation, "stages": stage_log, "success": result}
    args.out.with_suffix(".json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    env.close()
    print(f"success: {result}\nwrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
