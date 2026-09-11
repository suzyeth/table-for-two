"""Trace one stage of the scripted plan: hand targets, grip, contacts, object motion, close-up frames.

Runs the plan up to the stage before, then steps the chosen stage while logging,
every few control steps, each active arm's TCP, gripper command and angle, the
arm-object contacts and the watched object's position and holder, and renders
close-ups around that object into ``out/trace_stage<N>.png``.

Run:  .venv\\Scripts\\python.exe -m tools.trace_stage --stage 1 --object fork --seed 0
"""
import argparse

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sim.env import ARMS, DinnerTableEnv
from sim.task import DEFAULT_PLAN, Executor
from tools.grasp_lab import ROOT

LOG_EVERY = 10
TILE = (300, 400)


def arm_contacts(env, arm):
    """'geom->body(dist mm)' for contacts between ``arm`` and anything that is not the arm."""
    m, d = env.model, env.data
    found = []
    for k in range(d.ncon):
        c = d.contact[k]
        names = [m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]
        mine = [n.startswith(arm) for n in names]
        if any(mine) and not all(mine):
            geom = (c.geom1, c.geom2)[mine.index(True)]
            found.append(f"{m.geom(geom).name or geom}->{names[mine.index(False)]}({c.dist * 1000:.1f})")
    return found


def main():
    parser = argparse.ArgumentParser(description="Step through one plan stage with diagnostics.")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--object", required=True, help="object to watch and frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiles", type=int, default=11)
    parser.add_argument("--every", type=int, default=25, help="control steps between close-ups")
    parser.add_argument("--azimuth", type=float, default=90)
    parser.add_argument("--elevation", type=float, default=-25)
    parser.add_argument("--distance", type=float, default=0.28)
    args = parser.parse_args()

    env = DinnerTableEnv(obs_cameras=())
    env.reset(args.seed)
    executor = Executor(env)
    executor.reset()
    executor.run(DEFAULT_PLAN[:args.stage])
    m, d = env.model, env.data
    renderer = mujoco.Renderer(m, *TILE)
    tiles, step = [], 0
    stage = DEFAULT_PLAN[args.stage]
    print(f"stage {args.stage}: {stage}")

    def snap():
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = env.object_frame(args.object)[0]
        cam.distance, cam.azimuth, cam.elevation = args.distance, args.azimuth, args.elevation
        renderer.update_scene(d, camera=cam)
        tile = Image.fromarray(renderer.render())
        ImageDraw.Draw(tile).text((6, 6), f"step {step} holder {env.holder(args.object)}", fill=(255, 255, 0))
        tiles.append(tile)

    parts = executor.expand(stage)
    for part, sub in enumerate(parts):
        gens = {arm: factory() for arm, factory in sub.items()}
        print(f"-- part {part}: {list(sub)}")
        while gens:
            for arm in list(gens):
                try:
                    next(gens[arm])
                except StopIteration:
                    del gens[arm]
            if not gens:
                break
            env.step(np.concatenate([executor.skills[a].cmd for a in ARMS]))
            step += 1
            if step % LOG_EVERY == 0:
                origin = env.object_frame(args.object)[0]
                for arm in sub:
                    sk = executor.skills[arm]
                    print(f"step {step:4d} {arm:6s} tcp {np.round(env.tcp(arm), 3)} grip {sk.cmd[5]:+.2f}/"
                          f"{env.arm_qpos(arm)[5]:+.2f} {args.object} {np.round(origin, 3)} "
                          f"holder {env.holder(args.object)} contacts {arm_contacts(env, arm)[:3]}")
            if step % args.every == 0 and len(tiles) < args.tiles:
                snap()
    snap()
    columns = 4
    sheet = Image.new("RGB", (TILE[1] * columns, TILE[0] * ((len(tiles) + columns - 1) // columns)))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * TILE[1], (index // columns) * TILE[0]))
    path = ROOT / "out" / f"trace_stage{args.stage}.png"
    sheet.save(path)
    print("warnings:", executor.warnings()[:10])
    print("success:", env.success())
    print(f"close-ups -> {path}")


if __name__ == "__main__":
    main()
