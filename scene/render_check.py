"""Smoke test for the dual-arm scene: render every camera and a short motion clip.

Outputs (in ``out/``):
  cameras.png  - one tile per scene camera at the initial state
  motion.gif   - both arms driven through a few poses, drawer pulled open
It also prints joint tracking error so we know the actuators respond.

Run:  .venv\\Scripts\\python.exe scene\\render_check.py
"""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = ROOT / "scene" / "bimanual_table.xml"
OUT_DIR = ROOT / "out"
WIDTH, HEIGHT = 480, 360

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Joint targets (radians) per keyframe, same for both arms except mirrored pan.
POSES = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.4, -0.6, 0.8, 0.4, 0.0, 1.2],
    [-0.3, 0.5, -0.4, 0.8, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]
SECONDS_PER_POSE = 1.0
GIF_FPS = 20


def render_cameras(model, data, renderer):
    tiles = []
    for cam_id in range(model.ncam):
        renderer.update_scene(data, camera=cam_id)
        tiles.append(renderer.render())
    return np.concatenate(tiles, axis=1)


def set_arm_targets(model, data, pose):
    for prefix, pan_sign in (("left_", 1.0), ("right_", -1.0)):
        for joint, target in zip(ARM_JOINTS, pose):
            value = target * pan_sign if joint == "shoulder_pan" else target
            data.actuator(prefix + joint).ctrl = value


def tracking_error(model, data, pose):
    errors = []
    for prefix, pan_sign in (("left_", 1.0), ("right_", -1.0)):
        for joint, target in zip(ARM_JOINTS, pose):
            value = target * pan_sign if joint == "shoulder_pan" else target
            errors.append(abs(data.joint(prefix + joint).qpos[0] - value))
    return max(errors)


def run_motion(model, data, renderer):
    frames = []
    steps_per_pose = int(SECONDS_PER_POSE / model.opt.timestep)
    frame_every = max(1, int(1.0 / (GIF_FPS * model.opt.timestep)))
    drawer = data.joint("drawer_slide")
    for i, pose in enumerate(POSES):
        set_arm_targets(model, data, pose)
        for step in range(steps_per_pose):
            if i == 1:  # scripted drawer pull so we can see it slide
                drawer.qpos[0] = min(0.08, drawer.qpos[0] + 0.08 / steps_per_pose)
            mujoco.mj_step(model, data)
            if step % frame_every == 0:
                renderer.update_scene(data, camera="operator")
                frames.append(Image.fromarray(renderer.render()))
        print(f"pose {i}: max joint tracking error = {tracking_error(model, data, pose):.3f} rad")
    return frames


def main():
    OUT_DIR.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        Image.fromarray(render_cameras(model, data, renderer)).save(OUT_DIR / "cameras.png")
        frames = run_motion(model, data, renderer)
    frames[0].save(OUT_DIR / "motion.gif", save_all=True, append_images=frames[1:],
                   duration=int(1000 / GIF_FPS), loop=0)

    for name in ("plate", "mug", "bottle", "spoon", "fork"):
        print(f"{name:6s} final pos = {np.round(data.body(name).xpos, 3)}")
    print(f"drawer opened to {data.joint('drawer_slide').qpos[0]:.3f} m")
    print(f"wrote {OUT_DIR / 'cameras.png'} and {OUT_DIR / 'motion.gif'} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
