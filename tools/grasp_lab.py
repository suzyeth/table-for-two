"""Single-arm grasp lab: plan and verify contact-only grasps, one object at a time.

No weld or attach constraint exists here - an object rises only if the finger
pads hold it by friction. For each object the lab
  1. generates grasp candidates (approach tilt, sideways skew, grasp height,
     jaw orientation),
  2. keeps those the arm can actually reach (IK within 2 mm / 3 deg) without any
     arm link touching the table or the object at pre-grasp or grasp pose,
  3. runs the first feasible candidate: straight-line approach, close past
     contact so the servo keeps squeezing, lift 10 cm and hold,
  4. measures rise and slip relative to the fingers, sweeping mass and friction
     (the perturbations the brief scores).

Why sideways skew: the SO-101 has 5 joints before the gripper and its TCP sits
1.2 cm off the wrist-roll axis. Pointing the fingers down at a tilt *and*
rolling the jaws to squeeze horizontally leaves a 5-8 mm residual when the
approach is exactly radial; letting the approach swing a few degrees sideways
can restore an exact solution.

Objects are sized for the SO-101's working volume (a hobby-scale arm whose
top-down reach tops out ~8 cm above the table), like the cups used in real
SO-101 demos. The plate is a deep plate (low wall): a top-down pinch on its
wall gives a vertical contact line that resists pivoting, which a flat rim
pinch cannot.

Gripper geometry (menagerie SO-101, gripper frame): the fixed jaw's pad plane is
at x = -0.0081 and the moving jaw closes toward it along +x; fingertips are at
z = -0.10; the ``gripperframe`` site is at (0.012, -0.0002, -0.0981).

Run:  .venv\\Scripts\\python.exe -m tools.grasp_lab --object bottle
"""
import argparse
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from sim.ik import ArmIK, gripper_rotation

ROOT = Path(__file__).resolve().parent.parent
SO101 = ROOT / "third_party" / "mujoco_menagerie" / "robotstudio_so101"
PREFIX = "a_"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FIXED_PAD_X = -0.0081
SITE_LOCAL = np.array([0.012, -0.000218, -0.098127])
OPEN = 0.8  # widest fingertip gap (~55 mm)
SQUEEZE = -0.17  # commanded past contact so the servo presses the pads together
PREGRASP = 0.03  # top-down reach ends ~8 cm above the table, so the stand-off stays short
LIFT = 0.10
MAX_IK_POS = 0.002
MAX_IK_ROT = np.deg2rad(3)
UP = np.array([0.0, 0.0, 1.0])
YAW_OFFSETS_DEG = (0, 8, -8, 16, -16)
# The fixed jaw's convex collision hull reaches ~5 mm below the fingertip spheres,
# so fingers resting on the table beside a flat object read as a few mm of overlap.
FINGER_TABLE_ALLOWANCE = 0.007
LINK_PENETRATION = 0.001

# Object dimensions (m) and default placement in front of the arm base.
BOTTLE = {"radius": 0.016, "half_height": 0.035, "xy": (0.20, 0.0)}
MUG = {"radius": 0.020, "half_height": 0.030, "xy": (0.20, 0.0)}
PLATE = {"radius": 0.045, "height": 0.020, "wall": 0.003, "xy": (0.24, 0.0)}  # deep plate (low wall)
SHELL_SEGMENTS = 16
SPOON = {"half_size": (0.035, 0.006, 0.003), "xy": (0.20, 0.0)}


@dataclass(frozen=True)
class GraspSpec:
    """Where the object sits between the pads, and from which direction the hand comes."""
    name: str
    centre: tuple  # grasp centre in world frame (m)
    approach: tuple  # world direction the fingers point
    closing: tuple  # world direction fixed jaw -> moving jaw
    half_width: float  # half the object width across the jaws
    finger_overlap: float  # how far past the grasp centre the fingertips reach
    clearance: float = 0.004  # gap between fixed pad and object before closing


def add_shell_body(body, radius, height, wall, mass, friction, rgba):
    """Open-topped thin-walled cylinder (wall segments + base disc) on an existing body."""
    mid_r = radius - wall / 2
    seg_half = mid_r * np.tan(np.pi / SHELL_SEGMENTS) + 0.0006
    for i in range(SHELL_SEGMENTS):
        theta = 2 * np.pi * i / SHELL_SEGMENTS
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[wall / 2, seg_half, height / 2],
                      pos=[mid_r * np.cos(theta), mid_r * np.sin(theta), height / 2],
                      quat=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)], mass=mass * 0.8 / SHELL_SEGMENTS,
                      friction=friction, condim=4, rgba=rgba)
    # 8 mm base, as in the scene (a thinner disc lets stacked beads tunnel through it).
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, 0.004], pos=[0, 0, 0.004],
                  mass=mass * 0.2, friction=friction, condim=4, rgba=rgba)


def add_object(spec, name, mass, friction):
    """Realistic table-setting objects placed in front of the arm."""
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint()
    fr = [friction, 0.005, 0.0001]
    cyl, box = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX
    if name == "bottle":
        body.pos = [*BOTTLE["xy"], BOTTLE["half_height"] + 0.001]
        body.add_geom(type=cyl, size=[BOTTLE["radius"], BOTTLE["half_height"]], mass=mass, friction=fr, condim=4,
                      rgba=[0.3, 0.75, 0.4, 1])
    elif name == "mug":
        body.pos = [*MUG["xy"], MUG["half_height"] + 0.001]
        body.add_geom(type=cyl, size=[MUG["radius"], MUG["half_height"]], mass=mass, friction=fr, condim=4,
                      rgba=[0.2, 0.45, 0.8, 1])
    elif name == "plate":
        body.pos = [*PLATE["xy"], 0.0005]
        add_shell_body(body, PLATE["radius"], PLATE["height"], PLATE["wall"], mass, fr, [0.95, 0.95, 0.95, 1])
    elif name == "spoon":
        body.pos = [*SPOON["xy"], SPOON["half_size"][2] + 0.001]
        body.quat = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
        body.add_geom(type=box, size=list(SPOON["half_size"]), mass=mass, friction=fr, condim=4,
                      rgba=[0.75, 0.75, 0.8, 1])
    else:
        raise ValueError(f"unknown object '{name}'")


def _rotate_z(vector, degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1], vector[2]])


def candidates(name, obj_pos, base_pos):
    """Grasp candidates for an object, most natural first."""
    radial = np.array([obj_pos[0] - base_pos[0], obj_pos[1] - base_pos[1], 0.0])
    radial /= np.linalg.norm(radial)
    specs = []

    def tilted(direction, pitch_deg):
        pitch = np.deg2rad(pitch_deg)
        return np.cos(pitch) * direction - np.sin(pitch) * UP

    if name in ("bottle", "mug"):
        dims = BOTTLE if name == "bottle" else MUG
        radius, half_h = dims["radius"], dims["half_height"]
        top = obj_pos[2] + half_h
        across = np.cross(UP, radial)
        specs.append(GraspSpec(f"{name} top-down", (obj_pos[0], obj_pos[1], top - 0.012), tuple(-UP), tuple(across),
                               radius, 0.010))
        for pitch in (60, 45, 30):
            for yaw in YAW_OFFSETS_DEG:
                direction = _rotate_z(radial, yaw)
                approach = tilted(direction, pitch)
                closing = np.cross(UP, direction)  # horizontal, across the cylinder
                for height in (0.6, 0.75, 0.45):
                    centre = np.array([obj_pos[0], obj_pos[1], obj_pos[2] - half_h + 2 * half_h * height])
                    specs.append(GraspSpec(f"{name} pitch {pitch} yaw {yaw:+d} @ {height:.2f}h", tuple(centre),
                                           tuple(approach), tuple(closing), radius, radius + 0.006))
    elif name == "plate":
        # Top-down pinch on the wall nearest the arm; jaws close toward the plate centre,
        # so the fixed pad is outside the wall and the moving jaw presses from inside.
        wall_mid = PLATE["radius"] - PLATE["wall"] / 2
        wall_top = obj_pos[2] + PLATE["height"]
        # Same numbers as sim/grasping.py: moving jaw ~3 mm above the 8 mm floor.
        for yaw in (0, 15, -15):
            toward_centre = _rotate_z(radial, yaw)
            point = np.array([obj_pos[0], obj_pos[1], wall_top - 0.004]) - toward_centre * wall_mid
            specs.append(GraspSpec(f"plate wall top-down yaw {yaw:+d}", tuple(point), tuple(-UP),
                                   tuple(toward_centre), PLATE["wall"] / 2, -0.003))
    elif name == "spoon":
        # Fingertips stop ~1 mm above the table beside the 6 mm-thick handle.
        specs.append(GraspSpec("spoon top-down", tuple(obj_pos), tuple(-UP), (1, 0, 0), 0.006, 0.0015))
    return specs


def build(object_name, mass, friction):
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.meshdir = str((SO101 / "assets").resolve())
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05],
                            rgba=[0.55, 0.38, 0.24, 1])
    spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1])
    arm = mujoco.MjSpec.from_file(str(SO101 / "so101.xml"))
    arm.meshdir = str((SO101 / "assets").resolve())
    with contextlib.redirect_stderr(io.StringIO()):
        spec.attach(arm, prefix=PREFIX, frame=spec.worldbody.add_frame(pos=[0, 0, 0]))
        add_object(spec, object_name, mass, friction)
        return spec.compile()


def tcp_target(grasp):
    """TCP position that puts the grasp centre between the pads."""
    rot = gripper_rotation(grasp.approach, grasp.closing)
    local = np.array([FIXED_PAD_X + grasp.clearance + grasp.half_width, 0.0, SITE_LOCAL[2] + grasp.finger_overlap])
    return np.asarray(grasp.centre) + rot @ (SITE_LOCAL - local)


def _is_finger(model, geom):
    body = model.body(model.geom_bodyid[geom]).name
    return body in (PREFIX + "gripper", PREFIX + "moving_jaw_so101_v1") and "jaw" in (model.geom(geom).name or "jaw")


def arm_collision(model, data):
    """Name of the first arm/world geom pair in illegal contact, or None.

    Arm-to-table and arm-to-object contacts are illegal, except finger geoms
    touching the table within FINGER_TABLE_ALLOWANCE (needed for flat objects).
    """
    mujoco.mj_forward(model, data)
    for k in range(data.ncon):
        contact = data.contact[k]
        if contact.dist > -LINK_PENETRATION:
            continue
        geoms = (contact.geom1, contact.geom2)
        bodies = [model.body(model.geom_bodyid[g]).name for g in geoms]
        arm_side = [b.startswith(PREFIX) for b in bodies]
        if not any(arm_side) or all(arm_side):
            continue
        arm_geom = geoms[arm_side.index(True)]
        other_body = bodies[arm_side.index(False)]
        if other_body == "world" and _is_finger(model, arm_geom) and contact.dist > -FINGER_TABLE_ALLOWANCE:
            continue
        arm_name = model.geom(arm_geom).name or f"{model.body(model.geom_bodyid[arm_geom]).name}/geom{arm_geom}"
        return f"{arm_name} x {other_body} ({contact.dist * 1000:.1f} mm)"
    return None


def plan(model, data, ik, name, qadr, log=print):
    """First candidate that is reachable and collision-free; returns (spec, target, pre) or None."""
    base = data.xpos[model.body(PREFIX + "base").id].copy()
    obj_pos = data.xpos[model.body(name).id].copy()
    saved = data.qpos.copy()
    for grasp in candidates(name, obj_pos, base):
        target = tcp_target(grasp)
        pre = target - np.asarray(grasp.approach) * PREGRASP
        reason = None
        for label, point in (("grasp", target), ("pre-grasp", pre)):
            q, pos_err = ik.solve(data.qpos, point, approach=grasp.approach, closing=grasp.closing)
            if pos_err > MAX_IK_POS or ik.last_rot_err > MAX_IK_ROT:
                reason = f"{label} IK {pos_err * 1000:.1f} mm / {np.degrees(ik.last_rot_err):.1f} deg"
                break
            data.qpos[qadr[:5]] = q
            data.qpos[qadr[5]] = OPEN
            hit = arm_collision(model, data)
            if hit:
                reason = f"{label} collision {hit}"
                break
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)
        log(f"  candidate {grasp.name:34s} {'feasible' if reason is None else 'rejected: ' + reason}")
        if reason is None:
            return grasp, target, pre
    return None


def trial(object_name, mass, friction, render_path=None, verbose=False):
    model = build(object_name, mass, friction)
    data = mujoco.MjData(model)
    ik = ArmIK(model, PREFIX)
    act = [model.actuator(PREFIX + j).id for j in JOINTS]
    qadr = [model.jnt_qposadr[model.joint(PREFIX + j).id] for j in JOINTS]
    obj = model.body(object_name).id
    data.qpos[qadr[5]] = OPEN
    data.ctrl[act[5]] = OPEN
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)
    planned = plan(model, data, ik, object_name, qadr, log=print if verbose else (lambda *_: None))
    if planned is None:
        return None
    grasp, target, pre = planned
    approach = np.asarray(grasp.approach)
    q_pre, _ = ik.solve(data.qpos, pre, approach=approach, closing=grasp.closing)
    data.qpos[qadr[:5]] = q_pre
    data.qpos[qadr[5]] = OPEN
    data.ctrl[act[:5]] = q_pre
    data.ctrl[act[5]] = OPEN
    mujoco.mj_forward(model, data)

    def go(q_target, grip, seconds):
        start = data.ctrl[act].copy()
        goal = np.append(q_target, grip)
        steps = int(seconds / model.opt.timestep)
        for i in range(steps):
            s = (i + 1) / steps
            data.ctrl[act] = start + (goal - start) * (s * s * (3 - 2 * s))
            mujoco.mj_step(model, data)

    def settle(seconds):
        for _ in range(int(seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)

    settle(0.3)
    for k in range(1, 9):  # straight-line approach
        q, _ = ik.solve(data.qpos, pre + (target - pre) * k / 8, approach=approach, closing=grasp.closing,
                        restarts=False)
        go(q, OPEN, 0.15)
    settle(0.2)
    tcp_err = np.linalg.norm(data.site_xpos[ik.site_id] - target)
    go(data.ctrl[act[:5]].copy(), SQUEEZE, 0.6)
    settle(0.4)
    jaw_angle = float(data.qpos[qadr[5]])
    rel0 = data.xpos[obj] - data.site_xpos[ik.site_id]
    z0 = data.xpos[obj][2]
    for k in range(1, 6):
        q, _ = ik.solve(data.qpos, target + np.array([0, 0, LIFT * k / 5]), approach=None, restarts=False)
        go(q, SQUEEZE, 0.3)
    settle(1.0)
    rise = data.xpos[obj][2] - z0
    slip = np.linalg.norm((data.xpos[obj] - data.site_xpos[ik.site_id]) - rel0)
    tilt = float(np.degrees(np.arccos(np.clip(data.xmat[obj].reshape(3, 3)[2, 2], -1, 1))))
    if render_path:
        renderer = mujoco.Renderer(model, 360, 480)
        tiles = []
        for azimuth in (90, 180):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = data.xpos[obj]
            cam.distance, cam.azimuth, cam.elevation = 0.25, azimuth, -15
            renderer.update_scene(data, camera=cam)
            tiles.append(Image.fromarray(renderer.render()))
        sheet = Image.new("RGB", (960, 360))
        sheet.paste(tiles[0], (0, 0))
        sheet.paste(tiles[1], (480, 0))
        sheet.save(render_path)
    return {"grasp": grasp.name, "rise_cm": rise * 100, "slip_mm": slip * 1000, "jaw": jaw_angle,
            "tcp_err_mm": tcp_err * 1000, "tilt_deg": tilt}


def main():
    parser = argparse.ArgumentParser(description="Contact-only grasp trials for one object.")
    parser.add_argument("--object", default="bottle")
    parser.add_argument("--masses", type=float, nargs="+", default=[0.15, 0.4])
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.7, 1.3])
    args = parser.parse_args()
    passed, total = 0, 0
    for index, mass in enumerate(args.masses):
        for jndex, friction in enumerate(args.frictions):
            first = index == 0 and jndex == 0
            render = ROOT / "out" / f"grasp_lab_{args.object}.png" if first else None
            result = trial(args.object, mass, friction, render, verbose=first)
            total += 1
            if result is None:
                print(f"{args.object} m={mass:.2f} mu={friction:.1f}: no feasible grasp candidate")
                continue
            ok = result["rise_cm"] > 0.8 * LIFT * 100 and result["slip_mm"] < 10
            passed += ok
            print(f"{args.object} m={mass:.2f} mu={friction:.1f} [{result['grasp']}]: rise {result['rise_cm']:5.1f} cm  "
                  f"slip {result['slip_mm']:5.1f} mm  tilt {result['tilt_deg']:4.1f} deg  jaw {result['jaw']:5.2f}  "
                  f"tcp_err {result['tcp_err_mm']:4.1f} mm  -> {'HELD' if ok else 'FAILED'}")
    print(f"{args.object}: {passed}/{total} held")


if __name__ == "__main__":
    main()
