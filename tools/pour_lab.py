"""Pour lab: can one SO-101 pour real (bead) water from a bottle into a mug?

Contact-only, like the grasp lab: the bottle is an open thin-walled shell with
24 beads inside (0.27 g each), the mug is an open shell standing on the table
(in the full task the other arm steadies it by its handle).

How a person pours, and what this lab copies:
  * grip the bottle *body* from the side, jaws horizontal, so the mouth stays
    free above the fingers (a top-down grip at the mouth puts the fingers in
    the stream and forces the hand upside down to pour);
  * lift, bring the mouth over the mug, then tilt the bottle about the grip
    with the wrist while the mouth stays over the mug.

The SO-101 has 5 joints before the gripper, so a pose fixes the controlled point
(3) plus the direction of the bottle axis (2); the spin about that axis is left
to the arm. For every tilt step the lab solves for the lowest lip point sitting
over the mug centre with the whole bottle a set clearance above the rim, and
retries with more clearance if the arm cannot reach it.

Run:  .venv\\Scripts\\python.exe -m tools.pour_lab
"""
import argparse
import contextlib
import io

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from scene.build_scene import realistic_gripper
from sim.ik import ArmIK
from tools.grasp_lab import FIXED_PAD_X, JOINTS, OPEN, PREFIX, ROOT, SITE_LOCAL, SO101, SQUEEZE, UP, add_shell_body, \
    arm_collision

# A jaws-horizontal side grip near the table is only reachable with the arm stretched out
# (pinch point >= ~0.30 m from the base at 2-5 cm height), so the bottle stands at the far
# edge of the arm's workspace; the pour itself happens 11-20 cm up, reachable from ~0.22 m.
BOTTLE = {"xy": (0.30, -0.08), "radius": 0.016, "height": 0.070, "wall": 0.002, "mass": 0.04}
MUG = {"xy": (0.26, 0.09), "radius": 0.028, "height": 0.070, "wall": 0.0025, "mass": 0.15}
BEADS = {"count": 24, "radius": 0.004, "mass": 0.00027}

PAD_CLEARANCE = 0.004  # gap between fixed pad and bottle before closing
GRASP_HEIGHTS = (0.030, 0.024, 0.038)  # grasp centre above the bottle base
FINGER_OVERLAPS = (0.016, 0.012, 0.020)  # fingertips reach this far past the bottle axis
PREGRASP = 0.03  # jaw pinch point this far above the bottle mouth before descending
LIFT = 0.10
TILTS_DEG = (0, 20, 40, 60, 70, 80, 88, 95, 100, 105, 110, 120, 130, 140)
CLEARANCES = np.arange(0.010, 0.080, 0.005)  # whole bottle above the mug rim
POUR_START_DEG = 70  # from here on water may leave, so the lip must stay low over the mug
POUR_MAX_CLEARANCE = 0.03
MAX_POS_ERR = 0.003
MAX_ROT_ERR = np.deg2rad(3)
MIN_RADIAL_APPROACH = 0.3  # fingers must point away from the arm base, not back at it


def build(bottle_friction):
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
    fr = [bottle_friction, 0.005, 0.0001]
    for name, params, rgba in (("bottle", BOTTLE, [0.3, 0.75, 0.4, 0.6]), ("mug", MUG, [0.2, 0.45, 0.8, 1])):
        body = spec.worldbody.add_body(name=name, pos=[*params["xy"], 0.0005])
        body.add_freejoint()
        add_shell_body(body, params["radius"], params["height"], params["wall"], params["mass"], fr, rgba)
    inner = BOTTLE["radius"] - BOTTLE["wall"] - BEADS["radius"] - 0.0005
    for i in range(BEADS["count"]):
        layer, slot = divmod(i, 6)
        theta = 2 * np.pi * slot / 6 + layer * 0.5
        radial = inner * (0.55 if slot % 2 else 0.95)
        bead = spec.worldbody.add_body(name=f"water_{i}", pos=[BOTTLE["xy"][0] + radial * np.cos(theta),
                                                               BOTTLE["xy"][1] + radial * np.sin(theta),
                                                               0.009 + BEADS["radius"] + layer * 2.05 * BEADS["radius"]])
        bead.add_freejoint()
        bead.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BEADS["radius"], 0, 0], mass=BEADS["mass"],
                      friction=[0.3, 0.001, 0.0001], condim=3, solref=[0.004, 1.0], rgba=[0.35, 0.6, 1.0, 1])
    with contextlib.redirect_stderr(io.StringIO()):
        spec.attach(arm, prefix=PREFIX, frame=spec.worldbody.add_frame(pos=[0, 0, 0]))
        realistic_gripper(spec, PREFIX, pad_friction=bottle_friction)  # the sweep sets the pads
        return spec.compile()


def bead_census(model, data):
    """Beads in the mug, still in the bottle, and anywhere else."""
    counts = {"mug": 0, "bottle": 0, "spilled": 0}
    frames = {}
    for name, params in (("mug", MUG), ("bottle", BOTTLE)):
        body = model.body(name).id
        frames[name] = (data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy(), params)
    for i in range(BEADS["count"]):
        p = data.xpos[model.body(f"water_{i}").id]
        where = "spilled"
        for name, (origin, rot, params) in frames.items():
            local = (p - origin) @ rot
            if np.linalg.norm(local[:2]) < params["radius"] - params["wall"] and -0.002 < local[2] < params["height"]:
                where = name
        counts[where] += 1
    return counts


class Lab:
    """One pouring trial: model, arm handles and motion helpers."""

    def __init__(self, friction):
        self.model = build(friction)
        self.data = mujoco.MjData(self.model)
        self.ik = ArmIK(self.model, PREFIX)
        self.act = [self.model.actuator(PREFIX + j).id for j in JOINTS]
        self.qadr = [self.model.jnt_qposadr[self.model.joint(PREFIX + j).id] for j in JOINTS]
        self.bottle = self.model.body("bottle").id
        self.mug = self.model.body("mug").id
        self.base = None
        self.tiles = []
        self.renderer = None

    # ------------------------------------------------------------ motion
    def go(self, q_target, grip, seconds):
        start = self.data.ctrl[self.act].copy()
        goal = np.append(q_target, grip)
        steps = max(1, int(seconds / self.model.opt.timestep))
        for i in range(steps):
            s = (i + 1) / steps
            self.data.ctrl[self.act] = start + (goal - start) * (s * s * (3 - 2 * s))
            mujoco.mj_step(self.model, self.data)

    def settle(self, seconds):
        for _ in range(int(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def qpos_with(self, q):
        qpos = self.data.qpos.copy()
        qpos[self.qadr[:5]] = q
        return qpos

    def gripper_frame(self, q):
        """(position, rotation) of the gripper body for arm joints ``q`` (kinematics only)."""
        scratch = self.ik.scratch
        scratch.qpos[:] = self.qpos_with(q)
        mujoco.mj_kinematics(self.model, scratch)
        body = self.ik.gripper_body
        return scratch.xpos[body].copy(), scratch.xmat[body].reshape(3, 3).copy()

    def collides(self, q):
        saved = self.data.qpos.copy()
        self.data.qpos[:] = self.qpos_with(q)
        self.data.qpos[self.qadr[5]] = OPEN
        hit = arm_collision(self.model, self.data)
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return hit

    def ok(self, err):
        return err < MAX_POS_ERR and self.ik.last_rot_err < MAX_ROT_ERR

    # ------------------------------------------------------------- grasp
    def plan_side_grasp(self):
        """Side grasp on the bottle body: jaws horizontal, fingers pointing away from the base."""
        bottle_base = self.data.xpos[self.bottle].copy()
        radial = bottle_base - self.base
        radial[2] = 0.0
        radial /= np.linalg.norm(radial)
        for sign in (1.0, -1.0):  # which way up the jaw width points (sets the moving jaw side)
            for height in GRASP_HEIGHTS:
                for overlap in FINGER_OVERLAPS:
                    pinch = np.array([FIXED_PAD_X + PAD_CLEARANCE + BOTTLE["radius"], 0.0, SITE_LOCAL[2] + overlap])
                    centre = bottle_base + UP * height
                    tag = f"  sign {sign:+.0f} h {height * 100:.1f} overlap {overlap * 1000:.0f}:"
                    q, err = self.ik.solve(self.data.qpos, centre, approach=None, lateral=sign * UP, point=pinch)
                    if not self.ok(err):
                        print(tag, f"IK {err * 1000:.1f} mm / {np.degrees(self.ik.last_rot_err):.1f} deg")
                        continue
                    _, rot = self.gripper_frame(q)
                    approach = -rot[:, 2]
                    hit = self.collides(q)
                    if np.dot(approach, radial) < MIN_RADIAL_APPROACH or hit:
                        print(tag, f"approach.radial {np.dot(approach, radial):.2f} collision {hit}")
                        continue
                    # The finger plates straddle the bottle sideways, so the open hand comes straight
                    # down over it (a horizontal back-off is out of reach this low).
                    pre = centre + UP * (BOTTLE["height"] - height + PREGRASP)
                    q_pre, err = self.ik.solve(self.qpos_with(q), pre, approach=None, lateral=sign * UP, point=pinch,
                                               restarts=False)
                    hit = self.collides(q_pre)
                    if not self.ok(err) or hit:
                        print(tag, f"pre-grasp IK {err * 1000:.1f} mm collision {hit}")
                        continue
                    return {"sign": sign, "pinch": pinch, "centre": centre, "pre": pre, "q": q, "q_pre": q_pre,
                            "label": f"side grasp {'up' if sign > 0 else 'down'} h={height * 100:.1f}cm "
                                     f"overlap={overlap * 1000:.0f}mm"}
        return None

    def execute_grasp(self, grasp):
        lateral = grasp["sign"] * UP
        self.data.qpos[self.qadr[:5]] = grasp["q_pre"]
        self.data.ctrl[self.act[:5]] = grasp["q_pre"]
        mujoco.mj_forward(self.model, self.data)
        self.settle(0.3)
        for k in range(1, 11):  # straight down around the bottle
            point = grasp["pre"] + (grasp["centre"] - grasp["pre"]) * k / 10
            q, _ = self.ik.solve(self.data.qpos, point, approach=None, lateral=lateral, point=grasp["pinch"],
                                 restarts=False)
            self.go(q, OPEN, 0.15)
        self.settle(0.2)
        self.go(self.data.ctrl[self.act[:5]].copy(), SQUEEZE, 0.6)
        self.settle(0.4)
        for k in range(1, 6):  # straight up
            point = grasp["centre"] + UP * LIFT * k / 5
            q, _ = self.ik.solve(self.data.qpos, point, approach=None, lateral=lateral, point=grasp["pinch"],
                                 restarts=False)
            self.go(q, SQUEEZE, 0.3)
        self.settle(0.5)

    # -------------------------------------------------------------- pour
    def bottle_in_gripper(self):
        """Bottle lip ring and whole outline in gripper-body coordinates, measured as held."""
        body = self.ik.gripper_body
        g_pos, g_rot = self.data.xpos[body].copy(), self.data.xmat[body].reshape(3, 3).copy()
        b_pos, b_rot = self.data.xpos[self.bottle].copy(), self.data.xmat[self.bottle].reshape(3, 3).copy()
        ring = [np.array([np.cos(a), np.sin(a), 0.0]) * BOTTLE["radius"]
                for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        to_gripper = lambda p: g_rot.T @ (p - g_pos)
        lip = [to_gripper(b_pos + b_rot @ (r + UP * BOTTLE["height"])) for r in ring]
        outline = lip + [to_gripper(b_pos + b_rot @ r) for r in ring]
        axis = g_rot.T @ b_rot[:, 2]  # bottle axis (base -> mouth) in gripper coordinates
        return lip, outline, axis

    def pour_pose(self, q_seed, tilt, lean, rim, lip, outline, axis_g, clearance, restarts):
        """Arm joints putting the lowest lip over the mug centre with the bottle tilted ``tilt`` toward ``lean``."""
        bottle_axis = np.cos(tilt) * UP + np.sin(tilt) * lean
        q = q_seed
        err = np.inf
        for iteration in range(4):  # the lowest lip point depends on the pose: refine a few times
            _, rot = self.gripper_frame(q)
            lip_world = [rot @ p for p in lip]
            index = int(np.argmin([p[2] for p in lip_world]))
            lowest = min((rot @ p)[2] for p in outline)
            target = np.array([rim[0], rim[1], rim[2] + clearance + (lip_world[index][2] - lowest)])
            # Align the held bottle's axis (a fixed gripper-frame vector) with the wanted direction.
            q, err = self._solve_axis(q, target, lip[index], axis_g, bottle_axis, restarts and iteration == 0)
        return q, err

    def _solve_axis(self, q_seed, target, point, axis_g, direction, restarts):
        """IK for a point on the gripper plus one gripper-frame axis pointing along ``direction``."""
        # The held bottle axis is close to +/- gripper y; use the exact measured axis by
        # expressing the wanted direction for gripper y: rotate ``direction`` by the small
        # offset between the measured axis and y.
        y_axis = np.array([0.0, 1.0, 0.0]) * np.sign(axis_g[1])
        offset = _rotation_between(axis_g, y_axis)
        _, rot = self.gripper_frame(q_seed)
        wanted_y = rot @ offset @ rot.T @ direction
        return self.ik.solve(self.qpos_with(q_seed), target, approach=None, lateral=np.sign(axis_g[1]) * wanted_y,
                             point=point, restarts=restarts)

    def plan_pour(self, lean):
        lip, outline, axis_g = self.bottle_in_gripper()
        rim = self.data.xpos[self.mug] + UP * MUG["height"]
        q = self.data.qpos[self.qadr[:5]].copy()
        plan = []
        for index, tilt_deg in enumerate(TILTS_DEG):
            chosen = None
            limit = POUR_MAX_CLEARANCE if tilt_deg >= POUR_START_DEG else CLEARANCES[-1]
            for clearance in CLEARANCES[CLEARANCES <= limit + 1e-9]:
                q_try, err = self.pour_pose(q, np.deg2rad(tilt_deg), lean, rim, lip, outline, axis_g, clearance,
                                            restarts=index == 0)
                if self.ok(err):
                    chosen = (tilt_deg, clearance, q_try)
                    break
            if chosen:
                plan.append(chosen)
                q = chosen[2]
        return plan

    # ------------------------------------------------------------ render
    def snapshot(self, label, focus):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 300, 400)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = focus
        cam.distance, cam.azimuth, cam.elevation = 0.42, 200, -12
        self.renderer.update_scene(self.data, camera=cam)
        tile = Image.fromarray(self.renderer.render())
        ImageDraw.Draw(tile).text((6, 6), label, fill=(255, 255, 0))
        self.tiles.append(tile)

    def save_sheet(self, path, columns=4):
        rows = (len(self.tiles) + columns - 1) // columns
        sheet = Image.new("RGB", (400 * columns, 300 * rows))
        for index, tile in enumerate(self.tiles):
            sheet.paste(tile, ((index % columns) * 400, (index // columns) * 300))
        sheet.save(path)


def _rotation_between(a, b):
    """Rotation matrix turning unit vector ``a`` onto unit vector ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def run(friction, render=True):
    lab = Lab(friction)
    m, d = lab.model, lab.data
    d.qpos[lab.qadr[5]] = OPEN
    d.ctrl[lab.act[5]] = OPEN
    mujoco.mj_forward(m, d)
    lab.settle(1.5)
    lab.base = d.xpos[m.body(PREFIX + "base").id].copy()

    grasp = lab.plan_side_grasp()
    if grasp is None:
        print("no feasible side grasp")
        return None
    lab.execute_grasp(grasp)
    rim = d.xpos[lab.mug] + UP * MUG["height"]
    held = d.xpos[lab.bottle][2] > 0.05
    census = bead_census(m, d)
    print(f"{grasp['label']}: lifted={held}  beads still in bottle {census['bottle']}")
    if not held:
        return None

    towards = rim - d.xpos[lab.bottle]
    towards[2] = 0.0
    radial_out = rim - lab.base
    radial_out[2] = 0.0
    radial_out /= np.linalg.norm(radial_out)
    tangent = np.cross(UP, radial_out)
    best = None
    for name, lean in (("away from base", radial_out), ("left", tangent), ("right", -tangent)):
        plan = lab.plan_pour(lean)
        reach = max((p[0] for p in plan), default=-1)
        print(f"  lean {name:15s}: " + ", ".join(f"{p[0]}->{p[1] * 100:.1f}" for p in plan))
        if best is None or (reach, -np.mean([p[1] for p in plan] or [1])) > best[0]:
            best = ((reach, -np.mean([p[1] for p in plan] or [1])), name, plan)
    _, lean_name, plan = best
    print(f"  using lean {lean_name}")

    focus = rim + UP * 0.04
    if render:
        lab.snapshot("lifted", focus)
    for tilt, clearance, q in plan:
        lab.go(q, SQUEEZE, 0.5 if tilt < POUR_START_DEG else 0.8)
        if tilt >= 95:
            lab.settle(0.4)
        census = bead_census(m, d)
        tilt_now = np.degrees(np.arccos(np.clip(d.xmat[lab.bottle][8], -1, 1)))
        print(f"   tilt {tilt:3d} (bottle {tilt_now:5.1f} deg, clearance {clearance * 100:.1f} cm): "
              f"mug {census['mug']:2d}  bottle {census['bottle']:2d}  spilled {census['spilled']:2d}")
        if render:
            lab.snapshot(f"tilt {tilt}  mug {census['mug']}  spilled {census['spilled']}", focus)
    lab.settle(2.0)
    census = bead_census(m, d)
    held = d.xpos[lab.bottle][2] > 0.05
    mug_shift = np.linalg.norm(d.xpos[lab.mug][:2] - np.array(MUG["xy"])) * 1000
    print(f"friction {friction:.1f}: bottle held={held}  water in mug {census['mug']}/{BEADS['count']}  "
          f"still in bottle {census['bottle']}  spilled {census['spilled']}  mug moved {mug_shift:.1f} mm")
    if render:
        lab.snapshot("end", focus)
        lab.save_sheet(ROOT / "out" / "pour_lab_sequence.png")
    return census


def main():
    parser = argparse.ArgumentParser(description="Bead-water pouring trial with one SO-101.")
    parser.add_argument("--frictions", type=float, nargs="+", default=[1.0, 0.7])
    args = parser.parse_args()
    for index, friction in enumerate(args.frictions):
        run(friction, render=index == 0)


if __name__ == "__main__":
    main()
