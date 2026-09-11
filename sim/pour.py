"""Pouring with a side grip, as a person does it.

The bottle is gripped by its body with the jaws horizontal, so the mouth stays
free above the fingers; the open hand comes straight down around the bottle
(a horizontal approach is out of reach this low). To pour, the hand keeps the
lowest point of the bottle lip over the mug centre while tilting the bottle
toward it step by step; for each step the arm is solved for the lowest reachable
clearance above the rim. A 5-joint arm fixes the lip position (3) and the
bottle axis direction (2) and leaves the spin about that axis free.

Validated in ``tools/pour_lab.py``: 24/24, 24/24 and 22/24 beads land in the
scene's mug at bottle friction 1.0, 0.7 and 1.3, and the mug does not move.
"""
import numpy as np

from scene.build_scene import BOTTLE, MUG, TABLE_TOP_Z
from sim.env import CONTROL_HZ, GRIPPER_OPEN, GRIPPER_SQUEEZE
from sim.grasping import FIXED_PAD_X, PAD_CLEARANCE, SITE_LOCAL, UP, horizontal

# Grasp centre above the bottle base, preferred first. Gripping higher reaches bottles that
# stand a little closer to the arm (a jaws-horizontal grip near the table needs >= ~30 cm);
# at 5 cm the fingers still end ~1 cm below the 7 cm mouth. With 3.0-3.8 cm only, 28/30
# randomised seeds were reachable; with these, 30/30.
SIDE_HEIGHTS = (0.030, 0.024, 0.038, 0.044, 0.050)
SIDE_OVERLAPS = (0.016, 0.012, 0.020, 0.008)  # fingertips this far past the bottle axis
SIDE_PRE = 0.03  # pinch point this far above the mouth before descending
BOTTLE_LIFT = 0.10
MIN_RADIAL_APPROACH = 0.3  # fingers point away from the arm base
SIDE_POS_TOL = 0.003
SIDE_ROT_TOL = np.deg2rad(3)

TILTS_DEG = (0, 20, 40, 60, 70, 80, 88, 95, 100, 105, 110, 120, 130, 140)
CLEARANCES = np.arange(0.010, 0.080, 0.005)  # whole bottle above the mug rim
POUR_START_DEG = 70  # from here on water may leave, so the lip stays low over the mug
POUR_MAX_CLEARANCE = 0.03
POUR_HOLD_STEPS = 20


def _rotation_between(a, b):
    """Rotation matrix turning unit vector ``a`` onto unit vector ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


class PourMixin:
    """Bottle skills for ``ArmSkills`` (uses its frame/solve/line/grip/wait helpers)."""

    side = None  # the executed side grasp: pinch point and jaw-width sign

    # -------------------------------------------------------------- grasp
    def _plan_side_grasp(self):
        origin, _ = self.env.object_frame("bottle")
        radial = horizontal(origin - self.env.arm_base(self.arm))
        for sign in (1.0, -1.0):  # jaw width up or down (the wrist camera must not face the table)
            for height in SIDE_HEIGHTS:
                for overlap in SIDE_OVERLAPS:
                    pinch = np.array([FIXED_PAD_X + PAD_CLEARANCE + BOTTLE["radius"], 0.0, SITE_LOCAL[2] + overlap])
                    centre = origin + UP * height
                    orient = {"approach": None, "lateral": sign * UP, "point": pinch}
                    q, err = self.ik.solve(self._qpos(), centre, restarts=True, **orient)
                    if err > SIDE_POS_TOL or self.ik.last_rot_err > SIDE_ROT_TOL:
                        continue
                    if -self.frame(q)[1][:, 2] @ radial < MIN_RADIAL_APPROACH:
                        continue
                    if sign < 0 and self.frame(q)[1][2, 0] < 0:
                        continue
                    pre = centre + UP * (BOTTLE["height"] - height + SIDE_PRE)
                    q_pre, err = self.ik.solve(self._qpos(q), pre, restarts=False, **orient)
                    if err > SIDE_POS_TOL or self.ik.last_rot_err > SIDE_ROT_TOL:
                        continue
                    return {"orient": orient, "centre": centre, "pre": pre, "q_pre": q_pre, "sign": sign}
        return None

    def side_pick_bottle(self):
        """Lower the open hand around the bottle body, squeeze and lift."""
        plan = self._plan_side_grasp()
        if plan is None:
            self.warnings.append(f"{self.arm} no reachable side grasp on the bottle")
            return
        self.side = plan
        yield from self.rise()
        yield from self._to(plan["q_pre"], GRIPPER_OPEN, self.free_speed)
        yield from self.line(plan["centre"], plan["orient"], steps=10)
        yield from self.grip(GRIPPER_SQUEEZE)
        yield from self.wait(self.squeeze_settle)
        yield from self.line(plan["centre"] + UP * BOTTLE_LIFT, plan["orient"], steps=5)

    # --------------------------------------------------------------- pour
    def _bottle_in_gripper(self):
        """Bottle lip ring, outline and axis in gripper coordinates, measured as actually held."""
        d = self.env.data
        body = self.ik.gripper_body
        g_pos, g_rot = d.xpos[body].copy(), d.xmat[body].reshape(3, 3).copy()
        b_pos, b_rot = self.env.object_frame("bottle")
        ring = [np.array([np.cos(a), np.sin(a), 0.0]) * BOTTLE["radius"]
                for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        to_gripper = lambda p: g_rot.T @ (p - g_pos)
        lip = [to_gripper(b_pos + b_rot @ (r + UP * BOTTLE["height"])) for r in ring]
        outline = lip + [to_gripper(b_pos + b_rot @ r) for r in ring]
        return lip, outline, g_rot.T @ b_rot[:, 2]

    def _solve_axis(self, q_seed, target, point, axis_g, direction, restarts):
        """IK for a gripper-frame point plus the held bottle axis pointing along ``direction``.

        The bottle axis is a fixed vector in the gripper frame close to +/- gripper y, so the
        wanted direction is mapped onto gripper y through that small offset.
        """
        sign = np.sign(axis_g[1])
        offset = _rotation_between(axis_g, np.array([0.0, sign, 0.0]))
        _, rot = self.frame(q_seed)
        wanted_y = rot @ offset @ rot.T @ direction
        return self.ik.solve(self._qpos(q_seed), target, approach=None, lateral=sign * wanted_y, point=point,
                             restarts=restarts)

    def _pour_pose(self, q_seed, tilt, lean, rim, lip, outline, axis_g, clearance, restarts):
        bottle_axis = np.cos(tilt) * UP + np.sin(tilt) * lean
        q, err = q_seed, np.inf
        for iteration in range(4):  # the lowest lip point depends on the pose: refine
            _, rot = self.frame(q)
            lip_world = [rot @ p for p in lip]
            index = int(np.argmin([p[2] for p in lip_world]))
            lowest = min((rot @ p)[2] for p in outline)
            target = np.array([rim[0], rim[1], rim[2] + clearance + (lip_world[index][2] - lowest)])
            q, err = self._solve_axis(q, target, lip[index], axis_g, bottle_axis, restarts and iteration == 0)
        return q, err

    def _plan_pour(self, rim):
        lip, outline, axis_g = self._bottle_in_gripper()
        radial_out = horizontal(rim - self.env.arm_base(self.arm))
        lean = np.cross(UP, radial_out)  # tilt sideways, across the arm's reach
        q = self.cmd[:5].copy()
        plan = []
        for index, tilt_deg in enumerate(TILTS_DEG):
            limit = POUR_MAX_CLEARANCE if tilt_deg >= POUR_START_DEG else CLEARANCES[-1]
            for clearance in CLEARANCES[CLEARANCES <= limit + 1e-9]:
                q_try, err = self._pour_pose(q, np.deg2rad(tilt_deg), lean, rim, lip, outline, axis_g, clearance,
                                             restarts=index == 0)
                if err < SIDE_POS_TOL and self.ik.last_rot_err < SIDE_ROT_TOL:
                    plan.append((tilt_deg, clearance, q_try))
                    q = q_try
                    break
        return plan

    def pour_into(self, target="mug"):
        """Tilt the held bottle with its lip kept over ``target``, hold, and straighten up again."""
        rim = self.env.object_frame(target)[0] + UP * MUG["height"]
        plan = self._plan_pour(rim)
        if not plan or plan[-1][0] < 100:
            self.warnings.append(f"{self.arm} pour plan reaches only {plan[-1][0] if plan else 0} deg")
        for tilt, _, q in plan:
            yield from self._timed(q, GRIPPER_SQUEEZE, 0.5 if tilt < POUR_START_DEG else 0.8)
            if tilt >= 95:
                yield from self.wait(8)
        yield from self.wait(POUR_HOLD_STEPS)
        for _, _, q in reversed(plan[:-1]):
            yield from self._timed(q, GRIPPER_SQUEEZE, 0.3)

    def return_bottle(self, xy):
        """Stand the bottle back at ``xy``, open, and slide the hand up off it."""
        if self.side is None:
            return
        point = self.held_point("bottle")
        lateral = self.side["sign"] * UP
        rest = np.array([xy[0], xy[1], TABLE_TOP_Z])
        carry = {"approach": None, "lateral": lateral, "point": point}
        yield from self.line(rest + UP * 0.05, carry, steps=10, speed=self.free_speed)
        yield from self.lower_until_supported("bottle", rest, carry)
        yield from self.grip(GRIPPER_OPEN)
        yield from self.wait(6)
        pinch = self.side["orient"]["point"]
        leave = {"approach": None, "lateral": lateral, "point": pinch}
        # Slide the fixed finger off the bottle before lifting (see ArmSkills.release_and_retreat).
        backoff = self.point_world(pinch) - self.frame()[1][:, 0] * self.release_backoff
        yield from self.line(backoff, leave, steps=2)
        away = self.point_world(pinch) + UP * (BOTTLE["height"] + SIDE_PRE)
        yield from self.line(away, leave, steps=6)
        self.side = None

    # ------------------------------------------------------------ helpers
    def _timed(self, q, grip, seconds):
        start = self.cmd.copy()
        goal = start.copy()
        goal[:5] = q
        goal[5] = grip
        steps = max(1, int(round(seconds * CONTROL_HZ)))
        for i in range(1, steps + 1):
            s = i / steps
            self.cmd = start + (goal - start) * (s * s * (3 - 2 * s))
            yield self.cmd.copy()
