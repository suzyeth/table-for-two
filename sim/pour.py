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
from typing import NamedTuple

import mujoco
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
# Before the squeeze the hand slides along the closing axis until the fixed pad is this far from the
# bottle: otherwise the moving jaw pushed the light, tall bottle ~6 mm before the fixed pad caught it
# and tipped it 4.6-8.5 deg (6 of 20 seeds, higher grasps worst); with 1-2 mm left, none tipped.
SIDE_GRASP_GAP = 0.001
SIDE_ROT_TOL = np.deg2rad(3)

TILTS_DEG = (0, 20, 40, 60, 70, 80, 88, 95, 100, 105, 110, 120, 130, 140)
CLEARANCES = np.arange(0.010, 0.080, 0.005)  # whole bottle above the mug rim
POUR_START_DEG = 70  # from here on water may leave, so the lip stays low over the mug
POUR_MAX_CLEARANCE = 0.03
POUR_HOLD_STEPS = 20
MAX_POUR_JOINT_STEP = 0.3  # rad: no joint moves more than this between two pour waypoints
MIN_TILT_STEP_DEG = 1.0  # a tilt step is split no finer than this before the plan gives up there
MIN_MUG_CLEARANCE = 0.005  # jaws, gripper housing and the held bottle stay this far off the mug
UNWIND_LIFT = 0.035  # lift the tilted bottle this far before straightening it up again
POUR_JOINT_SPEED = 1.0  # rad/s: every pour move is slow enough that no joint turns faster than this on average
# With the side grip the arm's IK solutions for "lip over the mug at this tilt" come in two branches
# (seed 0: 0-90 deg and 105-140 deg; none in between at any lip height up to 12 cm), so a full pour
# must switch once. The switch goes between lifted poses near the end of one branch and the start of
# the other, and every pose on the straight joint-space move between them keeps this far off the mug.
SWITCH_MUG_CLEARANCE = 0.010
SWITCH_WINDOW_DEG = 10  # switch from within this many degrees of either branch's end
SWITCH_CHECK_SAMPLES = 21
TILT_SECONDS = 0.5  # per waypoint before the water can leave ...
FLOW_SECONDS = 0.8  # ... and from POUR_START_DEG on


class PourStep(NamedTuple):
    """One pour waypoint: bottle tilt (deg), clearance above the rim (m), arm joints, and the
    same tilt lifted clear of the mug for the way back. ``lifted`` marks the extra-high poses
    around a branch switch, where the pour does not pause."""
    tilt: float
    clearance: float
    q: np.ndarray
    q_raised: np.ndarray
    lifted: bool = False


def _clearance_order(tilt):
    """Clearances to try for ``tilt``: within the tight limit first (lowest first), then more."""
    limit = POUR_MAX_CLEARANCE if tilt >= POUR_START_DEG else CLEARANCES[-1]
    return list(CLEARANCES[CLEARANCES <= limit + 1e-9]) + list(CLEARANCES[CLEARANCES > limit + 1e-9])


def _step_seconds(tilt):
    return TILT_SECONDS if tilt < POUR_START_DEG else FLOW_SECONDS


def _move_seconds(q_from, q_to, tilt):
    """Duration of a pour move: the usual pace for ``tilt``, longer if a joint would exceed POUR_JOINT_SPEED."""
    return max(_step_seconds(tilt), float(np.max(np.abs(np.asarray(q_to) - np.asarray(q_from)))) / POUR_JOINT_SPEED)


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
        yield from self._close_fixed_pad(plan["orient"])
        yield from self.grip(GRIPPER_SQUEEZE)
        yield from self.wait(self.squeeze_settle)
        yield from self.line(plan["centre"] + UP * BOTTLE_LIFT, plan["orient"], steps=5)

    def _close_fixed_pad(self, orient, obj="bottle"):
        """Slide the open hand along its closing axis until the fixed pad is SIDE_GRASP_GAP from ``obj``,
        so the moving jaw does not shove the object across the gap (and tip it) before the pad catches it."""
        shift = self._fixed_pad_gap(obj) - SIDE_GRASP_GAP
        if shift <= 0:
            return
        closing = self.frame()[1][:, 0]
        yield from self.line(self.point_world(orient.get("point")) + closing * shift, orient, steps=4)

    def _fixed_pad_gap(self, obj):
        """Actual distance (m) from this hand's fixed jaw to ``obj``, read up to 3 cm."""
        cache = self.__dict__.setdefault("_pad_gap_geoms_cache", {})
        if obj not in cache:
            model = self.env.model
            collidable = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
            cache[obj] = ([g for g in collidable if model.geom_bodyid[g] == self.env.fixed_jaw[self.arm]],
                          [g for g in collidable if model.geom_bodyid[g] == self.env.body_ids[obj]])
        jaw, target = cache[obj]
        model, data = self.env.model, self.env.data
        return min(mujoco.mj_geomDistance(model, data, a, b, 0.03, None) for a in jaw for b in target)

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
        """Pour waypoints from upright to the last reachable tilt, as a list of ``PourStep``.

        After the first (upright) pose no joint moves more than MAX_POUR_JOINT_STEP between two
        waypoints: a tilt the arm cannot reach in one such step is split into smaller tilts. A
        tilt unreachable at the tight clearance above the rim is tried with more clearance rather
        than skipped, and a pose where the hand or the bottle would come within MIN_MUG_CLEARANCE
        of the mug is not used. Where no pose is reachable the plan stops there and says so,
        instead of jumping across the gap (which once whipped the wrist 2.2 rad into the mug).
        """
        lip, outline, axis_g = self._bottle_in_gripper()
        lean = np.cross(UP, horizontal(rim - self.env.arm_base(self.arm)))  # tilt across the arm's reach

        def pose(q_seed, tilt_deg, clearance, restarts=False):
            q, err = self._pour_pose(q_seed, np.deg2rad(tilt_deg), lean, rim, lip, outline, axis_g, clearance,
                                     restarts)
            return q if err < SIDE_POS_TOL and self.ik.last_rot_err < SIDE_ROT_TOL else None

        first = self._first_pour_step(pose)
        if first is None:
            self.warnings.append(f"{self.arm} pour plan: the upright pose over the mug is out of reach")
            return []
        plan = [first]
        for tilt in TILTS_DEG[1:]:
            if not self._extend_pour(plan, float(tilt), pose):
                break
        if plan[-1].tilt < TILTS_DEG[-1]:
            plan = self._cross_to_other_branch(plan, pose)
        if plan[-1].tilt < TILTS_DEG[-1]:
            self.warnings.append(f"{self.arm} pour plan stops at {plan[-1].tilt:.0f} deg: no pose further within "
                                 f"{MAX_POUR_JOINT_STEP} rad per step, and no switch clear of the mug")
        return plan

    def _first_pour_step(self, pose):
        tilt = float(TILTS_DEG[0])
        for clearance in _clearance_order(tilt):
            q = pose(self.cmd[:5].copy(), tilt, clearance, restarts=True)
            if q is not None and self._pour_pose_clear(q):
                # The lift itself may be a big move (it is timed by distance); the lifted poses after it
                # are each checked against the one before.
                return PourStep(tilt, float(clearance), q, self._raised(pose, q, tilt, clearance, q, capped=False))
        return None

    def _extend_pour(self, plan, tilt, pose):
        """Append waypoints up to ``tilt``, splitting the step while it moves a joint too far."""
        step = self._pour_step(plan[-1], tilt, pose)
        if step is not None:
            plan.append(step)
            return True
        if abs(tilt - plan[-1].tilt) <= MIN_TILT_STEP_DEG:
            return False
        return self._extend_pour(plan, (plan[-1].tilt + tilt) / 2, pose) and self._extend_pour(plan, tilt, pose)

    def _pour_step(self, previous, tilt, pose):
        for clearance in _clearance_order(tilt):
            q = pose(previous.q, tilt, clearance)
            if q is None or np.max(np.abs(q - previous.q)) > MAX_POUR_JOINT_STEP or not self._pour_pose_clear(q):
                continue
            return PourStep(tilt, float(clearance), q, self._raised(pose, previous.q_raised, tilt, clearance, q))
        return None

    def _raised(self, pose, q_seed, tilt, clearance, q, capped=True):
        """The same tilt UNWIND_LIFT higher, for the way back; ``q`` itself if that is not reachable.

        ``q_seed`` is the previous waypoint's lifted pose; with ``capped`` the lifted pose must be
        within MAX_POUR_JOINT_STEP of it, so the way back never jumps.
        """
        q_up = pose(q_seed, tilt, clearance + UNWIND_LIFT)
        if q_up is None or not self._pour_pose_clear(q_up):
            return q
        if capped and np.max(np.abs(q_up - q_seed)) > MAX_POUR_JOINT_STEP:
            return q
        return q_up

    def _cross_to_other_branch(self, low, pose):
        """``low`` continued past a gap in the IK solutions: one switch, then the rest on the other branch.

        The other branch is followed down from the last tilt. The switch goes from a lifted pose
        within SWITCH_WINDOW_DEG of the end of ``low`` to a lifted pose within SWITCH_WINDOW_DEG of
        the start of the other branch; of the switches whose straight joint-space move keeps
        SWITCH_MUG_CLEARANCE off the mug all the way, the one moving the joints least (then the
        least lifted) is used. ``low`` is returned unchanged if there is none.
        """
        high = self._other_branch(low[-1].tilt, pose)
        if not high:
            return low
        a_chains = [(i, self._lift_chain(low[i], pose)) for i in range(len(low))
                    if low[i].tilt >= low[-1].tilt - SWITCH_WINDOW_DEG]
        b_chains = [(j, self._lift_chain(high[j], pose)) for j in range(len(high))
                    if high[j].tilt <= high[0].tilt + SWITCH_WINDOW_DEG]
        pairs = [(float(np.max(np.abs(b.q - a.q))), a.clearance + b.clearance, i, ka, j, kb)
                 for i, a_chain in a_chains for ka, a in enumerate(a_chain)
                 for j, b_chain in b_chains for kb, b in enumerate(b_chain)]
        chains = {("a", i): chain for i, chain in a_chains} | {("b", j): chain for j, chain in b_chains}
        for _, _, i, ka, j, kb in sorted(pairs, key=lambda p: (p[0], p[1])):
            a_chain, b_chain = chains[("a", i)], chains[("b", j)]
            if self._path_clear(a_chain[ka].q, b_chain[kb].q):
                return low[:i + 1] + a_chain[1:ka + 1] + b_chain[kb:0:-1] + high[j:]
        return low

    def _other_branch(self, below_tilt, pose):
        """The pour's other IK branch, from its lowest tilt above ``below_tilt`` up to the last tilt."""
        top = float(TILTS_DEG[-1])
        first = None
        for clearance in _clearance_order(top):
            q = pose(self.cmd[:5].copy(), top, clearance, restarts=True)
            if q is not None and self._pour_pose_clear(q):
                first = PourStep(top, float(clearance), q, self._raised(pose, q, top, clearance, q, capped=False))
                break
        if first is None:
            return []
        high = [first]
        for tilt in reversed(TILTS_DEG[:-1]):
            if tilt <= below_tilt or not self._extend_pour(high, float(tilt), pose):
                break
        return high[::-1]

    def _lift_chain(self, step, pose):
        """``step`` and the same tilt ever higher above the rim, each within MAX_POUR_JOINT_STEP of the last."""
        chain = [step]
        for clearance in CLEARANCES[CLEARANCES > step.clearance + 1e-9]:
            q = pose(chain[-1].q, step.tilt, clearance)
            if q is None or np.max(np.abs(q - chain[-1].q)) > MAX_POUR_JOINT_STEP or not self._pour_pose_clear(q):
                break
            chain.append(PourStep(step.tilt, float(clearance), q, q, lifted=True))
        return chain

    def _path_clear(self, q_a, q_b, margin=SWITCH_MUG_CLEARANCE):
        """True if every pose on the straight joint-space move from ``q_a`` to ``q_b`` keeps ``margin`` off the mug."""
        return all(self._pour_pose_clear(q_a + (q_b - q_a) * s, margin)
                   for s in np.linspace(0.0, 1.0, SWITCH_CHECK_SAMPLES))

    def pour_into(self, target="mug"):
        """Tilt the held bottle with its lip kept over ``target``, hold, lift, and straighten up again.

        The way back runs through the lifted poses, each segment taking as long as it took on the
        way in, so the bottle and hand never swing back low over the mug.
        """
        rim = self.env.object_frame(target)[0] + UP * MUG["height"]
        plan = self._plan_pour(rim)
        if not plan or plan[-1].tilt < 100:
            self.warnings.append(f"{self.arm} pour plan reaches only {plan[-1].tilt if plan else 0:.0f} deg")
        if not plan:
            return
        previous = self.cmd[:5].copy()
        for step in plan:
            yield from self._timed(step.q, GRIPPER_SQUEEZE, _move_seconds(previous, step.q, step.tilt))
            previous = step.q
            if step.tilt >= 95 and not step.lifted:
                yield from self.wait(8)
        yield from self.wait(POUR_HOLD_STEPS)
        back = self._unwind_poses(plan)
        yield from self._timed(back[-1], GRIPPER_SQUEEZE, _move_seconds(plan[-1].q, back[-1], plan[-1].tilt))
        for k in range(len(plan) - 2, -1, -1):
            yield from self._timed(back[k], GRIPPER_SQUEEZE, _move_seconds(back[k + 1], back[k], plan[k + 1].tilt))

    def _unwind_poses(self, plan):
        """Arm joints for the way back, one per waypoint: its lifted pose where the moves stay clear, else the pour pose.

        The branch switch is retraced exactly between the two pour poses whose path was checked.
        The lift and every other move back must keep MIN_MUG_CLEARANCE along the whole straight
        joint-space path; where one does not, both its ends fall back to the pour poses, which the
        way in already went through. (Speed is capped by the timing, so a lift may be a big move.)
        """
        lowered = [bool(np.array_equal(step.q, step.q_raised)) for step in plan]
        for k in range(len(plan) - 1):
            if np.max(np.abs(plan[k + 1].q - plan[k].q)) > MAX_POUR_JOINT_STEP + 1e-9:
                lowered[k] = lowered[k + 1] = True

        def pose(i):
            return plan[i].q if lowered[i] else plan[i].q_raised

        changed = True
        while changed:
            changed = False
            if not lowered[-1] and not self._path_clear(plan[-1].q, plan[-1].q_raised, MIN_MUG_CLEARANCE):
                lowered[-1] = changed = True
            for k in range(len(plan) - 2, -1, -1):
                if (lowered[k] and lowered[k + 1]) or self._path_clear(pose(k + 1), pose(k), MIN_MUG_CLEARANCE):
                    continue
                for i in (k, k + 1):
                    if not lowered[i]:
                        lowered[i] = changed = True
        return [pose(i) for i in range(len(plan))]

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
    def _pour_pose_clear(self, q, margin=MIN_MUG_CLEARANCE):
        """True if at arm joints ``q`` the jaws, gripper housing and held bottle stay ``margin`` off the mug.

        The bottle is moved with the hand, keeping where it sits in the fingers right now.
        """
        model, scratch, data = self.env.model, self.ik.scratch, self.env.data
        body = self.ik.gripper_body
        hand_pos, hand_rot = data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()
        bottle_pos, bottle_rot = self.env.object_frame("bottle")
        new_pos, new_rot = self.frame(q)  # leaves ``scratch`` at joints q
        adr = model.jnt_qposadr[model.body_jntadr[self.env.body_ids["bottle"]]]
        scratch.qpos[adr:adr + 3] = new_pos + new_rot @ (hand_rot.T @ (bottle_pos - hand_pos))
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (new_rot @ hand_rot.T @ bottle_rot).ravel())
        scratch.qpos[adr + 3:adr + 7] = quat
        mujoco.mj_kinematics(model, scratch)
        hand_and_bottle, mug = self._mug_check_geoms()
        return all(mujoco.mj_geomDistance(model, scratch, a, b, margin, None) >= margin
                   for a in hand_and_bottle for b in mug)

    def _mug_check_geoms(self):
        """(collidable geoms of this hand's jaws/housing and of the bottle, collidable geoms of the mug)."""
        cached = getattr(self, "_mug_geoms_cache", None)
        if cached is None:
            model = self.env.model
            hand = set(self.env.finger_bodies[self.arm]) | {self.env.body_ids["bottle"]}
            mug = self.env.body_ids["mug"]
            collidable = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
            cached = ([g for g in collidable if model.geom_bodyid[g] in hand],
                      [g for g in collidable if model.geom_bodyid[g] == mug])
            self._mug_geoms_cache = cached
        return cached

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
