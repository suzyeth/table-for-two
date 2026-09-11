"""Scripted manipulation skills for one SO-101 arm.

Every skill is a generator that yields a 6-D joint target (5 arm joints plus
the gripper) once per control step. Cartesian goals are converted to joints
with IK at the moment the segment starts, so each segment reacts to where the
objects actually are (important under randomised initial placement).
"""
import mujoco
import numpy as np
from scipy.optimize import minimize

from sim.env import CONTROL_HZ, GRIPPER_CLOSED, GRIPPER_OPEN, TABLE_TOP_Z
from sim.ik import DOWN

MAX_JOINT_SPEED = 1.2  # rad/s for free-space moves
SLOW_JOINT_SPEED = 0.5  # rad/s near objects
MIN_SEGMENT_STEPS = 6
GRIP_STEPS = 8
SETTLE_STEPS = 3
CLEARANCE = 0.10  # hover height above grasp and place points (clears the open drawer front)

# TCP height above the object origin when grasping, and object half heights.
GRASP_Z = {"plate": 0.006, "mug": 0.03, "bottle": 0.03, "spoon": 0.004, "fork": 0.004}
HALF_HEIGHT = {"plate": 0.004, "mug": 0.035, "bottle": 0.06, "spoon": 0.003, "fork": 0.003}
PLACE_MARGIN = 0.004

DRAWER_PULL = 0.10
DRAWER_WAYPOINTS = 8

POUR_TILT = np.deg2rad(75)
POUR_HOLD_STEPS = int(1.2 * CONTROL_HZ)
POUR_ABOVE_MUG = 0.05  # bottle tip height above the mug rim; lower and the tilted bottle hits the rim
POUR_CORRECTIONS = 4  # closed-loop passes that fold the measured tip error back into the goal
POUR_CORRECTION_GAIN = 0.8
POUR_SETTLE_STEPS = 4
BOTTLE_TIP = 0.06  # bottle origin to tip along its axis
POUR_POS_SCALE = 0.01  # 1 cm tip error costs as much as...
POUR_TILT_SCALE = np.deg2rad(5)  # ...5 degrees of missing tilt
# (wrist_flex, wrist_roll) offsets used as optimiser seeds; the pour cost has local minima.
POUR_SEEDS = [(flex, roll) for flex in (0.0, -0.8, 0.8, -1.4, 1.4) for roll in (0.0, 1.5, -1.5)]


def smoothstep(s):
    return s * s * (3.0 - 2.0 * s)


class ArmSkills:
    def __init__(self, env, arm):
        self.env = env
        self.arm = arm
        self.cmd = env.home[arm].copy()
        self.last_pour_cost = None

    # -------------------------------------------------------- primitives
    def _interp(self, goal, speed=MAX_JOINT_SPEED):
        start = self.cmd.copy()
        dist = float(np.max(np.abs(goal[:5] - start[:5])))
        steps = max(MIN_SEGMENT_STEPS, int(np.ceil(dist / speed * CONTROL_HZ)))
        for i in range(1, steps + 1):
            self.cmd = start + (goal - start) * smoothstep(i / steps)
            yield self.cmd.copy()
        for _ in range(SETTLE_STEPS):
            yield self.cmd.copy()

    def move(self, pos, approach=DOWN, grip=None, speed=MAX_JOINT_SPEED, roll=None):
        """Move the TCP to ``pos`` (world frame) with the given approach axis."""
        q, _ = self.env.ik[self.arm].solve(self.env.data.qpos, np.asarray(pos, dtype=float), approach)
        goal = self.cmd.copy()
        goal[:5] = q
        if roll is not None:
            goal[4] = roll
        if grip is not None:
            goal[5] = grip
        yield from self._interp(goal, speed)

    def move_joints(self, q_arm, speed=MAX_JOINT_SPEED):
        goal = self.cmd.copy()
        goal[:5] = q_arm
        yield from self._interp(goal, speed)

    def grip(self, value, intent=None):
        """Ramp the gripper to ``value``; ``intent`` names the object a closing grip is meant for."""
        if intent is not None:
            self.env.set_grasp_intent(self.arm, intent)
        start = self.cmd.copy()
        for i in range(1, GRIP_STEPS + 1):
            self.cmd = start.copy()
            self.cmd[5] = start[5] + (value - start[5]) * i / GRIP_STEPS
            yield self.cmd.copy()

    def wait(self, steps):
        for _ in range(steps):
            yield self.cmd.copy()

    def home(self):
        yield from self._interp(self.env.home[self.arm].copy())

    # ------------------------------------------------------------ skills
    def pick(self, obj, offset=None):
        """Top-down grasp of ``obj``; ``offset`` (world frame) shifts the grasp point along the object."""
        shift = np.zeros(3) if offset is None else np.asarray(offset, dtype=float)
        above = self.env.grasp_point(obj) + shift + [0.0, 0.0, GRASP_Z[obj] + CLEARANCE]
        yield from self.move(above, grip=GRIPPER_OPEN)
        grasp = self.env.grasp_point(obj) + shift + [0.0, 0.0, GRASP_Z[obj]]
        yield from self.move(grasp, speed=SLOW_JOINT_SPEED)
        yield from self.grip(GRIPPER_CLOSED, intent=obj)
        yield from self.move(grasp + [0.0, 0.0, CLEARANCE], speed=SLOW_JOINT_SPEED)

    def place(self, obj, xy):
        """Lower the held object so its origin lands on ``xy``, then release."""
        offset = self.env.grasp_point(obj) - self.env.tcp(self.arm)  # object origin relative to TCP
        centre_z = TABLE_TOP_Z + HALF_HEIGHT[obj] + PLACE_MARGIN
        tcp_goal = np.array([xy[0] - offset[0], xy[1] - offset[1], centre_z - offset[2]])
        yield from self.move(tcp_goal + [0.0, 0.0, CLEARANCE])
        yield from self.move(tcp_goal, speed=SLOW_JOINT_SPEED)
        yield from self.grip(GRIPPER_OPEN)
        yield from self.move(tcp_goal + [0.0, 0.0, CLEARANCE], speed=SLOW_JOINT_SPEED)

    def pick_place(self, obj, xy):
        yield from self.pick(obj)
        yield from self.place(obj, xy)

    def open_drawer(self):
        """Grip the handle from the front (approach +x) and pull straight out."""
        into_cabinet = np.array([1.0, 0.0, 0.0])
        handle = self.env.grasp_point("drawer")
        yield from self.move(handle + [-CLEARANCE, 0.0, 0.0], approach=into_cabinet, grip=GRIPPER_OPEN)
        yield from self.move(handle + [-0.004, 0.0, 0.0], approach=into_cabinet, speed=SLOW_JOINT_SPEED)
        yield from self.grip(GRIPPER_CLOSED, intent="drawer")
        start = self.env.tcp(self.arm)
        for i in range(1, DRAWER_WAYPOINTS + 1):
            waypoint = start + [-DRAWER_PULL * i / DRAWER_WAYPOINTS, 0.0, 0.0]
            yield from self.move(waypoint, approach=into_cabinet, speed=SLOW_JOINT_SPEED)
        yield from self.grip(GRIPPER_OPEN)
        # Back straight out with the same wrist attitude, then rise; switching
        # attitude while still at the handle makes the wrist flip into the table.
        yield from self.move(self.env.tcp(self.arm) + [-0.04, 0.0, 0.0], approach=into_cabinet,
                             speed=SLOW_JOINT_SPEED)
        yield from self.move(self.env.tcp(self.arm) + [0.0, 0.0, CLEARANCE], approach=into_cabinet,
                             speed=SLOW_JOINT_SPEED)

    def lift_to(self, pos, roll=None):
        yield from self.move(pos, roll=roll)

    # -------------------------------------------------------------- pouring
    def _held_pose(self, obj, q_arm):
        """Forward kinematics of a welded object for arm joints ``q_arm``: (origin, rotation)."""
        env, m = self.env, self.env.model
        ik = env.ik[self.arm]
        d = ik.scratch
        d.qpos[:] = env.data.qpos
        d.qpos[ik.qpos_adr] = q_arm
        mujoco.mj_kinematics(m, d)
        eq = env.eq_ids[(self.arm, obj)]
        body = env.gripper_body[self.arm]
        origin = d.xpos[body] + d.xmat[body].reshape(3, 3) @ m.eq_data[eq, 3:6]
        quat, rot = np.zeros(4), np.zeros(9)
        mujoco.mju_mulQuat(quat, d.xquat[body], m.eq_data[eq, 6:10])
        mujoco.mju_quat2Mat(rot, quat)
        return origin, rot.reshape(3, 3)

    def _pour_joints(self, tip_goal):
        """Arm joints that put the held bottle's tip at ``tip_goal`` tilted by at least POUR_TILT."""
        ik = self.env.ik[self.arm]
        bounds = list(zip(ik.lo, ik.hi))

        def cost(q):
            origin, rot = self._held_pose("bottle", q)
            tip = origin + rot @ np.array([0.0, 0.0, BOTTLE_TIP])
            tilt = np.arccos(np.clip(rot[2, 2], -1.0, 1.0))
            pos_term = np.sum((tip - tip_goal) ** 2) / POUR_POS_SCALE ** 2
            tilt_term = (max(0.0, POUR_TILT - tilt) / POUR_TILT_SCALE) ** 2
            return pos_term + tilt_term

        start = self.env.arm_qpos(self.arm)[:5]
        best = None
        for flex_offset, roll_offset in POUR_SEEDS:
            seed = start.copy()
            seed[3] += flex_offset
            seed[4] += roll_offset
            seed = np.clip(seed, ik.lo, ik.hi)
            result = minimize(cost, seed, method="L-BFGS-B", bounds=bounds)
            if best is None or result.fun < best.fun:
                best = result
            if best.fun < 0.05:  # tip within ~2 mm and fully tilted: good enough
                break
        self.last_pour_cost = float(best.fun)
        return best.x

    def _bottle_tip(self):
        """Current world position of the bottle tip (actual simulation state)."""
        body = self.env.body_ids["bottle"]
        rot = self.env.data.xmat[body].reshape(3, 3)
        return self.env.data.xpos[body] + rot @ np.array([0.0, 0.0, BOTTLE_TIP])

    def pour_into(self, target_obj="mug"):
        """Bring the held bottle beside ``target_obj``, tilt its tip over the rim, hold, straighten.

        The arm sags under the bottle's weight and the held mug drifts, so after
        the open-loop move the tip error is measured and integrated into the goal
        for a few correction passes before holding the pour.
        """
        mug = self.env.grasp_point(target_obj)
        side = np.sign(mug[1] - self.env.tcp(self.arm)[1]) or 1.0
        rim_offset = np.array([0.0, 0.0, HALF_HEIGHT[target_obj] + POUR_ABOVE_MUG])
        tip_goal = mug + rim_offset
        yield from self.move(tip_goal + [0.0, -side * 0.07, 0.06])
        upright = self.cmd[:5].copy()
        yield from self.move_joints(self._pour_joints(tip_goal), speed=SLOW_JOINT_SPEED)
        correction = np.zeros(3)
        for _ in range(POUR_CORRECTIONS):
            tip_goal = self.env.grasp_point(target_obj) + rim_offset
            correction += POUR_CORRECTION_GAIN * (tip_goal - self._bottle_tip())
            yield from self.move_joints(self._pour_joints(tip_goal + correction), speed=SLOW_JOINT_SPEED)
            yield from self.wait(POUR_SETTLE_STEPS)
        yield from self.wait(POUR_HOLD_STEPS)
        yield from self.move_joints(upright, speed=SLOW_JOINT_SPEED)

    def release_and_retreat(self):
        yield from self.grip(GRIPPER_OPEN)
        yield from self.move(self.env.tcp(self.arm) + [0.0, 0.0, 0.05], speed=SLOW_JOINT_SPEED)
