"""Contact-only scripted skills for one SO-101 arm.

Every skill is a generator that yields a 6-D joint target (5 arm joints plus
the gripper) once per control step. Nothing is attached to the gripper: a skill
places the open jaws around an object, closes past contact so the servo keeps
squeezing, and from then on the object moves only because friction holds it.
Cartesian goals are converted to joints with IK when each segment starts, from
the actual object poses, so skills react to randomised placement and to any
slip in the hand (``place`` measures where the object really sits in the hand).

Moves between places go up to a transit height first, because the top-down
reach of the SO-101 ends ~8 cm above the table, then across, then down.
"""
import mujoco
import numpy as np

from scene.build_scene import TABLE_TOP_Z, UTENSIL_HANDLE
from sim.env import CONTROL_HZ, GRIPPER_OPEN, GRIPPER_SQUEEZE
from sim.grasping import DOWN, PRE_HEIGHT, SITE_LOCAL, UP, horizontal, top_down_grasp
from sim.ik import gripper_rotation
from sim.pour import PourMixin, _rotation_between

MAX_JOINT_SPEED = 1.2  # rad/s for free-space moves
SLOW_JOINT_SPEED = 0.4  # rad/s near and in contact with objects
MIN_SEGMENT_STEPS = 6
LINE_MIN_STEPS = 2
GRIP_STEPS = 12
SQUEEZE_SETTLE_STEPS = 8
SETTLE_STEPS = 3
# Top-down reach tops out at 9 cm above the table 18-24 cm from the base and 4-6 cm
# at 12 or 28 cm, so hands travel 6 cm up (above the plate, utensils and drawer).
TRANSIT_TCP_Z = TABLE_TOP_Z + 0.06
LIFT = 0.04
SLIDE_HEIGHT = 0.004  # a sideways approach slides in this far above the grasp, then settles
PLACE_ABOVE = 0.03
TOUCH_STEP = 0.0015  # set-down increment; with TOUCH_MIN_STEPS control steps each, ~1.5 cm/s
TOUCH_MIN_STEPS = 2
TOUCH_OVERSHOOT = 0.006  # keep lowering this far past the nominal rest height before giving up
RETREAT = 0.03
RELEASE_BACKOFF = 0.003  # after opening, move the fixed finger this far off the object before lifting
DRAWER_PULL = 0.100
IK_POS_TOL = 0.004  # warn beyond this
IK_ROT_TOL = np.deg2rad(4)
IK_RETRY_POS = 0.0015  # retry with restarts beyond this
IK_RETRY_ROT = np.deg2rad(2)
# Height of each object's origin above the table when it rests there.
REST_HEIGHT = {"plate": 0.0, "mug": 0.0, "bottle": 0.0, "spoon": UTENSIL_HANDLE[2], "fork": UTENSIL_HANDLE[2]}


def smoothstep(s):
    return s * s * (3.0 - 2.0 * s)


class ArmSkills(PourMixin):
    free_speed = MAX_JOINT_SPEED
    slow_speed = SLOW_JOINT_SPEED
    squeeze_settle = SQUEEZE_SETTLE_STEPS
    release_backoff = RELEASE_BACKOFF

    def __init__(self, env, arm):
        self.env = env
        self.arm = arm
        self.ik = env.ik[arm]
        self.cmd = env.home[arm].copy()
        self.orient = {"approach": None}  # orientation of the last Cartesian segment
        self.release_to = GRIPPER_OPEN  # jaw opening used when letting go of the current object
        self.warnings = []

    # ------------------------------------------------------- kinematics
    def _qpos(self, q=None):
        qpos = self.env.data.qpos.copy()
        qpos[self.ik.qpos_adr] = self.cmd[:5] if q is None else q
        return qpos

    def frame(self, q=None):
        """Gripper body (position, rotation) for arm joints ``q`` (default: the commanded ones)."""
        scratch = self.ik.scratch
        scratch.qpos[:] = self._qpos(q)
        mujoco.mj_kinematics(self.env.model, scratch)
        body = self.ik.gripper_body
        return scratch.xpos[body].copy(), scratch.xmat[body].reshape(3, 3).copy()

    def point_world(self, point, q=None):
        """World position of a gripper-frame point (None = the TCP site) for joints ``q``."""
        pos, rot = self.frame(q)
        return pos + rot @ (SITE_LOCAL if point is None else point)

    def held_point(self, obj):
        """Origin of ``obj`` in gripper coordinates, from the actual simulation state."""
        d = self.env.data
        body = self.ik.gripper_body
        return d.xmat[body].reshape(3, 3).T @ (self.env.object_frame(obj)[0] - d.xpos[body])

    def solve(self, target, orient, restarts=False):
        """IK from the commanded joints; retry with restarts if that lands off by more than a mm or two.

        (A local solution a few mm off is not harmless: a grasp that stops 4 mm high catches
        only the top edge of a handle.)
        """
        q, err = self.ik.solve(self._qpos(), target, restarts=restarts, **orient)
        rot_err = self.ik.last_rot_err
        if not restarts and (err > IK_RETRY_POS or rot_err > IK_RETRY_ROT):
            q_alt, err_alt = self.ik.solve(self._qpos(), target, restarts=True, **orient)
            if err_alt + 0.3 * self.ik.last_rot_err < err + 0.3 * rot_err:
                q, err, rot_err = q_alt, err_alt, self.ik.last_rot_err
        if err > IK_POS_TOL or rot_err > IK_ROT_TOL:
            self.warnings.append(f"{self.arm} IK {err * 1000:.1f} mm / {np.degrees(rot_err):.1f} deg "
                                 f"at {np.round(target, 3).tolist()}")
        return q, err

    # ------------------------------------------------------- primitives
    def _interp(self, goal, speed=MAX_JOINT_SPEED, min_steps=MIN_SEGMENT_STEPS, settle=SETTLE_STEPS):
        start = self.cmd.copy()
        dist = float(np.max(np.abs(goal[:5] - start[:5])))
        steps = max(min_steps, int(np.ceil(dist / speed * CONTROL_HZ)))
        for i in range(1, steps + 1):
            self.cmd = start + (goal - start) * smoothstep(i / steps)
            yield self.cmd.copy()
        for _ in range(settle):
            yield self.cmd.copy()

    def _to(self, q, grip=None, speed=MAX_JOINT_SPEED, min_steps=MIN_SEGMENT_STEPS, settle=SETTLE_STEPS):
        goal = self.cmd.copy()
        goal[:5] = q
        if grip is not None:
            goal[5] = grip
        yield from self._interp(goal, speed, min_steps, settle)

    def line(self, end, orient, steps=6, speed=SLOW_JOINT_SPEED, grip=None):
        """Straight-line Cartesian move of the controlled point (``orient['point']``) to ``end``."""
        start = self.point_world(orient.get("point"))
        end = np.asarray(end, dtype=float)
        turn = self._closing_turn(orient)
        self.orient = orient
        for k in range(1, steps + 1):
            step_orient = orient if turn is None else {**orient, "closing": turn(k / steps)}
            q, _ = self.solve(start + (end - start) * k / steps, step_orient)
            yield from self._to(q, grip, speed, LINE_MIN_STEPS, settle=0)
        for _ in range(SETTLE_STEPS):
            yield self.cmd.copy()

    def _closing_turn(self, orient):
        """For a top-down hand, s -> jaw direction turning the current jaws onto ``orient``'s by fraction s."""
        if orient.get("approach") is None or orient.get("closing") is None:
            return None
        current_x = self.frame()[1][:, 0]
        if abs(current_x[2]) > 0.2:  # hand not top-down yet: nothing sensible to interpolate
            return None
        current, target = horizontal(current_x), horizontal(orient["closing"])
        angle = np.arctan2(np.cross(current, target)[2], current @ target)
        if abs(angle) < 1e-3:
            return None

        def at(fraction):
            c, s = np.cos(angle * fraction), np.sin(angle * fraction)
            return np.array([c * current[0] - s * current[1], s * current[0] + c * current[1], 0.0])
        return at

    def grip(self, value, steps=GRIP_STEPS):
        start = self.cmd.copy()
        for i in range(1, steps + 1):
            self.cmd = start.copy()
            self.cmd[5] = start[5] + (value - start[5]) * i / steps
            yield self.cmd.copy()

    def wait(self, steps):
        for _ in range(steps):
            yield self.cmd.copy()

    def rise(self):
        """Lift straight up to the transit height, keeping the current hand orientation."""
        tcp_z = self.point_world(None)[2]
        if tcp_z < TRANSIT_TCP_Z - 0.005:
            here = self.point_world(self.orient.get("point"))
            yield from self.line(here + UP * (TRANSIT_TCP_Z - tcp_z), self.orient, steps=4, speed=MAX_JOINT_SPEED)

    def _tcp_above_point(self, orient):
        """TCP height minus controlled-point height for a top-down hand with ``orient``."""
        point = orient.get("point")
        if point is None:
            return 0.0
        rot = gripper_rotation(DOWN, orient["closing"])
        return float((rot @ (SITE_LOCAL - point))[2])

    def transit(self, target, orient, grip=None):
        """Move the controlled point to ``target``: up, across at transit height, down.

        From high up (e.g. home, where the hand is not top-down and top-down poses are out
        of reach) the move to the hover pose is a joint-space move; otherwise it is a
        straight line at transit height.
        """
        yield from self.rise()
        target = np.asarray(target, dtype=float)
        hover = target.copy()
        hover[2] = max(target[2], TRANSIT_TCP_Z - self._tcp_above_point(orient))
        if self.point_world(None)[2] > TRANSIT_TCP_Z + 0.02:
            q, _ = self.solve(hover, orient, restarts=True)
            self.orient = orient
            yield from self._to(q, grip, MAX_JOINT_SPEED)
        else:
            yield from self.line(hover, orient, steps=8, speed=MAX_JOINT_SPEED, grip=grip)
        if hover[2] > target[2] + 1e-4:
            yield from self.line(target, orient, steps=4, speed=MAX_JOINT_SPEED)

    def home(self):
        yield from self.rise()
        yield from self._interp(self.env.home[self.arm].copy())

    # ----------------------------------------------------------- grasps
    def _choose_orientation(self, grasp, camera_side=None):
        """Top-down hand orientation for ``grasp``; symmetric grasps may close either way round.

        ``camera_side`` (world direction) picks, for a symmetric grasp, the way round that puts
        the hand's jaw-width axis - the side the wrist-camera mount sticks out 4-8 cm - that way.
        """
        options = [grasp.closing] + ([-grasp.closing] if grasp.symmetric else [])
        if camera_side is not None and len(options) > 1:
            options = [c for c in options if np.cross(UP, c) @ np.asarray(camera_side) > 0] or options
        best = None
        for closing in options:
            orient = {"approach": DOWN, "closing": closing, "point": grasp.pinch}
            q, err = self.ik.solve(self._qpos(), grasp.centre, restarts=True, **orient)
            cost = err + 0.3 * self.ik.last_rot_err + 0.01 * float(np.abs(q - self.cmd[:5]).sum())
            if best is None or cost < best[0]:
                best = (cost, orient)
        return best[1]

    def pick(self, obj, along=0.0, toward=None, lift=LIFT, approach_from=None, camera_side=None, grasp_fn=None):
        """Top-down contact grasp: descend around ``obj``, squeeze past contact, lift ``lift``.

        ``approach_from`` (world offset) makes the hand line up that far beside the grasp first
        and slide in over it, e.g. to stay clear of the other hand during a hand-over;
        ``camera_side`` chooses which way a symmetric grasp faces (see _choose_orientation);
        ``grasp_fn`` supplies the grasp instead of ``top_down_grasp`` (called again to re-measure).
        """
        if grasp_fn is None:
            grasp_fn = lambda: top_down_grasp(self.env, self.arm, obj, along, toward)
        grasp = grasp_fn()
        orient = self._choose_orientation(grasp, camera_side)
        self.release_to = grasp.open_to
        above = grasp.centre + UP * PRE_HEIGHT
        if approach_from is not None:
            # Line up beside the grasp just above its height and slide the open jaws in along
            # the object (top-down reach ends ~9 cm up, so there is no room to come from above).
            level = grasp.centre + UP * SLIDE_HEIGHT
            yield from self.transit(level + np.asarray(approach_from, dtype=float), orient, grip=grasp.open_to)
            yield from self.line(level, orient, steps=6)
        else:
            yield from self.transit(above, orient, grip=grasp.open_to)
        # Re-measure just before closing in: the object may have moved (e.g. sagging in the
        # other hand during a hand-over).
        centre = grasp_fn().centre
        yield from self.line(centre, orient, steps=6)
        yield from self.grip(GRIPPER_SQUEEZE)
        yield from self.wait(SQUEEZE_SETTLE_STEPS)
        if lift > 0:
            yield from self.line(centre + UP * lift, orient, steps=4)

    def carry(self, obj, goal, closing=None, level=False):
        """Move the held object's origin to ``goal`` with the hand top-down (optionally re-oriented).

        ``level=True`` tips the wrist so the object is carried level: an object held off its
        centre of mass (a plate pinched by its wall) hangs tilted in the fingers, and would
        otherwise touch down on one edge. The tilt is measured from the actual state.
        """
        if closing is None:
            closing = horizontal(self.frame()[1][:, 0])
        closing = np.asarray(closing, dtype=float)
        orient = {"approach": DOWN, "closing": closing, "point": self.held_point(obj)}
        if level:
            orient["approach"], orient["closing"] = self._levelling(obj, closing)
        yield from self.transit(goal, orient)

    def _levelling(self, obj, closing):
        """(approach, closing) for the hand that holds ``obj`` upright, jaws heading along ``closing``."""
        body = self.ik.gripper_body
        up_in_hand = self.env.data.xmat[body].reshape(3, 3).T @ self.env.object_frame(obj)[1][:, 2]
        top_down = gripper_rotation(DOWN, closing)
        correction = _rotation_between(top_down @ up_in_hand, UP)
        hand = correction @ top_down
        return -hand[:, 2], hand[:, 0]

    def place(self, obj, xy, closing=None, level=False):
        """Set the held object down with its origin on ``xy``, open, and back off upward."""
        rest = np.array([xy[0], xy[1], TABLE_TOP_Z + REST_HEIGHT[obj]])
        yield from self.carry(obj, rest + UP * PLACE_ABOVE, closing, level)
        yield from self.lower_until_supported(obj, rest)
        yield from self.release_and_retreat()

    def lower_until_supported(self, obj, rest, orient=None):
        """Lower the held object slowly until it rests on something, like setting a cup down.

        Moves the controlled point toward ``rest`` in TOUCH_STEP increments at ~1.5 cm/s and
        stops at the first contact between the object and anything but the fingers, so the
        object is never let go in mid-air however it sits in the hand.
        """
        orient = orient or self.orient
        self.orient = orient
        here = self.point_world(orient.get("point"))
        floor = np.asarray(rest, dtype=float) - UP * TOUCH_OVERSHOOT
        distance = float(np.linalg.norm(floor - here))
        steps = max(1, int(np.ceil(distance / TOUCH_STEP)))
        for k in range(1, steps + 1):
            if self.env.supported(obj):
                break
            q, _ = self.solve(here + (floor - here) * k / steps, orient)
            yield from self._to(q, None, SLOW_JOINT_SPEED, TOUCH_MIN_STEPS, settle=0)
        if not self.env.supported(obj):
            self.warnings.append(f"{self.arm} set {obj} down without it touching a support")
        yield from self.wait(SETTLE_STEPS)

    def release_and_retreat(self):
        """Open, slide the fixed finger off the object, then lift away.

        Opens only as far as on the way in (a wide-open jaw can shove what it just let go).
        Lifting straight after opening is not enough: the fixed finger can still press on the
        object - two hands holding a plate from inside its walls spread it like chopsticks and
        lifted it 11 cm before dropping it - so the hand first moves back along the closing
        axis, away from the object.
        """
        yield from self.grip(self.release_to)
        yield from self.wait(2 * SETTLE_STEPS)
        closing = self.frame()[1][:, 0]
        here = self.point_world(self.orient.get("point"))
        yield from self.line(here - closing * RELEASE_BACKOFF, self.orient, steps=2)
        here = self.point_world(self.orient.get("point"))
        yield from self.line(here + UP * RETREAT, self.orient, steps=3)

    def open_drawer(self):
        """Pinch the bar handle top-down and pull the drawer straight out."""
        yield from self.pick("drawer", lift=0)
        orient = self.orient
        here = self.point_world(orient["point"])
        yield from self.line(here + np.array([-DRAWER_PULL, 0.0, 0.0]), orient, steps=10)
        yield from self.release_and_retreat()
