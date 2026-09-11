"""Damped-least-squares IK for one SO-101 arm with full gripper orientation.

The arm has 5 joints before the gripper. A target is the ``gripperframe`` site
position plus up to two gripper axes:
  * the approach axis (the direction the fingers point, gripper -z), and
  * optionally the jaw closing axis (fixed jaw -> moving jaw, gripper +x).
When the approach axis lies in the arm's vertical plane both can be met
exactly (pan + three pitch joints + wrist roll), which is what real grasps
need: fingers pointing into the object and jaws squeezing across it.

DLS with an orientation target has local minima, so ``solve`` restarts from
the current configuration, a few canonical arm postures and seeded random
postures, and keeps the solution with the lowest combined position +
orientation error.
"""
import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
DOWN = np.array([0.0, 0.0, -1.0])

POS_TOL = 0.001
ROT_TOL = 0.01  # rad
MAX_ITERS = 300
DAMPING = 1e-5
STEP = 0.5
ROT_WEIGHT = 0.3  # metres of position error traded per radian of orientation error
GOOD_ENOUGH = 0.002  # stop restarting once cost is below this
RANDOM_SEEDS = 10
# Canonical postures (pan, lift, elbow, wrist_flex, roll); pan is replaced by the target azimuth.
CANONICAL = (
    (0.0, -1.0, 1.0, 0.8, 0.0),
    (0.0, -0.5, 0.5, 1.4, 0.0),
    (0.0, 0.3, -0.3, 1.5, 0.0),
    (0.0, -1.4, 1.5, 0.2, 0.0),
    (0.0, -0.8, 1.2, 1.2, 1.57),
    (0.0, -0.8, 1.2, 1.2, -1.57),
)


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


class ArmIK:
    """IK helper bound to one arm prefix; works on a scratch MjData copy."""

    def __init__(self, model, prefix):
        self.model = model
        self.prefix = prefix
        self.site_id = model.site(prefix + "gripperframe").id
        self.gripper_body = model.body(prefix + "gripper").id
        self.base_body = model.body(prefix + "base").id
        self.joint_ids = [model.joint(prefix + j).id for j in ARM_JOINTS]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.lo = model.jnt_range[self.joint_ids, 0]
        self.hi = model.jnt_range[self.joint_ids, 1]
        self.scratch = mujoco.MjData(model)
        self.rng = np.random.default_rng(0)
        self.last_rot_err = 0.0

    def gripper_axes(self, data):
        """(approach, closing) unit vectors of the gripper in world frame."""
        rot = data.xmat[self.gripper_body].reshape(3, 3)
        return -rot[:, 2], rot[:, 0]

    def approach_axis(self, data):
        return self.gripper_axes(data)[0]

    def tcp(self, data):
        return data.site_xpos[self.site_id].copy()

    def _point_world(self, data, point):
        """World position of the controlled point: the TCP site, or a gripper-local point."""
        if point is None:
            return data.site_xpos[self.site_id]
        return data.xpos[self.gripper_body] + data.xmat[self.gripper_body].reshape(3, 3) @ point

    def _errors(self, data, target, axes, point):
        pos_err = target - self._point_world(data, point)
        rot_err = np.zeros(3)
        rot = data.xmat[self.gripper_body].reshape(3, 3)
        for local_column, sign, desired in axes:
            rot_err += np.cross(sign * rot[:, local_column], desired)
        return pos_err, rot_err

    def _descend(self, qpos_full, q_start, target, axes, point, rot_weight):
        d = self.scratch
        d.qpos[:] = qpos_full
        q = np.clip(q_start, self.lo, self.hi)
        best = (np.inf, q.copy(), np.inf, np.inf)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        oriented = bool(axes)
        for _ in range(MAX_ITERS):
            d.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            pos_err, rot_err = self._errors(d, target, axes, point)
            pos_norm, rot_norm = float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))
            cost = pos_norm + rot_weight * rot_norm
            if cost < best[0]:
                best = (cost, q.copy(), pos_norm, rot_norm)
            if pos_norm < POS_TOL and (not oriented or rot_norm < ROT_TOL):
                break
            if point is None:
                mujoco.mj_jacSite(self.model, d, jacp, None, self.site_id)
            else:
                mujoco.mj_jac(self.model, d, jacp, None, self._point_world(d, point), self.gripper_body)
            rows, errs = [jacp[:, self.dof_adr]], [pos_err]
            if oriented:
                mujoco.mj_jacBody(self.model, d, None, jacr, self.gripper_body)
                rows.append(rot_weight * jacr[:, self.dof_adr])
                errs.append(rot_weight * rot_err)
            jac, err = np.vstack(rows), np.concatenate(errs)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + DAMPING * np.eye(jac.shape[0]), err)
            q = np.clip(q + STEP * dq, self.lo, self.hi)
        return best

    def _seeds(self, qpos_full, target):
        current = np.asarray(qpos_full)[self.qpos_adr].copy()
        yield current
        base = self.scratch.xpos[self.base_body] if self.scratch.xpos.any() else np.zeros(3)
        azimuth = np.arctan2(target[1] - base[1], target[0] - base[0])
        for posture in CANONICAL:
            seed = np.array(posture, dtype=float)
            seed[0] = azimuth
            yield seed
        for _ in range(RANDOM_SEEDS):
            yield self.rng.uniform(self.lo, self.hi)

    def solve(self, qpos_full, target, approach=DOWN, rot_weight=ROT_WEIGHT, closing=None, restarts=True,
              lateral=None, point=None):
        """Return (arm joints, position residual); ``last_rot_err`` holds the orientation residual (rad).

        Orientation targets are world directions for gripper axes: ``approach`` (-z, where the
        fingers point), ``closing`` (+x, fixed jaw -> moving jaw) and ``lateral`` (+y, along the
        jaw width). With all three None the solve is position only. ``point`` is a point in the
        gripper body frame to place at ``target`` instead of the TCP site (e.g. a lip of a held
        bottle). ``restarts=False`` only refines from the current configuration (use it for
        small steps along a path).
        """
        target = np.asarray(target, dtype=float)
        axes = [(column, sign, _unit(direction))
                for column, sign, direction in ((2, -1.0, approach), (0, 1.0, closing), (1, 1.0, lateral))
                if direction is not None]
        point = None if point is None else np.asarray(point, dtype=float)
        self.scratch.qpos[:] = qpos_full
        mujoco.mj_kinematics(self.model, self.scratch)
        best = None
        for index, seed in enumerate(self._seeds(qpos_full, target)):
            result = self._descend(qpos_full, seed, target, axes, point, rot_weight)
            if best is None or result[0] < best[0]:
                best = result
            if best[0] < GOOD_ENOUGH or (not restarts and index == 0):
                break
        _, q, pos_norm, rot_norm = best
        self.last_rot_err = rot_norm
        return q, pos_norm


def gripper_rotation(approach, closing):
    """World rotation of the gripper body for a given approach and jaw-closing axis."""
    z_axis = -_unit(approach)
    closing = np.asarray(closing, dtype=float)
    x_axis = _unit(closing - np.dot(closing, z_axis) * z_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])
