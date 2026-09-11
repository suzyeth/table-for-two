"""Damped-least-squares IK for one SO-101 arm.

Solves for the five arm joints (gripper excluded) so the arm's ``gripperframe``
site reaches a target position while the gripper's approach axis points along
a requested direction (straight down by default). A 5-DoF arm can satisfy a
3-D position plus a 2-D pointing constraint, so both are solved jointly; if the
pointing constraint is infeasible the solver relaxes it and keeps position.
"""
import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
DOWN = np.array([0.0, 0.0, -1.0])

POS_TOL = 0.002
MAX_ITERS = 300
DAMPING = 1e-4
STEP = 0.6
ROT_WEIGHT = 0.15  # metres of position error traded per radian of pointing error


class ArmIK:
    """IK helper bound to one arm prefix; works on a scratch MjData copy."""

    def __init__(self, model, prefix):
        self.model = model
        self.prefix = prefix
        self.site_id = model.site(prefix + "gripperframe").id
        self.gripper_body = model.body(prefix + "gripper").id
        self.joint_ids = [model.joint(prefix + j).id for j in ARM_JOINTS]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.lo = model.jnt_range[self.joint_ids, 0]
        self.hi = model.jnt_range[self.joint_ids, 1]
        self.scratch = mujoco.MjData(model)

    def approach_axis(self, data):
        """Unit vector from the wrist-roll axis toward the fingertips (gripper body -z)."""
        return -data.xmat[self.gripper_body].reshape(3, 3)[:, 2]

    def tcp(self, data):
        return data.site_xpos[self.site_id].copy()

    def solve(self, qpos_full, target, approach=DOWN, rot_weight=ROT_WEIGHT):
        """Return (arm joint vector, position residual) for ``target``.

        ``qpos_full`` seeds the solve (usually the current simulation state).
        ``approach=None`` solves position only.
        """
        d = self.scratch
        d.qpos[:] = qpos_full
        q = d.qpos[self.qpos_adr].copy()
        best_q, best_err = q.copy(), np.inf
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(MAX_ITERS):
            d.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            pos_err = target - d.site_xpos[self.site_id]
            pos_norm = float(np.linalg.norm(pos_err))
            if pos_norm < best_err:
                best_q, best_err = q.copy(), pos_norm
            mujoco.mj_jacSite(self.model, d, jacp, None, self.site_id)
            rows, errs = [jacp[:, self.dof_adr]], [pos_err]
            if approach is not None:
                axis = self.approach_axis(d)
                rot_err = np.cross(axis, approach)
                mujoco.mj_jacBody(self.model, d, None, jacr, self.gripper_body)
                rows.append(rot_weight * jacr[:, self.dof_adr])
                errs.append(rot_weight * rot_err)
                if pos_norm < POS_TOL and np.linalg.norm(rot_err) < 0.02:
                    break
            elif pos_norm < POS_TOL:
                break
            jac, err = np.vstack(rows), np.concatenate(errs)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + DAMPING * np.eye(jac.shape[0]), err)
            q = np.clip(q + STEP * dq, self.lo, self.hi)
        if approach is not None and best_err > 0.01:
            # Pointing constraint infeasible here: fall back to a lighter weight.
            return self.solve(qpos_full, target, approach, rot_weight * 0.3) if rot_weight > 0.01 \
                else self.solve(qpos_full, target, None)
        return best_q, best_err
