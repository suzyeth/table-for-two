"""Reachability check: can each arm's gripper frame reach the task targets?

Runs damped-least-squares IK (position only, joint limits clamped) from the
home pose toward a point just above each object and toward the drawer handle
(closed and fully open), then prints the residual error per arm.

Run:  .venv\\Scripts\\python.exe scene\\reach_check.py
"""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = ROOT / "scene" / "bimanual_table.xml"

IK_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ARMS = ("left_", "right_")
REACH_TOL = 0.01  # metres
APPROACH_HEIGHT = 0.03  # grasp point above the object centre
IK_ITERS = 400
IK_DAMPING = 1e-4
IK_STEP = 0.5


def solve_ik(model, data, prefix, target):
    """Move the arm's joints in ``data`` toward ``target``; return residual distance."""
    site_id = model.site(prefix + "gripperframe").id
    joint_ids = [model.joint(prefix + j).id for j in IK_JOINTS]
    dof_adr = [model.jnt_dofadr[j] for j in joint_ids]
    qpos_adr = [model.jnt_qposadr[j] for j in joint_ids]
    jacp = np.zeros((3, model.nv))
    for _ in range(IK_ITERS):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        err = target - data.site_xpos[site_id]
        if np.linalg.norm(err) < 1e-3:
            break
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        jac = jacp[:, dof_adr]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + IK_DAMPING * np.eye(3), err)
        for adr, jid, delta in zip(qpos_adr, joint_ids, dq):
            lo, hi = model.jnt_range[jid]
            data.qpos[adr] = np.clip(data.qpos[adr] + IK_STEP * delta, lo, hi)
    mujoco.mj_kinematics(model, data)
    return float(np.linalg.norm(target - data.site_xpos[site_id]))


def task_targets(model):
    """World-frame grasp points for every object, with the drawer closed and open."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lift = np.array([0.0, 0.0, APPROACH_HEIGHT])
    targets = {name: data.body(name).xpos.copy() + lift
               for name in ("plate", "mug", "bottle", "spoon", "fork")}
    targets["handle (closed)"] = data.geom("drawer_handle").xpos.copy()
    data.joint("drawer_slide").qpos[0] = model.jnt_range[model.joint("drawer_slide").id][1]
    mujoco.mj_forward(model, data)
    targets["handle (open)"] = data.geom("drawer_handle").xpos.copy()
    return targets


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    targets = task_targets(model)
    print(f"{'target':16s} {'xyz':>24s}   " + "   ".join(f"{a[:-1]:>8s}" for a in ARMS))
    for name, target in targets.items():
        results = []
        for prefix in ARMS:
            data = mujoco.MjData(model)
            residual = solve_ik(model, data, prefix, target)
            mark = "ok" if residual < REACH_TOL else "FAR"
            results.append(f"{residual * 100:5.1f}cm {mark:3s}")
        print(f"{name:16s} {np.array2string(target, precision=3):>24s}   " + "   ".join(results))


if __name__ == "__main__":
    main()
