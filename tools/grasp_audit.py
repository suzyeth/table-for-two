"""Grasp audit: measure how the SO-101 fingers really meet each object.

Collision meshes in MuJoCo are convex hulls, and the fixed jaw's hull fills
the space between the fingers, so hull distances are meaningless for grasp
geometry. This tool works on the *visual* mesh vertices (the true shape):

  1. jaw gap vs gripper angle (closest fingertip-zone vertices of both jaws),
     plus the jaw closing direction in the gripper frame;
  2. for every grasp in a scripted run: the deepest penetration of finger
     vertices into the object's primitive geometry (negative = inside), the
     nearest finger-to-surface distance when not touching, and the gripper angle.

Run:  .venv\\Scripts\\python.exe -m tools.grasp_audit --seed 0
"""
import argparse

import mujoco
import numpy as np

from sim.env import ARMS, DinnerTableEnv
from sim.task import DEFAULT_PLAN, Executor

FINGER_BODIES = ("gripper", "moving_jaw_so101_v1")
FINGER_MESHES = ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1")  # the two jaw parts
TIP_ZONE = 0.035  # vertices within this distance of the fingertip count as "finger"
ANGLES = (-0.17, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.74)


def jaw_vertices(model, data, arm):
    """World-frame vertices of the two jaw visual meshes: (fixed_jaw_pts, moving_jaw_pts)."""
    clouds = []
    for body_name, mesh_name in zip(FINGER_BODIES, FINGER_MESHES):
        body = model.body(arm + body_name).id
        mesh_id = model.mesh(arm + mesh_name).id if _has_mesh(model, arm + mesh_name) else model.mesh(mesh_name).id
        pts = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == body and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH \
                    and model.geom_dataid[g] == mesh_id and model.geom_group[g] == 2:
                start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
                verts = model.mesh_vert[start:start + count]
                pts.append(verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
        clouds.append(np.vstack(pts))
    return clouds


def _has_mesh(model, name):
    try:
        model.mesh(name)
        return True
    except KeyError:
        return False


def fingertip_zone(points, approach, origin):
    reach = (points - origin) @ approach
    return points[reach > reach.max() - TIP_ZONE]


def jaw_gap_table(env):
    model = env.model
    data = mujoco.MjData(model)
    data.qpos[:] = env.data.qpos
    arm = ARMS[0]
    gripper_adr = env.qpos_idx[arm][5]
    body = env.gripper_body[arm]
    print("gripper angle (rad) -> fingertip gap (mm)")
    for angle in ANGLES:
        data.qpos[gripper_adr] = angle
        mujoco.mj_kinematics(model, data)
        approach = -data.xmat[body].reshape(3, 3)[:, 2]
        origin = data.site_xpos[env.tcp_site[arm]]
        fixed, moving = (fingertip_zone(c, approach, origin) for c in jaw_vertices(model, data, arm))
        dists = np.linalg.norm(fixed[:, None, :] - moving[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        axis = moving[j] - fixed[i]
        axis_g = data.xmat[body].reshape(3, 3).T @ (axis / (np.linalg.norm(axis) + 1e-9))
        print(f"  {angle:6.2f} -> {dists[i, j] * 1000:6.1f}   closing axis (gripper frame) {np.round(axis_g, 2)}")


def signed_distance(model, data, geom, points):
    """Signed distance from points to a box/cylinder/plane geom (negative = inside)."""
    local = (points - data.geom_xpos[geom]) @ data.geom_xmat[geom].reshape(3, 3)
    size = model.geom_size[geom]
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        q = np.abs(local) - size[:3]
        outside = np.linalg.norm(np.maximum(q, 0), axis=1)
        inside = np.minimum(q.max(axis=1), 0)
        return outside + inside
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radial = np.linalg.norm(local[:, :2], axis=1) - size[0]
        axial = np.abs(local[:, 2]) - size[1]
        q = np.stack([radial, axial], axis=1)
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    raise ValueError(f"unsupported geom type {kind}")


def object_geoms(env, obj):
    model = env.model
    if obj == "drawer":
        return [model.geom("drawer_handle").id]
    body = env.body_ids[obj]
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]


def audit_run(env, seed):
    executor = Executor(env)
    env.reset(seed)
    executor.reset()
    model, data = env.model, env.data
    seen, previous = {}, dict(env.held)

    def check(_action, _stage):
        nonlocal previous
        for arm, obj in env.held.items():
            if obj and previous.get(arm) != obj and obj not in seen:
                fingers = np.vstack(jaw_vertices(model, data, arm))
                depth = min(signed_distance(model, data, g, fingers).min() for g in object_geoms(env, obj))
                angle = float(data.qpos[env.qpos_idx[arm][5]])
                seen[obj] = (depth, angle, arm.rstrip("_"))
        previous = dict(env.held)

    executor.run(DEFAULT_PLAN, on_step=check)
    print(f"\ngrasps on seed {seed}: finger-to-surface (mm; negative = fingers inside the object)")
    for obj, (depth, angle, arm) in seen.items():
        verdict = "INSIDE" if depth < -0.002 else ("floating" if depth > 0.004 else "touching")
        print(f"  {obj:7s} {depth * 1000:7.1f}  gripper {angle:5.2f} rad  {arm:5s}  {verdict}")
    return seen


def main():
    parser = argparse.ArgumentParser(description="Audit finger/object geometry of every grasp.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    env = DinnerTableEnv(obs_cameras=())
    jaw_gap_table(env)
    audit_run(env, args.seed)


if __name__ == "__main__":
    main()
