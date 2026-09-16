"""Can one hand steady the mug while the other pours? A geometric search over layouts and grips.

The brief has one hand hold the cup while the other pours. An earlier search (12 mug positions x
pour directions x a body side-grip or a handle pinch) found at best 3 mm between the arms. This
one tries grips that search did not: a top-down pinch on the mug rim from any of 8 directions
(the fixed finger inside the wall, as the two-handed plate carry holds the plate), with the mug on
the table or lifted 4 or 8 cm towards the bottle by the steadying hand.

For each layout the right arm's pour is planned with the real planner (``PourMixin._plan_pour``,
from the state after the bottle is lifted on seed 0); the left arm is solved onto the rim pinch;
then, over every pour and lifted pose, the smallest distance between the left arm and the right
arm or the bottle (moving with the hand) is measured. Physics is not stepped: this only says
whether a layout is geometrically possible, i.e. worth building as a skill.

Run:  .venv\\Scripts\\python.exe -m tools.steady_search            (~10-20 min, one process)
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from scene.build_scene import MUG, TABLE_TOP_Z
from sim.env import DinnerTableEnv
from sim.grasping import DOWN, PLATE_OVERLAP, UP, pinch_point
from sim.task import DEFAULT_PLAN, Executor

ROOT = Path(__file__).resolve().parent.parent
POUR_STAGE = 3  # DEFAULT_PLAN index of the pour
FULL_POUR_DEG = 100  # a plan that stops below this tilt leaves water in the bottle
FEASIBLE_CLEARANCE = 0.010  # m between the arms over the whole pour
CHECK_RANGE = 0.05  # distances are capped here
IK_POS_TOL = 0.003
IK_ROT_TOL = np.deg2rad(5)
RIM_PINCH_BELOW_TOP = 0.004
MUG_XS = (0.00, 0.03, 0.06, 0.09, 0.12)
MUG_YS = (-0.10, -0.05, 0.00, 0.05)
LIFTS = (0.0, 0.04, 0.08)  # mug raised this far by the steadying hand
GRIP_DIRECTIONS_DEG = tuple(range(0, 360, 45))  # world yaw of the pinched wall, from the mug centre


def rim_pinch(mug_origin, yaw_deg):
    """(centre, closing, pinch) for a top-down pinch on the mug wall at world yaw ``yaw_deg``."""
    out = np.array([np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg)), 0.0])
    wall_mid = MUG["radius"] - MUG["wall"] / 2
    centre = mug_origin + UP * (MUG["height"] - RIM_PINCH_BELOW_TOP) + out * wall_mid
    return centre, out, pinch_point(MUG["wall"] / 2, PLATE_OVERLAP)


class Search:
    def __init__(self, seed):
        self.env = DinnerTableEnv(obs_cameras=())
        self.env.reset(seed)
        self.executor = Executor(self.env)
        self.executor.reset()
        self.executor.run(DEFAULT_PLAN[:POUR_STAGE])
        data, model = self.env.data, self.env.model
        self.snapshot = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
        self.right, self.left = self.executor.skills["right_"], self.executor.skills["left_"]
        self.right_cmd, self.left_cmd = self.right.cmd.copy(), self.left.cmd.copy()
        # Where the bottle sits in the right hand, kept for every pose (as _pour_pose_clear does).
        hand = self.right.ik.gripper_body
        hand_pos, hand_rot = data.xpos[hand].copy(), data.xmat[hand].reshape(3, 3).copy()
        bottle_pos, bottle_rot = self.env.object_frame("bottle")
        self.bottle_local = (hand_rot.T @ (bottle_pos - hand_pos), hand_rot.T @ bottle_rot)
        self.mug_adr = model.jnt_qposadr[model.body_jntadr[self.env.body_ids["mug"]]]
        self.bottle_adr = model.jnt_qposadr[model.body_jntadr[self.env.body_ids["bottle"]]]
        names = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
        collidable = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
        self.left_geoms = [g for g in collidable if names[g].startswith("left_")]
        self.other_geoms = [g for g in collidable if names[g].startswith("right_") or names[g] == "bottle"]
        self.scratch = mujoco.MjData(model)

    def restore(self, mug_xy, lift):
        data = self.env.data
        data.qpos[:], data.qvel[:], data.ctrl[:] = self.snapshot
        data.qpos[self.mug_adr:self.mug_adr + 3] = (mug_xy[0], mug_xy[1], TABLE_TOP_Z + lift)
        data.qpos[self.mug_adr + 3:self.mug_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.env.model, data)
        self.right.cmd, self.left.cmd = self.right_cmd.copy(), self.left_cmd.copy()
        self.right.warnings.clear()

    def clearance(self, q_left, q_right):
        """Smallest left-arm to right-arm/bottle distance with both arms at the given joints."""
        model, scratch = self.env.model, self.scratch
        scratch.qpos[:] = self.env.data.qpos
        scratch.qpos[self.left.ik.qpos_adr] = q_left
        scratch.qpos[self.right.ik.qpos_adr] = q_right
        mujoco.mj_kinematics(model, scratch)
        hand = self.right.ik.gripper_body
        pos, rot = scratch.xpos[hand], scratch.xmat[hand].reshape(3, 3)
        offset, bottle_rot = self.bottle_local
        scratch.qpos[self.bottle_adr:self.bottle_adr + 3] = pos + rot @ offset
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rot @ bottle_rot).ravel())
        scratch.qpos[self.bottle_adr + 3:self.bottle_adr + 7] = quat
        mujoco.mj_kinematics(model, scratch)
        return min(mujoco.mj_geomDistance(model, scratch, a, b, CHECK_RANGE, None)
                   for a in self.left_geoms for b in self.other_geoms)

    def layout(self, mug_xy, lift):
        """Rows (one per grip direction) for one mug position and lift."""
        self.restore(mug_xy, lift)
        mug_origin = self.env.object_frame("mug")[0]
        plan = self.right._plan_pour(mug_origin + UP * MUG["height"])
        max_tilt = plan[-1].tilt if plan else 0.0
        base = {"mug_xy": [round(v, 3) for v in mug_xy], "lift": lift, "max_tilt": max_tilt}
        if max_tilt < FULL_POUR_DEG:
            return [{**base, "grip_yaw": None, "clearance_mm": None, "why": "no full pour"}]
        poses = [q for step in plan for q in (step.q, step.q_raised)]
        rows = []
        for yaw in GRIP_DIRECTIONS_DEG:
            centre, closing, pinch = rim_pinch(mug_origin, yaw)
            q_left, err = self.left.ik.solve(self.left._qpos(), centre, restarts=True, approach=DOWN,
                                             closing=closing, point=pinch)
            if err > IK_POS_TOL or self.left.ik.last_rot_err > IK_ROT_TOL:
                rows.append({**base, "grip_yaw": yaw, "clearance_mm": None,
                             "why": f"left hand cannot reach ({err * 1000:.1f} mm)"})
                continue
            worst = min(self.clearance(q_left, q_right) for q_right in poses)
            rows.append({**base, "grip_yaw": yaw, "clearance_mm": round(worst * 1000, 1),
                         "why": "ok" if worst >= FEASIBLE_CLEARANCE else "arms too close"})
        return rows


def main():
    parser = argparse.ArgumentParser(description="Search layouts where one hand steadies the mug while the other pours.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "steady_search.json")
    args = parser.parse_args()
    started = time.perf_counter()
    search = Search(args.seed)
    print(f"state before the pour ready in {time.perf_counter() - started:.0f} s", flush=True)
    rows = []
    for x, y, lift in itertools.product(MUG_XS, MUG_YS, LIFTS):
        layout_rows = search.layout((x, y), lift)
        rows += layout_rows
        best = max((r["clearance_mm"] for r in layout_rows if r["clearance_mm"] is not None), default=None)
        print(f"mug ({x:.2f}, {y:+.2f}) lift {lift:.2f}: max tilt {layout_rows[0]['max_tilt']:.0f} deg, "
              f"best clearance {best} mm", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    ranked = sorted((r for r in rows if r["clearance_mm"] is not None), key=lambda r: -r["clearance_mm"])
    print("\nbest layouts (clearance between the arms over the whole pour):")
    for r in ranked[:15]:
        print(f"  mug {r['mug_xy']} lift {r['lift']:.2f} grip yaw {r['grip_yaw']:3d}: {r['clearance_mm']:6.1f} mm "
              f"(pour to {r['max_tilt']:.0f} deg)")
    feasible = [r for r in ranked if r["clearance_mm"] >= FEASIBLE_CLEARANCE * 1000]
    print(f"\n{len(feasible)} of {len(rows)} layout x grip combinations keep >= {FEASIBLE_CLEARANCE * 1000:.0f} mm; "
          f"{time.perf_counter() - started:.0f} s; wrote {args.out}")


if __name__ == "__main__":
    main()
