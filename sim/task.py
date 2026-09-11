"""Dinner-table task: plan format, bimanual executor and seeded evaluation.

A plan is a list of stages; each stage is a list of subtasks that run at the
same time (at most one per arm). Subtasks use a small, closed vocabulary so a
language planner can emit them as JSON:

  {"skill": "open_drawer", "arm": "left"}
  {"skill": "pick_place", "arm": "right", "object": "plate"}
  {"skill": "handoff", "object": "fork", "giver": "left", "receiver": "right"}
  {"skill": "pick_hold", "arm": "left", "object": "mug"}
  {"skill": "pick_lift", "arm": "right", "object": "bottle"}
  {"skill": "pour", "arm": "right", "into": "mug"}
  {"skill": "place", "arm": "left", "object": "mug"}
  {"skill": "return", "arm": "right", "object": "bottle"}
  {"skill": "home", "arm": "left"}

Run:  .venv\\Scripts\\python.exe -m sim.task --seeds 0 1 2 --video out/scripted.mp4
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sim.env import ARMS, TABLE_TOP_Z, DinnerTableEnv
from sim.skills import ArmSkills

ARM_KEY = {"left": "left_", "right": "right_"}
# Where the left arm holds the mug for pouring (mirrored in y when the right arm
# holds it): slightly toward the pouring arm's side, so the pourer's shoulder
# never has to swing into the holding arm. Chosen by a sweep over four
# candidates on the seeds that failed before (6/6 poured, 6/6 full task).
HOLD_POINT = np.array([0.00, -0.03, TABLE_TOP_Z + 0.10])
HANDOFF_POINT = np.array([-0.04, 0.0, TABLE_TOP_Z + 0.13])
HANDOFF_REACH = 0.03  # receiver grasps this far along the object from its centre
HANDOFF_GIVER_OFFSET = 0.02  # giver grasps this far from the centre on its own side

DEFAULT_INSTRUCTION = (
    "Open the drawer, take out the spoon and fork, put the plate on the placemat, "
    "set the spoon on the left and the fork on the right, then hold the mug with the "
    "left arm and pour from the bottle with the right arm."
)
# Utensils come straight after the drawer: later arm motions can nudge the
# drawer back in, which would hide them under the cabinet top.
DEFAULT_PLAN = [
    [{"skill": "open_drawer", "arm": "left"}],
    [{"skill": "pick_place", "arm": "left", "object": "spoon"}],
    [{"skill": "handoff", "object": "fork", "giver": "left", "receiver": "right"}],
    [{"skill": "pick_place", "arm": "right", "object": "plate"}],
    [{"skill": "pick_hold", "arm": "left", "object": "mug"},
     {"skill": "pick_lift", "arm": "right", "object": "bottle"}],
    [{"skill": "pour", "arm": "right", "into": "mug"}],
    [{"skill": "return", "arm": "right", "object": "bottle"},
     {"skill": "place", "arm": "left", "object": "mug"}],
    [{"skill": "home", "arm": "left"}, {"skill": "home", "arm": "right"}],
]


class PlanError(ValueError):
    """Raised when a plan references an unknown skill, arm or object."""


class Executor:
    """Turns plan subtasks into per-arm generators and steps them in lockstep."""

    def __init__(self, env):
        self.env = env
        self.skills = {a: ArmSkills(env, a) for a in ARMS}
        self.start_xy = {}

    def reset(self):
        self.skills = {a: ArmSkills(self.env, a) for a in ARMS}
        self.start_xy = {o: self.env.grasp_point(o)[:2].copy() for o in ("bottle", "mug", "plate")}

    # ----------------------------------------------------------- expansion
    def _arm(self, subtask, key="arm"):
        try:
            return ARM_KEY[subtask[key]]
        except KeyError as exc:
            raise PlanError(f"subtask needs a valid '{key}' (left/right): {subtask}") from exc

    def _target_xy(self, obj):
        m = self.env.model
        return self.env.data.site_xpos[m.site(obj + "_target").id][:2].copy()

    def expand(self, stage):
        """Return a list of {arm: generator-factory} sub-stages for one plan stage."""
        handoffs = [s for s in stage if s.get("skill") == "handoff"]
        if handoffs:
            if len(stage) != 1:
                raise PlanError("a handoff must be the only subtask in its stage")
            return self._handoff(handoffs[0])
        sub = {}
        for subtask in stage:
            arm = self._arm(subtask)
            if arm in sub:
                raise PlanError(f"two subtasks for {arm} in one stage: {stage}")
            sub[arm] = self._factory(arm, subtask)
        return [sub]

    def _factory(self, arm, subtask):
        sk, skill, obj = self.skills[arm], subtask.get("skill"), subtask.get("object")
        if skill == "open_drawer":
            return sk.open_drawer
        if skill == "pick_place":
            return lambda: sk.pick_place(obj, self._target_xy(obj))
        if skill == "pick_hold":
            # Hold the mug on the holding arm's own side of the table.
            hold = HOLD_POINT * np.array([1.0, 1.0 if arm == "left_" else -1.0, 1.0])
            return lambda: _chain(sk.pick(obj), sk.lift_to(hold))
        if skill == "pick_lift":
            return lambda: sk.pick(obj)
        if skill == "pour":
            return lambda: sk.pour_into(subtask.get("into", "mug"))
        if skill == "place":
            return lambda: sk.place(obj, self._target_xy(obj))
        if skill == "return":
            return lambda: sk.place(obj, self.start_xy[obj])
        if skill == "home":
            return sk.home
        raise PlanError(f"unknown skill '{skill}'")

    def _handoff(self, subtask):
        giver, receiver = self._arm(subtask, "giver"), self._arm(subtask, "receiver")
        obj = subtask["object"]
        g, r = self.skills[giver], self.skills[receiver]
        toward_receiver = np.array([0.0, 1.0 if receiver == "left_" else -1.0, 0.0])

        def long_axis():
            """Utensil long axis (body x) in world frame, pointing toward the receiver."""
            axis = self.env.data.xmat[self.env.body_ids[obj]].reshape(3, 3)[:, 0]
            return axis * (np.sign(axis @ toward_receiver) or 1.0)

        def giver_pick():
            # Grip the giver's end so the receiver's end stays free.
            yield from g.pick(obj, offset=-HANDOFF_GIVER_OFFSET * long_axis())

        def receiver_grasp():
            point = self.env.grasp_point(obj) + HANDOFF_REACH * long_axis() + [0.0, 0.0, 0.004]
            yield from r.move(point + [0.0, 0.0, 0.05], grip=1.0)
            yield from r.move(point, speed=0.5)
            yield from r.grip(0.05, intent=obj)

        # Utensils already lie along y, so a top-down pick keeps the long axis
        # spanning the gap between the arms; no wrist roll needed.
        return [
            {giver: giver_pick},
            {giver: lambda: g.lift_to(HANDOFF_POINT)},
            {receiver: receiver_grasp},
            {giver: g.release_and_retreat},
            {giver: g.home, receiver: lambda: r.place(obj, self._target_xy(obj))},
        ]

    # ------------------------------------------------------------- running
    def _report(self, label):
        env = self.env
        positions = {o: np.round(env.grasp_point(o), 3).tolist() for o in ("plate", "mug", "bottle", "spoon", "fork")}
        drawer = float(env.data.qpos[env.drawer_qadr])
        print(f"  [{label}] held={env.held} drawer={drawer:.3f} pour={env.pour_time:.2f}s")
        print(f"      {positions}")

    def run(self, plan, on_step=None, max_steps=4000, verbose=False):
        """Execute ``plan``; ``on_step(action, stage_name)`` is called before every env step."""
        steps = 0
        for index, stage in enumerate(plan):
            label = "+".join(s.get("skill", "?") for s in stage)
            for sub_index, sub in enumerate(self.expand(stage)):
                if verbose and sub_index:
                    self._report(f"{index}:{label} part {sub_index}")
                gens = {arm: factory() for arm, factory in sub.items()}
                while gens and steps < max_steps:
                    for arm in list(gens):
                        try:
                            next(gens[arm])
                        except StopIteration:
                            del gens[arm]
                    if not gens:
                        break
                    action = np.concatenate([self.skills[a].cmd for a in ARMS])
                    if on_step:
                        on_step(action, f"{index}:{label}")
                    self.env.step(action)
                    steps += 1
            if verbose:
                self._report(f"{index}:{label} done")
        return steps


def _chain(*gens):
    for gen in gens:
        yield from gen


def run_seed(env, executor, seed, plan, video=None, video_camera="operator", video_every=2,
             verbose=False):
    env.reset(seed)
    executor.reset()
    frames = []

    def record(_action, _stage):
        if video is not None and env.time_step % video_every == 0:
            frames.append(env.render(video_camera))

    steps = executor.run(plan, on_step=record, verbose=verbose)
    result = env.success()
    if video is not None:
        video.extend(frames)
    return result, steps


def main():
    parser = argparse.ArgumentParser(description="Run the scripted dinner-table plan on seeded scenes.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--plan", type=Path, help="JSON plan file; defaults to the built-in plan")
    parser.add_argument("--video", type=Path, help="write an MP4 of all seeds")
    parser.add_argument("--verbose", action="store_true", help="print held objects and positions per stage")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else DEFAULT_PLAN
    env = DinnerTableEnv(obs_cameras=())
    executor = Executor(env)
    frames = [] if args.video else None
    results = []
    for seed in args.seeds:
        result, steps = run_seed(env, executor, seed, plan, video=frames, verbose=args.verbose)
        results.append(result)
        failed = [k for k, v in result.items() if not v and k != "all"]
        print(f"seed {seed:3d}: {'OK  ' if result['all'] else 'FAIL'} steps={steps:4d} "
              f"({steps / 20:.1f}s sim) failed={failed}")
    rate = np.mean([r["all"] for r in results])
    print(f"success rate: {rate * 100:.0f}% over {len(results)} seeds")
    for key in results[0]:
        if key != "all":
            print(f"  {key:14s} {np.mean([r[key] for r in results]) * 100:5.0f}%")
    if args.video and frames:
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=10)
        print(f"wrote {args.video} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
