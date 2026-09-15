"""Dinner-table task: plan format, bimanual executor and seeded evaluation.

A plan is a list of stages; each stage is a list of subtasks that run at the
same time (at most one per arm). Subtasks use a small, closed vocabulary so a
language planner can emit them as JSON:

  {"skill": "open_drawer", "arm": "left"}
  {"skill": "pick_place", "arm": "right", "object": "mug"}      (plate, spoon, fork or mug)
  {"skill": "handoff", "object": "spoon", "giver": "left", "receiver": "right"}
  {"skill": "bimanual_place", "object": "plate"}                 (both hands carry it level)
  {"skill": "pick_lift", "arm": "right", "object": "bottle"}
  {"skill": "pour", "arm": "right", "into": "mug"}
  {"skill": "return", "arm": "right", "object": "bottle"}
  {"skill": "place", "arm": "left", "object": "mug"}
  {"skill": "home", "arm": "left"}

All grasps are contact-only (see sim/skills.py): an object moves only while
the fingers hold it by friction.

Run:  .venv\\Scripts\\python.exe -m sim.task --seeds 0 1 2 --video out/scripted.mp4
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sim.env import ARMS, CONTROL_HZ, TABLE_TOP_Z, DinnerTableEnv
from sim.bimanual import carry_together, lockstep
from sim.grasping import horizontal, utensil_axis
from sim.skills import ArmSkills

ARM_KEY = {"left": "left_", "right": "right_"}
UTENSILS = ("spoon", "fork")
# Utensils are laid along x beside the plate: jaws close along y when setting them down.
UTENSIL_PLACE_CLOSING = np.array([0.0, 1.0, 0.0])
# Utensils are taken near the handle end facing the arms (the part out of the drawer).
TOWARD_ARMS = np.array([-1.0, 0.0, 0.0])
UTENSIL_PICK_ALONG = 0.015
# Objects the hand tips level while carrying. Not the plate: pinched by one wall, levelling it
# moves its centre of mass further from the pinch and it twists out of the fingers (audit:
# 6-8 deg tilt without levelling, 22 deg and one drop with it).
LEVEL_CARRY = ("mug",)
# Hand-over: the giver turns the utensil and holds its origin at HANDOFF_POINT with the head
# toward the receiver, 30 deg off crosswise; each hand takes its own end of the 7 cm handle,
# 2.8 cm from the centre. Each wrist-camera mount sticks out 4-8 cm to one side of its
# hand and each wrist link ~4 cm to the other, so both hands keep the camera side facing
# away from the other hand. A search over hand-over angle, grip points, height and camera
# sides (both arms solved, collision-body distances measured) found this the only layout
# with >1 cm between the arms; straight crosswise they touch.
HANDOFF_POINT = np.array([-0.01, 0.0, TABLE_TOP_Z + 0.06])
HANDOFF_YAW_DEG = -30  # head direction turned from crosswise (sign mirrors for a right-arm giver)
HANDOFF_GIVER_ALONG = 0.028
HANDOFF_RECEIVER_ALONG = 0.028
# The receiver first lines up this far out on its own side, then moves in over its end.
HANDOFF_APPROACH = 0.05
HOLD_POINT = np.array([0.02, 0.0, TABLE_TOP_Z + 0.03])  # pick_hold: mug origin held here
# Two-handed plate carry: the left hand pinches the wall 105 deg from +x, the right hand the
# opposite wall. From a search over plate start and grip angle with both arms solved at the
# start, midway and target, checking every collision body (arms, cabinet, props, table):
# 20 mm between the arms and 12 mm to everything else throughout.
PLATE_GRIP_DEG = 105
BIMANUAL_OBJECTS = ("plate",)
MAX_STEPS = 6000
SETTLE_BEFORE_SCORING_S = 1.0
# Every stage ends with the arms held still this long, recorded under the finished stage's label, so a
# policy learns to stop once a stage is done (the v2 policy, trained on demos whose next stage began 3
# control steps after the last motion, never came to rest and ran on into the next skill). Longer than
# the evaluator's settle window (policy/rollout.py, 2 s) and any pause inside a stage (the pour's 1.4 s).
STAGE_END_HOLD_S = 2.5

DEFAULT_INSTRUCTION = (
    "Carry the plate to the placemat with both hands, open the drawer and put the mug at the front "
    "right, lay the fork left of the plate, pour the water from the bottle into the mug and put "
    "the bottle back, then hand the spoon to the right arm to lay on the right."
)
# The plate goes first, while the drawer is still shut (the pulled-out drawer blocks the left
# hand's grip on the plate), and before the fork (setting a plate beside a laid fork knocks
# it); the right arm finishes with the bottle before it takes the spoon.
DEFAULT_PLAN = [
    [{"skill": "bimanual_place", "object": "plate"}],
    [{"skill": "open_drawer", "arm": "left"}, {"skill": "pick_place", "arm": "right", "object": "mug"}],
    [{"skill": "pick_place", "arm": "left", "object": "fork"},
     {"skill": "pick_lift", "arm": "right", "object": "bottle"}],
    [{"skill": "pour", "arm": "right", "into": "mug"}],
    [{"skill": "return", "arm": "right", "object": "bottle"}],
    [{"skill": "handoff", "object": "spoon", "giver": "left", "receiver": "right"}],
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
        self.start_xy = {o: self.env.object_frame(o)[0][:2].copy() for o in ("bottle", "mug", "plate")}

    def warnings(self):
        return [w for a in ARMS for w in self.skills[a].warnings]

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
        together = [s for s in stage if s.get("skill") == "bimanual_place"]
        if together:
            if len(stage) != 1:
                raise PlanError("a two-handed carry must be the only subtask in its stage")
            return self._bimanual_place(together[0])
        sub = {}
        for subtask in stage:
            arm = self._arm(subtask)
            if arm in sub:
                raise PlanError(f"two subtasks for {arm} in one stage: {stage}")
            sub[arm] = self._factory(arm, subtask)
        return [sub]

    def _place_closing(self, obj):
        return UTENSIL_PLACE_CLOSING if obj in UTENSILS else None

    def _turned_closing(self, skills, obj, new_axis):
        """Jaw closing direction that turns the held utensil's head to point along ``new_axis``."""
        axis = utensil_axis(self.env, obj)
        angle = np.arctan2(np.cross(axis, new_axis)[2], axis @ new_axis)
        closing = horizontal(skills.frame()[1][:, 0])
        c, s = np.cos(angle), np.sin(angle)
        return np.array([c * closing[0] - s * closing[1], s * closing[0] + c * closing[1], 0.0])

    def _factory(self, arm, subtask):
        sk, skill, obj = self.skills[arm], subtask.get("skill"), subtask.get("object")
        if skill == "open_drawer":
            return sk.open_drawer
        if skill == "pick_place":
            along, toward = ("balance", None) if obj in UTENSILS else (0.0, None)
            return lambda: _chain(sk.pick(obj, along, toward),
                                  sk.place(obj, self._target_xy(obj), self._place_closing(obj), level=obj in LEVEL_CARRY),
                                  sk.home())
        if skill == "pick_hold":
            return lambda: _chain(sk.pick(obj), sk.carry(obj, HOLD_POINT))
        if skill == "pick_lift":
            return sk.side_pick_bottle if obj == "bottle" else (lambda: sk.pick(obj))
        if skill == "pour":
            return lambda: sk.pour_into(subtask.get("into", "mug"))
        if skill == "place":
            return lambda: _chain(sk.place(obj, self._target_xy(obj), self._place_closing(obj)), sk.home())
        if skill == "return":
            if obj == "bottle":
                return lambda: _chain(sk.return_bottle(self.start_xy[obj]), sk.home())
            return lambda: _chain(sk.place(obj, self.start_xy[obj]), sk.home())
        if skill == "home":
            return sk.home
        raise PlanError(f"unknown skill '{skill}'")

    def _bimanual_place(self, subtask):
        """Both arms carry the object together; one generator drives both arms' commands."""
        obj = subtask.get("object")
        if obj not in BIMANUAL_OBJECTS:
            raise PlanError(f"only the {', '.join(BIMANUAL_OBJECTS)} is carried with both hands")
        angle = np.deg2rad(PLATE_GRIP_DEG)
        left_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
        dirs = {"left_": left_dir, "right_": -left_dir}
        both = self.skills
        return [{"left_": lambda: _chain(carry_together(self.env, both, obj, self._target_xy(obj), dirs),
                                          lockstep(both["left_"].home(), both["right_"].home()))}]

    def _handoff(self, subtask):
        giver, receiver = self._arm(subtask, "giver"), self._arm(subtask, "receiver")
        if giver == receiver:
            raise PlanError("a handoff needs two different arms")
        obj = subtask["object"]
        if obj not in UTENSILS:
            raise PlanError("only the spoon or fork can be handed over")
        g, r = self.skills[giver], self.skills[receiver]
        side = -1.0 if giver == "left_" else 1.0
        yaw = np.deg2rad(HANDOFF_YAW_DEG) * -side
        toward_receiver = np.array([-np.sin(yaw) * side, np.cos(yaw) * side, 0.0])
        # The giver's camera side starts toward the handle end (away from the head), so once the
        # hand turns the head toward the receiver it faces back to the giver's own side.
        return [
            {giver: lambda: g.pick(obj, along=HANDOFF_GIVER_ALONG, toward=TOWARD_ARMS,
                                   camera_side=-utensil_axis(self.env, obj))},
            {giver: lambda: g.carry(obj, HANDOFF_POINT, closing=self._turned_closing(g, obj, toward_receiver))},
            {receiver: lambda: r.pick(obj, along=HANDOFF_RECEIVER_ALONG, toward=toward_receiver, lift=0,
                                      approach_from=toward_receiver * HANDOFF_APPROACH,
                                      camera_side=toward_receiver)},
            {giver: lambda: _chain(g.release_and_retreat(), g.home())},
            {receiver: lambda: _chain(r.place(obj, self._target_xy(obj), UTENSIL_PLACE_CLOSING), r.home())},
        ]

    # ------------------------------------------------------------- running
    def _report(self, label):
        env = self.env
        positions = {o: np.round(env.object_frame(o)[0], 3).tolist() for o in ("plate", "mug", "bottle", "spoon", "fork")}
        holders = {o: env.holder(o) for o in ("plate", "mug", "bottle", "spoon", "fork")}
        drawer = float(env.data.qpos[env.drawer_qadr])
        print(f"  [{label}] held={ {o: h for o, h in holders.items() if h} } drawer={drawer:.3f} "
              f"water_in_mug={env.water_in_mug()}")
        print(f"      {positions}")

    def run(self, plan, on_step=None, max_steps=MAX_STEPS, verbose=False):
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
            hold = np.concatenate([self.skills[a].cmd for a in ARMS])
            for _ in range(int(round(STAGE_END_HOLD_S * CONTROL_HZ))):  # the stage is done: arms still
                if steps >= max_steps:
                    break
                if on_step:
                    on_step(hold, f"{index}:{label}")
                self.env.step(hold)
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
    env.hold(SETTLE_BEFORE_SCORING_S)  # let beads, plate and utensils come to rest before scoring
    result = env.success()
    if video is not None:
        video.extend(frames)
    return result, steps


def main():
    parser = argparse.ArgumentParser(description="Run the scripted dinner-table plan on seeded scenes.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--plan", type=Path, help="JSON plan file; defaults to the built-in plan")
    parser.add_argument("--stages", type=int, help="run only the first N stages (debugging)")
    parser.add_argument("--video", type=Path, help="write an MP4 of all seeds")
    parser.add_argument("--camera", default="operator")
    parser.add_argument("--verbose", action="store_true", help="print held objects and positions per stage")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else DEFAULT_PLAN
    if args.stages:
        plan = plan[:args.stages]
    env = DinnerTableEnv(obs_cameras=())
    executor = Executor(env)
    frames = [] if args.video else None
    results = []
    for seed in args.seeds:
        result, steps = run_seed(env, executor, seed, plan, video=frames, video_camera=args.camera,
                                 verbose=args.verbose)
        results.append(result)
        failed = [k for k, v in result.items() if not v and k != "all"]
        print(f"seed {seed:3d}: {'OK  ' if result['all'] else 'FAIL'} steps={steps:4d} "
              f"({steps / 20:.1f}s sim) failed={failed}")
        if args.verbose:
            for warning in executor.warnings()[:12]:
                print(f"    warning: {warning}")
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
