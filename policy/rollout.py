"""Evaluate the learned ACT policy (OpenVINO) on seeded dinner-table scenes.

The plan (from the language planner or the default plan) is run stage by
stage. For each stage the policy receives the camera views, the joint state
and the stage's one-hot subtask; it outputs joint targets at 10 Hz. A stage
ends when its success predicate holds, or after a timeout. With ``--settle`` it
ends only once the policy's commands have also come to rest - for policies
trained on demos that end every stage with a still hold (sim/task.py
STAGE_END_HOLD_S); earlier policies never stop and run on into the next skill.

Modes:
  * ``policy`` - the learned policy alone; a timed-out stage is a failure;
  * ``hybrid`` - if the policy times out, the scripted skill finishes that
    stage and the stage is counted as "assisted" (reported separately).

Run:  .venv\\Scripts\\python.exe -m policy.rollout --policy models/policy_v2/act_fp32.xml ^
        --checkpoint outputs/act_contact_v2/checkpoints/060000/pretrained_model --seeds 0 1 2
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from data.record import (CAMERAS, EVAL_SEEDS, IMAGE_SIZE, RECORD_EVERY, SUBTASK_VOCAB, policy_state, stage_signature,
                         subtask_onehot)
from policy.video_views import compose_views
from policy.ov_policy import OVActPolicy
from sim.env import ARMS, TABLE_TOP_Z, UPRIGHT_TOL_DEG, DinnerTableEnv
from sim.task import ARM_KEY, DEFAULT_PLAN, Executor

ROOT = Path(__file__).resolve().parent.parent
STAGE_TIMEOUT_S = 40.0  # the longest contact stage (hand-over) takes ~25 s scripted
HOME_SECONDS = 2.0
LIFTED_ABOVE_TABLE = 0.06
CONFIRM_SECONDS = 1.0
# With ``settle`` a stage ends where the demonstrations end it: after the check passes the policy keeps running the
# same stage (letting go, going home) until its commands have been still for SETTLE_STILL_STEPS,
# counted from when the check passed, or for at most SETTLE_LIMIT_STEPS more steps. 2 s is longer
# than any pause inside a demonstrated stage (the pour holds the tilted bottle still for 1.4 s),
# so a pause under way cannot end the stage before the bottle is straightened.
SETTLE_STILL_RAD = 0.01  # a command changing less than this per 10 Hz step counts as still
SETTLE_STILL_STEPS = 20
SETTLE_LIMIT_STEPS = 100  # 10 s


def is_scored(stage):
    """``home`` stages have no physical outcome and are never counted as policy successes."""
    return any(sub["skill"] != "home" for sub in stage)


def stage_done(env, stage, elapsed_s):
    """Success predicate for one plan stage, read from the physical state.

    This is an oracle: the evaluator, not the policy, decides when a stage is complete,
    using simulator state (contacts, poses, bead counts). Holding is contact-based: an
    object counts as held by an arm only while both of that arm's jaws touch it.
    """
    result = env.success()
    lifted = TABLE_TOP_Z + LIFTED_ABOVE_TABLE
    checks = []
    for sub in stage:
        skill, obj = sub["skill"], sub.get("object")
        if skill == "open_drawer":
            checks.append(result["drawer_open"] and env.holder("drawer") is None)
        elif skill in ("pick_place", "handoff", "place", "bimanual_place"):
            checks.append(result[f"{obj}_placed"])
        elif skill in ("pick_hold", "pick_lift"):
            arm = ARM_KEY[sub["arm"]]
            checks.append(env.holder(obj) == arm and env.object_frame(obj)[0][2] > lifted)
        elif skill == "pour":
            checks.append(result["poured"])
        elif skill == "return":
            checks.append(result["bottle_returned"] if obj == "bottle" else result.get(f"{obj}_placed", False))
        elif skill == "home":
            checks.append(elapsed_s >= HOME_SECONDS)
    return all(checks)


def policy_input(env, policy):
    """The state the checkpoint was trained on: its first ``state_dim`` values (a checkpoint trained on
    joint positions only, v1/v2, gets the 12 positions; newer ones also get the 12 velocities)."""
    state = policy_state(env)
    dim = getattr(policy, "state_dim", None) or len(state)
    if dim > len(state):
        raise ValueError(f"the checkpoint expects a {dim}-D state; the simulator gives {len(state)} values")
    return state[:dim]


def run_stage_policy(env, policy, stage, stage_index_in_vocab, frames=None, head=None, video_cameras=("operator",),
                     settle=False):
    """Drive one stage with the policy; True if its predicate holds, and still holds after
    the arms have been held still for CONFIRM_SECONDS (so a bead splash or a plate that is
    about to tip over does not count).

    With ``settle``, once the predicate holds the policy keeps running this stage until its
    commands have been still for SETTLE_STILL_STEPS (or for at most SETTLE_LIMIT_STEPS), so
    the next stage starts where a demonstrated one does - after the hand has let go and gone
    home. A predicate that no longer holds once settled sends the stage back to running. Only
    for policies trained on demos that end each stage with a still hold: the v2 policy never
    comes to rest and runs on into the next skill (pour 6/10 -> 3/10 with settling).

    With ``head`` (a stage-completion head, policy/stage_head.py) the *policy side* decides
    when the stage is over: the loop ends when the head has said "done" for its streak, and
    the oracle predicate is then used only to score that decision. Without it the oracle
    both ends the stage and scores it.
    """
    onehot = subtask_onehot(stage_index_in_vocab)
    policy.reset()
    if head is not None:
        head.reset()
    steps = int(STAGE_TIMEOUT_S * 20 / RECORD_EVERY)
    done_at, still, previous = None, 0, None
    for k in range(steps + SETTLE_LIMIT_STEPS):  # the extra steps only let a late success settle
        if k >= steps and done_at is None:
            break
        images = {f"observation.images.{cam}": env.render(cam, IMAGE_SIZE) for cam in CAMERAS}
        state = policy_input(env, policy)
        action = policy.select_action(state, onehot, images)
        command = np.asarray(action, dtype=float)
        still = still + 1 if previous is not None and np.max(np.abs(command - previous)) < SETTLE_STILL_RAD else 0
        previous = command.copy()
        for _ in range(RECORD_EVERY):  # dataset is 10 Hz, control loop is 20 Hz
            env.step(action)
        if frames is not None:
            frames.append(compose_views(env, video_cameras))
        elapsed = (k + 1) * RECORD_EVERY / 20
        if head is not None:
            head.done(state, onehot, images)
            if head.fired:
                env.hold(CONFIRM_SECONDS)
                return stage_done(env, stage, elapsed + CONFIRM_SECONDS)
            continue
        if done_at is None and k < steps and stage_done(env, stage, elapsed):
            done_at, still = k, 0
        if done_at is not None and (not settle or still >= SETTLE_STILL_STEPS or k - done_at >= SETTLE_LIMIT_STEPS):
            env.hold(CONFIRM_SECONDS)
            if stage_done(env, stage, elapsed + CONFIRM_SECONDS):
                return True
            done_at = None
    return False


def run_stage_scripted(env, executor, stage):
    for arm in ARMS:
        executor.skills[arm].cmd = env.data.ctrl[env.act_idx[arm]].copy()
    executor.run([stage])


def evaluate(policy, seeds, plan, mode, video_frames=None, head=None, video_cameras=("operator",), settle=False):
    env = DinnerTableEnv(obs_cameras=())
    executor = Executor(env)
    per_seed = []
    for seed in seeds:
        env.reset(seed)
        executor.reset()
        stages = []
        clean = True  # every earlier scored stage solved by the policy itself
        for stage in plan:
            signature = stage_signature(stage)
            if signature not in SUBTASK_VOCAB:
                raise ValueError(f"stage '{signature}' was never demonstrated; the policy cannot run it")
            ok = run_stage_policy(env, policy, stage, SUBTASK_VOCAB.index(signature), video_frames, head, video_cameras,
                                  settle)
            assisted = False
            if not ok and mode == "hybrid":
                run_stage_scripted(env, executor, stage)
                assisted = True
            scored = is_scored(stage)
            # "policy_ok" is only meaningful if the stage started from a state the policy earned.
            stages.append({"stage": signature, "scored": scored, "policy_ok": ok and scored,
                           "policy_ok_from_clean_state": ok and scored and clean, "assisted": assisted})
            if scored and not ok:
                clean = False
        env.hold(CONFIRM_SECONDS)
        result = env.success()
        per_seed.append({"seed": seed, "success": result, "stages": stages})
        scored_stages = [s for s in stages if s["scored"]]
        policy_stages = sum(s["policy_ok"] for s in scored_stages)
        print(f"seed {seed}: all={result['all']} policy-solved stages {policy_stages}/{len(scored_stages)} "
              f"assisted={[s['stage'] for s in stages if s['assisted']]}")
    env.close()
    return per_seed


def evaluate_stagewise(policy, seeds, plan, head=None, settle=False):
    """Per-skill success: script every earlier stage, then let the policy do just this one.

    Chained rollouts hide which skills the policy has learned, because one failed
    stage leaves every later stage in a state it never saw in training.
    """
    env = DinnerTableEnv(obs_cameras=())
    executor = Executor(env)
    names = [stage_signature(stage) for stage in plan]
    wins = {name: 0 for name in names if is_scored(plan[names.index(name)])}
    for seed in seeds:
        for index, stage in enumerate(plan):
            if not is_scored(stage):
                continue
            env.reset(seed)
            executor.reset()
            executor.run(plan[:index])
            for arm in ARMS:
                executor.skills[arm].cmd = env.data.ctrl[env.act_idx[arm]].copy()
            ok = run_stage_policy(env, policy, stage, SUBTASK_VOCAB.index(names[index]), head=head, settle=settle)
            wins[names[index]] += ok
        print(f"seed {seed}: cumulative per-stage wins {wins}")
    env.close()
    return {name: wins[name] / len(seeds) for name in wins}


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval (low, high) for ``successes`` out of ``n``: with 10 scenes a single
    rate says little (the same checkpoint once scored the pour 8/10, then 3/10)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(per_seed, policy):
    keys = [k for k in per_seed[0]["success"] if k != "all"]
    scored = [i for i, s in enumerate(per_seed[0]["stages"]) if s["scored"]]
    names = {i: per_seed[0]["stages"][i]["stage"] for i in scored}
    return {
        "task_success_rate": float(np.mean([r["success"]["all"] for r in per_seed])),
        "task_success_ci95": wilson_interval(sum(bool(r["success"]["all"]) for r in per_seed), len(per_seed)),
        "subgoal_success": {k: float(np.mean([r["success"][k] for r in per_seed])) for k in keys},
        "policy_stage_success": {names[i]: float(np.mean([r["stages"][i]["policy_ok"] for r in per_seed]))
                                 for i in scored},
        "policy_stage_success_ci95": {
            names[i]: wilson_interval(sum(bool(r["stages"][i]["policy_ok"]) for r in per_seed), len(per_seed))
            for i in scored},
        "policy_stage_success_from_clean_state": {
            names[i]: float(np.mean([r["stages"][i]["policy_ok_from_clean_state"] for r in per_seed])) for i in scored},
        "assisted_stages": {names[i]: int(sum(r["stages"][i]["assisted"] for r in per_seed)) for i in scored},
        "assisted_stages_per_episode": float(np.mean([sum(s["assisted"] for s in r["stages"]) for r in per_seed])),
        "episodes": len(per_seed),
        "policy_device": policy.device,
        "policy_infer_ms_mean": round(float(np.mean(policy.infer_ms)), 3) if policy.infer_ms else None,
        "policy_compile_s": round(policy.compile_s, 2),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate the OpenVINO ACT policy on seeded scenes.")
    # Both required: the old defaults were the v1 INT8 export and the v1 checkpoint.
    parser.add_argument("--policy", type=Path, required=True, help="OpenVINO IR, e.g. models/policy_v2/act_fp32.xml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="its LeRobot pretrained_model directory")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    parser.add_argument("--mode", choices=("policy", "hybrid"), default="hybrid")
    parser.add_argument("--switch", choices=("oracle", "head"), default="oracle",
                        help="who ends a stage: the simulator oracle, or the policy's stage-completion head")
    parser.add_argument("--head", type=Path, default=ROOT / "models" / "stage_head" / "stage_head_int8.xml")
    parser.add_argument("--settle", action="store_true",
                        help="after a stage's check passes, run on until the policy's commands come to rest (only "
                             "for policies trained on demos with a still hold at the end of every stage)")
    parser.add_argument("--ensemble", type=float, default=None, metavar="M",
                        help="temporal ensembling: infer every step and blend overlapping chunks with "
                             "weights exp(-M*i) (ACT paper uses 0.01); default = LeRobot action queue")
    parser.add_argument("--stagewise", action="store_true",
                        help="score each stage separately, starting it from a scripted state")
    parser.add_argument("--plan", type=Path, help="JSON plan (e.g. from the planner); default plan otherwise")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-cameras", default="operator",
                        help="comma-separated cameras for --video; several are tiled into one labelled frame "
                             "(e.g. front_high,overhead,left_wrist_cam,right_wrist_cam)")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "rollout_report.json")
    return parser


def main():
    args = build_parser().parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else DEFAULT_PLAN
    policy = OVActPolicy(args.policy, args.checkpoint, device=args.device, ensemble_m=args.ensemble)
    head = None
    if args.switch == "head":
        from policy.stage_head import OVStageHead
        head = OVStageHead(args.head, device=args.device)
    if args.stagewise:
        per_stage = evaluate_stagewise(policy, args.seeds, plan, head, args.settle)
        report = {"policy": str(args.policy), "mode": "stagewise", "switch": args.switch, "settle": args.settle,
                  "ensemble": args.ensemble, "seeds": args.seeds, "per_stage_success": per_stage,
                  "per_stage_ci95": {name: wilson_interval(round(rate * len(args.seeds)), len(args.seeds))
                                     for name, rate in per_stage.items()},
                  "policy_infer_ms_mean": round(float(np.mean(policy.infer_ms)), 3) if policy.infer_ms else None}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(report, indent=1))
        return
    frames = [] if args.video else None
    per_seed = evaluate(policy, args.seeds, plan, args.mode, frames, head,
                        tuple(c.strip() for c in args.video_cameras.split(",") if c.strip()), args.settle)
    report = {"policy": str(args.policy), "mode": args.mode, "switch": args.switch, "settle": args.settle,
              "ensemble": args.ensemble, "seeds": args.seeds,
              "summary": summarise(per_seed, policy), "episodes": per_seed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    if args.video and frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=10)
        print(f"wrote {args.video} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
