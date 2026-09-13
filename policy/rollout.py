"""Evaluate the learned ACT policy (OpenVINO) on seeded dinner-table scenes.

The plan (from the language planner or the default plan) is run stage by
stage. For each stage the policy receives the camera views, the joint state
and the stage's one-hot subtask; it outputs joint targets at 10 Hz. A stage
ends when its success predicate holds, or after a timeout.

Modes:
  * ``policy`` - the learned policy alone; a timed-out stage is a failure;
  * ``hybrid`` - if the policy times out, the scripted skill finishes that
    stage and the stage is counted as "assisted" (reported separately).

Run:  .venv\\Scripts\\python.exe -m policy.rollout --policy models/policy/act_int8.xml --seeds 0 1 2
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data.record import CAMERAS, IMAGE_SIZE, RECORD_EVERY, SUBTASK_VOCAB, stage_signature, subtask_onehot
from policy.export_openvino import DEFAULT_CKPT
from policy.video_views import compose_views
from policy.ov_policy import OVActPolicy
from sim.env import ARMS, TABLE_TOP_Z, UPRIGHT_TOL_DEG, DinnerTableEnv
from sim.task import ARM_KEY, DEFAULT_PLAN, Executor

ROOT = Path(__file__).resolve().parent.parent
STAGE_TIMEOUT_S = 40.0  # the longest contact stage (hand-over) takes ~25 s scripted
HOME_SECONDS = 2.0
LIFTED_ABOVE_TABLE = 0.06
CONFIRM_SECONDS = 1.0


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


def run_stage_policy(env, policy, stage, stage_index_in_vocab, frames=None, head=None, video_cameras=("operator",)):
    """Drive one stage with the policy; True if its predicate holds, and still holds after
    the arms have been held still for CONFIRM_SECONDS (so a bead splash or a plate that is
    about to tip over does not count).

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
    for k in range(steps):
        images = {f"observation.images.{cam}": env.render(cam, IMAGE_SIZE) for cam in CAMERAS}
        state = env.joint_state()
        action = policy.select_action(state, onehot, images)
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
        if stage_done(env, stage, elapsed):
            env.hold(CONFIRM_SECONDS)
            if stage_done(env, stage, elapsed + CONFIRM_SECONDS):
                return True
    return False


def run_stage_scripted(env, executor, stage):
    for arm in ARMS:
        executor.skills[arm].cmd = env.data.ctrl[env.act_idx[arm]].copy()
    executor.run([stage])


def evaluate(policy, seeds, plan, mode, video_frames=None, head=None, video_cameras=("operator",)):
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
            ok = run_stage_policy(env, policy, stage, SUBTASK_VOCAB.index(signature), video_frames, head, video_cameras)
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


def evaluate_stagewise(policy, seeds, plan, head=None):
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
            ok = run_stage_policy(env, policy, stage, SUBTASK_VOCAB.index(names[index]), head=head)
            wins[names[index]] += ok
        print(f"seed {seed}: cumulative per-stage wins {wins}")
    env.close()
    return {name: wins[name] / len(seeds) for name in wins}


def summarise(per_seed, policy):
    keys = [k for k in per_seed[0]["success"] if k != "all"]
    scored = [i for i, s in enumerate(per_seed[0]["stages"]) if s["scored"]]
    names = {i: per_seed[0]["stages"][i]["stage"] for i in scored}
    return {
        "task_success_rate": float(np.mean([r["success"]["all"] for r in per_seed])),
        "subgoal_success": {k: float(np.mean([r["success"][k] for r in per_seed])) for k in keys},
        "policy_stage_success": {names[i]: float(np.mean([r["stages"][i]["policy_ok"] for r in per_seed]))
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate the OpenVINO ACT policy on seeded scenes.")
    parser.add_argument("--policy", type=Path, default=ROOT / "models" / "policy" / "act_int8.xml")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--mode", choices=("policy", "hybrid"), default="hybrid")
    parser.add_argument("--switch", choices=("oracle", "head"), default="oracle",
                        help="who ends a stage: the simulator oracle, or the policy's stage-completion head")
    parser.add_argument("--head", type=Path, default=ROOT / "models" / "stage_head" / "stage_head_int8.xml")
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
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else DEFAULT_PLAN
    policy = OVActPolicy(args.policy, args.checkpoint, device=args.device, ensemble_m=args.ensemble)
    head = None
    if args.switch == "head":
        from policy.stage_head import OVStageHead
        head = OVStageHead(args.head, device=args.device)
    if args.stagewise:
        per_stage = evaluate_stagewise(policy, args.seeds, plan, head)
        report = {"policy": str(args.policy), "mode": "stagewise", "switch": args.switch, "ensemble": args.ensemble, "seeds": args.seeds, "per_stage_success": per_stage,
                  "policy_infer_ms_mean": round(float(np.mean(policy.infer_ms)), 3) if policy.infer_ms else None}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(report, indent=1))
        return
    frames = [] if args.video else None
    per_seed = evaluate(policy, args.seeds, plan, args.mode, frames, head,
                        tuple(c.strip() for c in args.video_cameras.split(",") if c.strip()))
    report = {"policy": str(args.policy), "mode": args.mode, "switch": args.switch, "ensemble": args.ensemble, "seeds": args.seeds,
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
