"""Record corrective demonstrations from states the learned policy reaches (DAgger-style).

The plain demonstrations (``data/record.py``) only ever show the scripted skills
starting from states the scripted skills produced. The chained rollout breaks
because the states the *policy* leaves behind drift away from that distribution.
So here the policy drives the plan stage by stage, and at a split stage - or at
the first stage the policy fails, whichever comes first - the scripted skills
take over and finish the table. Only that scripted suffix is recorded, and only
when it actually sets the full table. Every kept frame therefore starts from a
state the policy itself produced.

The split stage cycles by seed over the boundaries where both hands are empty
(before the drawer stage, the fork stage and the hand-over), because a scripted
skill cannot inherit a grasp the policy made. The dataset has the same features as the plain one and
merges with it through ``tools/aggregate_contact.py``.

Run:  .venv\\Scripts\\python.exe -m data.record_dagger --episodes 100 --policy models/policy_v2/act_fp32.xml
        --checkpoint outputs/act_contact_v2/checkpoints/060000/pretrained_model --ensemble 0.01
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from data.record import (CAMERAS, FPS, IMAGE_SIZE, RECORD_EVERY, RECORD_NOISE, RECORD_SPREAD, REPO_ID,
                         SUBTASK_VOCAB, dataset_features, skip_reason, subtask_onehot)
from sim.env import ARMS, DinnerTableEnv
from sim.task import DEFAULT_INSTRUCTION, DEFAULT_PLAN, Executor
from data.command_noise import CommandNoise, free_space_gains, noisy
from tools.demo_gate import DemoGate, watched

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "dinner_table_contact_dagger"
FIRST_DAGGER_SEED = 500  # plain demos use 100-449; evaluation seeds 0-9 and 1000-1029 stay unseen
SPLIT_STAGES = (1, 2, 5)  # stage boundaries where both hands are empty: a scripted skill cannot take over a grasp
# the policy made (stages 3 and 4 start with the bottle in hand), and stage 0 has nothing policy-made yet.


def split_stage(seed, splits=SPLIT_STAGES):
    """Stage at which the scripted skills take over, cycling over ``splits`` by seed."""
    return splits[seed % len(splits)]


def takeover_stage(split, first_failed):
    """The suffix starts at the split, or earlier at the first stage the policy failed."""
    return split if first_failed is None else min(split, first_failed)


def vocab_index(stage_label, offset):
    """``Executor.run`` labels stages relative to the plan it ran; add the takeover offset."""
    return int(stage_label.split(":", 1)[0]) + offset


def run_dagger_episode(env, executor, policy, seed, split, noise_sigma=0.0, spread=1.0):
    """Policy prefix, scripted (recorded) suffix.

    Returns (frames, success, takeover, first_failed, defects); ``defects`` is what the demo gate
    found in the recorded suffix only (the policy prefix is not recorded). ``noise_sigma`` and
    ``spread`` are as in ``data.record``; the noise is added to the scripted suffix only.
    """
    from policy.rollout import run_stage_policy  # imported here: rollout pulls in OpenVINO

    env.reset(seed, spread=spread)
    executor.reset()
    first_failed = None
    for index, stage in enumerate(DEFAULT_PLAN[:split]):
        if not run_stage_policy(env, policy, stage, index):
            first_failed = index
            break
    takeover = takeover_stage(split, first_failed)

    for arm in ARMS:  # the scripted skills continue from the policy's last command, not from home
        executor.skills[arm].cmd = env.data.ctrl[env.act_idx[arm]].copy()
    # Not recorded: whatever the policy left moving (a failed stage ends without the evaluator's
    # confirmation hold) comes to rest first, so the gate does not blame the scripted suffix for it.
    env.hold(1.0)
    frames = []
    gate = DemoGate(env, executor)

    def capture(action, stage_label):
        gate.label = stage_label
        if env.time_step % RECORD_EVERY:
            return
        frame = {
            "observation.state": env.joint_state().astype(np.float32),
            "observation.environment_state": subtask_onehot(vocab_index(stage_label, takeover)),
            "action": np.asarray(action, dtype=np.float32),
            "task": DEFAULT_INSTRUCTION,
        }
        for cam in CAMERAS:
            frame[f"observation.images.{cam}"] = env.render(cam, IMAGE_SIZE)
        frames.append(frame)

    with watched(env, gate):
        with noisy(env, CommandNoise(noise_sigma, seed, gain=free_space_gains(executor))):
            executor.run(DEFAULT_PLAN[takeover:], on_step=capture)
        env.hold(1.0)
    return frames, env.success(), takeover, first_failed, gate.defects(executor.warnings())


def build_parser():
    parser = argparse.ArgumentParser(description="Record scripted corrections from policy-reached states.")
    parser.add_argument("--episodes", type=int, default=100, help="number of successful corrective episodes to keep")
    parser.add_argument("--start-seed", type=int, default=FIRST_DAGGER_SEED)
    parser.add_argument("--max-tries", type=int, default=160)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="delete an existing dataset at --root")
    parser.add_argument("--policy", type=Path, required=True, help="OpenVINO IR of the policy that drives the prefix")
    parser.add_argument("--checkpoint", type=Path, required=True, help="its LeRobot pretrained_model directory")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--ensemble", type=float, default=None, metavar="M", help="temporal ensembling coefficient")
    parser.add_argument("--splits", default=",".join(map(str, SPLIT_STAGES)),
                        help="comma-separated takeover stages to cycle over (default: empty-handed boundaries)")
    parser.add_argument("--noise", type=float, default=RECORD_NOISE,
                        help="command noise sigma (rad) on the scripted suffix's free-space moves")
    parser.add_argument("--spread", type=float, default=RECORD_SPREAD, help="placement spread of the table props")
    return parser


def main():
    from policy.ov_policy import OVActPolicy

    args = build_parser().parse_args()
    splits = tuple(int(x) for x in args.splits.split(","))

    if args.root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.root)

    policy = OVActPolicy(args.policy, args.checkpoint, device=args.device, ensemble_m=args.ensemble)
    dataset = LeRobotDataset.create(repo_id=REPO_ID, fps=FPS, features=dataset_features(),
                                    root=args.root, robot_type="bimanual_so101_sim", use_videos=False,
                                    image_writer_threads=4)
    env = DinnerTableEnv(obs_cameras=())
    executor = Executor(env)
    kept, tried, started = 0, 0, time.perf_counter()
    takeovers = {i: 0 for i in range(len(DEFAULT_PLAN))}
    for seed in range(args.start_seed, args.start_seed + args.max_tries):
        if kept >= args.episodes:
            break
        tried += 1
        split = split_stage(seed, splits)
        frames, result, takeover, first_failed, defects = run_dagger_episode(env, executor, policy, seed, split,
                                                                             args.noise, args.spread)
        why = f"policy failed stage {first_failed}" if first_failed is not None else f"split at {split}"
        reason = skip_reason(result, defects)
        if reason:
            print(f"seed {seed}: skipped, scripted suffix from stage {takeover} ({why}) {reason}")
            continue
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        kept += 1
        takeovers[takeover] += 1
        print(f"seed {seed}: kept episode {kept}/{args.episodes} ({len(frames)} frames, "
              f"takeover at stage {takeover}, {why})")
    dataset.finalize()
    env.close()
    print(f"kept {kept} of {tried} tried seeds in {(time.perf_counter() - started) / 60:.1f} min -> {args.root}")
    print("takeover stage histogram:", {SUBTASK_VOCAB[i]: n for i, n in takeovers.items() if n})


if __name__ == "__main__":
    main()
