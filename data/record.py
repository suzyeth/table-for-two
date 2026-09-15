"""Record scripted demonstrations of the dinner-table task as a LeRobot dataset.

Each episode runs the scripted plan on a randomised seed; only episodes that fully
succeed and pass the demo gate (``tools/demo_gate.py``: no drop, tip, knock, arm
hitting itself or unsafe planner warning) are kept. Frames are stored at 10 Hz (every second control
step) with two 128x128 camera views, the 12-D joint state, the 12-D joint
target that the script commanded, and a one-hot of the current subtask so a
policy can be conditioned on the planner's step.

Training seeds start at 100 so the evaluation seeds 0-9 stay unseen.

Run:  .venv\\Scripts\\python.exe -m data.record --episodes 50 --root data\\my_dataset --overwrite
      (for a full dataset use tools/record_parallel.py, which runs several of these side by side)
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from sim.env import ARMS, JOINTS, DinnerTableEnv
from sim.task import DEFAULT_INSTRUCTION, DEFAULT_PLAN, Executor
from data.command_noise import CommandNoise, free_space_gains, noisy
from tools.demo_gate import DemoGate, watched

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "local/bimanual_dinner_table_contact"
# Two scene views plus a camera on each wrist: with contact-only grasps the policy has to
# see how the jaws sit on the object, which the scene cameras barely resolve.
CAMERAS = ("overhead", "operator", "left_wrist_cam", "right_wrist_cam")
IMAGE_SIZE = (128, 128)
RECORD_EVERY = 2  # 20 Hz control loop -> 10 Hz dataset
FPS = 10
FIRST_TRAIN_SEED = 100
# 30 scenes no demonstration is recorded on (demos from FIRST_TRAIN_SEED, DAgger from 500): with 10
# scenes the 95% interval on a success rate was ~0.5 wide, with 30 it is ~0.3.
EVAL_SEEDS = range(1000, 1030)
# Recording defaults, from a demo-gate sweep on training seeds 100-109: 0.02 rad of free-space
# command noise with a 1.5x placement spread kept 9/10 (spread 1.5 alone: 9/10; uniform noise of
# 0.01 rad: 1/10), with noise on 22% of the control steps. The evaluation scenes keep spread 1.0.
RECORD_NOISE = 0.02
RECORD_SPREAD = 1.5


def stage_signature(stage):
    """Stable text id for a plan stage, e.g. 'pick_place:plate' or 'pick_hold:mug|pick_lift:bottle'."""
    parts = sorted(f"{s['skill']}:{s.get('object', s.get('into', ''))}" for s in stage)
    return "|".join(parts)


SUBTASK_VOCAB = [stage_signature(stage) for stage in DEFAULT_PLAN]
JOINT_NAMES = [f"{arm}{joint}" for arm in ARMS for joint in JOINTS]
# The policy sees joint velocities as well as positions: from one frame and positions alone it
# could not time the pour's tilt against the water flow (review 2).
STATE_NAMES = JOINT_NAMES + [f"{name}.vel" for name in JOINT_NAMES]


def policy_state(env):
    """``observation.state``: the 12 joint positions, then the 12 joint velocities (float32)."""
    return np.concatenate([env.joint_state(), env.joint_velocity()]).astype(np.float32)


def dataset_features():
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "observation.environment_state": {"dtype": "float32", "shape": (len(SUBTASK_VOCAB),),
                                          "names": SUBTASK_VOCAB},
        "action": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "image", "shape": (*IMAGE_SIZE, 3), "names": ["height", "width", "channels"]}
    return features


def subtask_onehot(stage_index):
    onehot = np.zeros(len(SUBTASK_VOCAB), dtype=np.float32)
    onehot[stage_index] = 1.0
    return onehot


def skip_reason(result, defects):
    """None if the episode may be kept; otherwise why not (failed checks, or what the demo gate found)."""
    if not result["all"]:
        return f"failed {[k for k, v in result.items() if not v and k != 'all']}"
    if defects:
        return f"defects {defects}"
    return None


def parse_shard(text):
    """``'K/N'`` -> ``(K, N)``: this recorder is shard K of N (0 <= K < N)."""
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError as exc:
        raise ValueError(f"shard must look like K/N, e.g. 2/8: {text!r}") from exc
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard K/N needs 0 <= K < N: {text!r}")
    return index, count


def shard_seeds(start, tries, shard):
    """The seeds shard ``(K, N)`` tries: start+K, start+K+N, ... - shards never share a scene."""
    index, count = shard
    return [start + index + count * i for i in range(tries)]


def record_episode(env, executor, seed, noise_sigma=0.0, spread=1.0):
    """Run one seeded episode; return (frames, success dict, demo-gate defects).

    ``noise_sigma``: on free-space moves (going home, moving from high up to a hover pose) the arm
    executes the scripted command plus slowly wandering noise (``data/command_noise.py``) while the
    clean command is recorded as the action label.
    ``spread``: placement spread of the table props (``DinnerTableEnv.reset``).
    """
    env.reset(seed, spread=spread)
    executor.reset()
    frames = []
    gate = DemoGate(env, executor)

    def capture(action, stage_label):
        gate.label = stage_label
        if env.time_step % RECORD_EVERY:
            return
        stage_index = int(stage_label.split(":", 1)[0])
        frame = {
            "observation.state": policy_state(env),
            "observation.environment_state": subtask_onehot(stage_index),
            "action": np.asarray(action, dtype=np.float32),
            "task": DEFAULT_INSTRUCTION,
        }
        for cam in CAMERAS:
            frame[f"observation.images.{cam}"] = env.render(cam, IMAGE_SIZE)
        frames.append(frame)

    with watched(env, gate):
        with noisy(env, CommandNoise(noise_sigma, seed, gain=free_space_gains(executor))):
            executor.run(DEFAULT_PLAN, on_step=capture)
        env.hold(1.0)  # score the episode once everything has come to rest (no noise)
    return frames, env.success(), gate.defects(executor.warnings())


def build_parser():
    parser = argparse.ArgumentParser(description="Record successful scripted episodes to a LeRobot dataset.")
    parser.add_argument("--episodes", type=int, default=150, help="number of successful episodes to keep")
    parser.add_argument("--start-seed", type=int, default=FIRST_TRAIN_SEED)
    parser.add_argument("--max-tries", type=int, default=200)
    parser.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="K/N",
                        help="record only shard K of N (seeds start+K, start+K+N, ...), for parallel recorders")
    # Required: with --overwrite a forgotten --root used to delete the old dataset at its default path.
    parser.add_argument("--root", type=Path, required=True, help="new dataset directory")
    parser.add_argument("--overwrite", action="store_true", help="delete an existing dataset at --root")
    parser.add_argument("--noise", type=float, default=RECORD_NOISE,
                        help="sigma (rad) of the slowly wandering noise added to the executed arm command on "
                             "free-space moves only; the clean command stays the label")
    parser.add_argument("--spread", type=float, default=RECORD_SPREAD,
                        help="placement spread of the plate, mug and bottle (evaluation scenes use 1.0)")
    return parser


def main():
    args = build_parser().parse_args()

    if args.root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.root)

    dataset = LeRobotDataset.create(repo_id=REPO_ID, fps=FPS, features=dataset_features(),
                                    root=args.root, robot_type="bimanual_so101_sim", use_videos=False,
                                    image_writer_threads=4)
    env = DinnerTableEnv(obs_cameras=())
    executor = Executor(env)
    kept, tried, started = 0, 0, time.perf_counter()
    for seed in shard_seeds(args.start_seed, args.max_tries, args.shard):
        if kept >= args.episodes:
            break
        tried += 1
        frames, result, defects = record_episode(env, executor, seed, args.noise, args.spread)
        reason = skip_reason(result, defects)
        if reason:
            print(f"seed {seed}: skipped ({reason})")
            continue
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        kept += 1
        print(f"seed {seed}: kept episode {kept}/{args.episodes} ({len(frames)} frames)")
    dataset.finalize()
    env.close()
    elapsed = time.perf_counter() - started
    print(f"kept {kept} of {tried} tried seeds in {elapsed / 60:.1f} min -> {args.root}")
    print(f"subtask vocabulary: {SUBTASK_VOCAB}")


if __name__ == "__main__":
    main()
