"""Record scripted demonstrations of the dinner-table task as a LeRobot dataset.

Each episode runs the scripted plan on a randomised seed; only fully
successful episodes are kept. Frames are stored at 10 Hz (every second control
step) with two 128x128 camera views, the 12-D joint state, the 12-D joint
target that the script commanded, and a one-hot of the current subtask so a
policy can be conditioned on the planner's step.

Training seeds start at 100 so the evaluation seeds 0-9 stay unseen.

Run:  .venv\\Scripts\\python.exe -m data.record --episodes 50 --overwrite
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from sim.env import ARMS, JOINTS, DinnerTableEnv
from sim.task import DEFAULT_INSTRUCTION, DEFAULT_PLAN, Executor

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "dinner_table_contact"
REPO_ID = "local/bimanual_dinner_table_contact"
# Two scene views plus a camera on each wrist: with contact-only grasps the policy has to
# see how the jaws sit on the object, which the scene cameras barely resolve.
CAMERAS = ("overhead", "operator", "left_wrist_cam", "right_wrist_cam")
IMAGE_SIZE = (128, 128)
RECORD_EVERY = 2  # 20 Hz control loop -> 10 Hz dataset
FPS = 10
FIRST_TRAIN_SEED = 100


def stage_signature(stage):
    """Stable text id for a plan stage, e.g. 'pick_place:plate' or 'pick_hold:mug|pick_lift:bottle'."""
    parts = sorted(f"{s['skill']}:{s.get('object', s.get('into', ''))}" for s in stage)
    return "|".join(parts)


SUBTASK_VOCAB = [stage_signature(stage) for stage in DEFAULT_PLAN]
JOINT_NAMES = [f"{arm}{joint}" for arm in ARMS for joint in JOINTS]


def dataset_features():
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES},
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


def record_episode(env, executor, seed):
    """Run one seeded episode; return (frames, success dict)."""
    env.reset(seed)
    executor.reset()
    frames = []

    def capture(action, stage_label):
        if env.time_step % RECORD_EVERY:
            return
        stage_index = int(stage_label.split(":", 1)[0])
        frame = {
            "observation.state": env.joint_state().astype(np.float32),
            "observation.environment_state": subtask_onehot(stage_index),
            "action": np.asarray(action, dtype=np.float32),
            "task": DEFAULT_INSTRUCTION,
        }
        for cam in CAMERAS:
            frame[f"observation.images.{cam}"] = env.render(cam, IMAGE_SIZE)
        frames.append(frame)

    executor.run(DEFAULT_PLAN, on_step=capture)
    env.hold(1.0)  # score the episode once everything has come to rest
    return frames, env.success()


def main():
    parser = argparse.ArgumentParser(description="Record successful scripted episodes to a LeRobot dataset.")
    parser.add_argument("--episodes", type=int, default=150, help="number of successful episodes to keep")
    parser.add_argument("--start-seed", type=int, default=FIRST_TRAIN_SEED)
    parser.add_argument("--max-tries", type=int, default=200)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="delete an existing dataset at --root")
    args = parser.parse_args()

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
    for seed in range(args.start_seed, args.start_seed + args.max_tries):
        if kept >= args.episodes:
            break
        tried += 1
        frames, result = record_episode(env, executor, seed)
        if not result["all"]:
            failed = [k for k, v in result.items() if not v and k != "all"]
            print(f"seed {seed}: skipped (failed {failed})")
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
