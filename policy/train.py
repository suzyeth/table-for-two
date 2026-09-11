"""Train the ACT visuomotor policy on the recorded dinner-table demonstrations.

Reproducible wrapper around LeRobot's training script with the settings used
for the submission. The dataset is the local LeRobot dataset written by
``data/record.py``; nothing is downloaded from or pushed to the Hugging Face Hub.

Policy inputs: four 128x128 camera views (overhead, operator and one on each
wrist), the 12-D joint state and the one-hot subtask from the planner
(``observation.environment_state``). Output: a chunk of future 12-D joint
targets at 10 Hz.

Windows note: LeRobot marks the newest checkpoint with a ``last`` symlink,
which Windows refuses without Developer Mode. Training therefore runs
in-process with that one helper replaced by a plain-text pointer file
(``checkpoints/last_checkpoint.txt``); ``latest_pretrained`` resolves it.

Run:  .venv\\Scripts\\python.exe -m policy.train --steps 20000
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "act_contact"  # contact-physics policy (v1/v2 weld-era runs keep their own dirs)
POINTER_FILE = "last_checkpoint.txt"

CHUNK_SIZE = 20  # 2 s of actions at 10 Hz
N_ACTION_STEPS = 10  # re-plan every second


def latest_pretrained(output_dir=OUTPUT_DIR):
    """Path of the newest ``pretrained_model`` directory under ``output_dir/checkpoints``."""
    checkpoints = Path(output_dir) / "checkpoints"
    numbered = sorted((p for p in checkpoints.glob("[0-9]*") if p.is_dir()), key=lambda p: int(p.name))
    if not numbered:
        return checkpoints / "last" / "pretrained_model"
    return numbered[-1] / "pretrained_model"


def _write_pointer(checkpoint_dir):
    """Replacement for LeRobot's symlink-based ``update_last_checkpoint``."""
    checkpoint_dir = Path(checkpoint_dir)
    (checkpoint_dir.parent / POINTER_FILE).write_text(checkpoint_dir.name, encoding="utf-8")


def build_args(args):
    from data.record import REPO_ID

    return [
        f"--dataset.repo_id={REPO_ID}",
        f"--dataset.root={args.dataset_root}",
        "--policy.type=act",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={CHUNK_SIZE}",
        f"--policy.n_action_steps={N_ACTION_STEPS}",
        f"--output_dir={args.output_dir}",
        "--job_name=act_contact",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--save_freq={args.save_freq}",
        "--log_freq=100",
        "--wandb.enable=false",
    ]


def main():
    from data.record import DATA_ROOT

    parser = argparse.ArgumentParser(description="Train ACT on the local dinner-table dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="print the arguments without training")
    args = parser.parse_args()

    if not (args.dataset_root / "meta" / "info.json").exists():
        raise SystemExit(f"no LeRobot dataset at {args.dataset_root}; run: python -m data.record --episodes 150")
    if args.output_dir.exists():
        raise SystemExit(f"{args.output_dir} exists; choose another --output-dir or remove it first")

    train_args = build_args(args)
    print("lerobot-train " + " ".join(train_args))
    if args.dry_run:
        return

    os.environ["HF_HUB_OFFLINE"] = "1"  # local dataset only; never reach the Hub
    import lerobot.scripts.lerobot_train as lerobot_train

    lerobot_train.update_last_checkpoint = _write_pointer
    sys.argv = ["lerobot-train", *train_args]
    lerobot_train.main()


if __name__ == "__main__":
    main()
