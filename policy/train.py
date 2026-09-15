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
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POINTER_FILE = "last_checkpoint.txt"

CHUNK_SIZE = 20  # 2 s of actions at 10 Hz
N_ACTION_STEPS = 10  # re-plan every second
# LeRobot keeps the last ceil(n * EVAL_SPLIT) episodes per task out of training (15 of 300), so
# checkpoints can be compared on demos they never saw (review 2: there was no split at all).
EVAL_SPLIT = 0.05
# Image augmentation: LeRobot's colour and sharpness jitters, without its RandomAffine - shifting or
# turning a wrist camera image would change what the policy sees of its own fingers. LeRobot's parser
# cannot set one entry of this dict (``...tfs.affine.weight=0`` is rejected), so the whole dict is passed.
COLOUR_TRANSFORMS = {
    "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}},
    "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}},
    "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
    "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}},
    "sharpness": {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
}


def latest_pretrained(output_dir):
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
        f"--dataset.eval_split={EVAL_SPLIT}",
        "--dataset.image_transforms.enable=true",
        "--dataset.image_transforms.tfs=" + json.dumps(COLOUR_TRANSFORMS),
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
        f"--policy.use_amp={'true' if args.amp else 'false'}",
    ]


def build_parser():
    parser = argparse.ArgumentParser(description="Train ACT on the local dinner-table dataset.")
    # Both required: the old defaults pointed at the single-layout dataset and the v1 output directory.
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="LeRobot dataset from data.record or tools.record_parallel (e.g. .../merged)")
    parser.add_argument("--output-dir", type=Path, required=True, help="new checkpoint directory, e.g. outputs/act_contact_v3")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true",
                        help="automatic mixed precision (float16 autocast on CUDA); weights are still saved in float32")
    from policy.pour_oversampling import POUR_OVERSAMPLE

    parser.add_argument("--pour-oversample", type=int, default=POUR_OVERSAMPLE,
                        help="show every pour frame this many times per epoch (1 = off); see policy/pour_oversampling.py")
    parser.add_argument("--dry-run", action="store_true", help="print the arguments without training")
    return parser


def main():
    args = build_parser().parse_args()
    if not (args.dataset_root / "meta" / "info.json").exists():
        raise SystemExit(f"no LeRobot dataset at {args.dataset_root}; run: python -m tools.record_parallel --root ...")
    if args.output_dir.exists():
        raise SystemExit(f"{args.output_dir} exists; choose another --output-dir or remove it first")

    train_args = build_args(args)
    print("lerobot-train " + " ".join(train_args))
    if args.dry_run:
        return

    os.environ["HF_HUB_OFFLINE"] = "1"  # local dataset only; never reach the Hub
    import lerobot.scripts.lerobot_train as lerobot_train

    lerobot_train.update_last_checkpoint = _write_pointer
    from policy.pour_oversampling import install

    install(lerobot_train, args.dataset_root, args.pour_oversample)
    sys.argv = ["lerobot-train", *train_args]
    lerobot_train.main()


if __name__ == "__main__":
    main()
