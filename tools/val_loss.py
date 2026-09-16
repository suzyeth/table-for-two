"""Validation loss of every saved checkpoint on the episodes held out from training.

policy/train.py holds out the last 5% of the episodes (``--dataset.eval_split``). A run started
before it also passed ``--eval_steps`` never scored them; this does it afterwards, from each
checkpoint's own train_config.json, so the held-out episodes are exactly the ones training left
out. Frames are taken evenly spaced over all held-out episodes (LeRobot's own in-training check
takes the first frames of each), without image augmentation, as LeRobot's eval split is built.

Run:  .venv\\Scripts\\python.exe -m tools.val_loss --output-dir outputs/act_contact_v3
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR_FILE = "policy_preprocessor.json"


def checkpoints(output_dir):
    """``pretrained_model`` directories of the numbered checkpoints, oldest first."""
    root = Path(output_dir) / "checkpoints"
    numbered = sorted((p for p in root.glob("[0-9]*") if p.is_dir()), key=lambda p: int(p.name))
    return [p / "pretrained_model" for p in numbered]


def evenly_spaced(length, count):
    """``count`` frame indices spread evenly over ``length`` frames (all of them if fewer)."""
    if count >= length:
        return list(range(length))
    return np.linspace(0, length - 1, count).round().astype(int).tolist()


def _device_overrides(pretrained, device):
    steps = json.loads((pretrained / PREPROCESSOR_FILE).read_text(encoding="utf-8")).get("steps", [])
    return {s["registry_name"]: {"device": device} for s in steps
            if s.get("registry_name") and "device" in s.get("config", {})}


def held_out_dataset(pretrained):
    import lerobot.policies.act.configuration_act  # noqa: F401  (registers policy type "act" for the parser)
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets

    cfg = TrainPipelineConfig.from_pretrained(pretrained)
    _, eval_dataset = make_train_eval_datasets(cfg)
    if eval_dataset is None:
        raise SystemExit(f"{pretrained} was trained without a held-out split (dataset.eval_split = 0)")
    return eval_dataset


def val_loss(pretrained, loader, camera_keys, device):
    """Mean ACT training loss (L1 + KL term) over ``loader`` for one checkpoint."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(pretrained)).to(device).eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=str(pretrained),
                                      preprocessor_overrides=_device_overrides(pretrained, device))
    total, frames = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            for key in camera_keys:
                if key in batch and batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].to(dtype=torch.float32) / 255.0
            size = len(next(iter(batch.values())))
            loss, _ = policy.forward(pre(batch))
            total += float(loss) * size
            frames += size
    return total / max(frames, 1)


def main():
    parser = argparse.ArgumentParser(description="Validation loss of every checkpoint on the held-out episodes.")
    parser.add_argument("--output-dir", type=Path, required=True, help="training output, e.g. outputs/act_contact_v3")
    parser.add_argument("--samples", type=int, default=3000, help="held-out frames scored per checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--only", nargs="*", help="checkpoint step names to score, e.g. 005000 (default: all)")
    parser.add_argument("--out", type=Path, help="JSON report (default: out/<output-dir name>_val_loss.json)")
    args = parser.parse_args()

    found = checkpoints(args.output_dir)
    if args.only:
        found = [p for p in found if p.parent.name in set(args.only)]
    if not found:
        raise SystemExit(f"no numbered checkpoints under {args.output_dir / 'checkpoints'}")
    dataset = held_out_dataset(found[-1])
    indices = evenly_spaced(len(dataset), args.samples)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)
    print(f"{len(dataset)} held-out frames in {len(dataset.episodes)} episodes; scoring {len(indices)}", flush=True)
    report = {"output_dir": str(args.output_dir), "held_out_episodes": [int(e) for e in dataset.episodes],
              "held_out_frames": len(dataset), "scored_frames": len(indices), "val_loss": {}}
    for pretrained in found:
        started = time.perf_counter()
        loss = val_loss(pretrained, loader, dataset.meta.camera_keys, args.device)
        report["val_loss"][pretrained.parent.name] = round(loss, 5)
        print(f"step {pretrained.parent.name}: val loss {loss:.4f} ({time.perf_counter() - started:.0f} s)", flush=True)
    best = min(report["val_loss"], key=report["val_loss"].get)
    report["best_checkpoint"] = best
    out = args.out or ROOT / "out" / f"{Path(args.output_dir).name}_val_loss.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"lowest validation loss at step {best}; wrote {out}")


if __name__ == "__main__":
    main()
