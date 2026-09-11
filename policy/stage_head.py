"""Stage-completion head: the policy decides for itself when a subtask is finished.

Without it, the evaluator advances the subtask token from simulator state (an
oracle). This small network looks at the same four cameras, joint state and the
current subtask token as the ACT policy and outputs the probability that the
subtask is complete. It is trained on the same demonstrations: the last
DONE_WINDOW frames of every stage are labelled "done" (the scripted skill has
finished and the arm is settling), everything else "not done". At rollout the
stage advances after the head says "done" for DONE_STREAK consecutive calls
(``policy/rollout.py --switch head``); the oracle is then used only to score.

Train:   python -m policy.stage_head train            (~10 min on an RTX 4060)
Export:  python -m policy.stage_head export           (OpenVINO IR FP32 + INT8)
Both write to models/stage_head/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.record import CAMERAS, DATA_ROOT, IMAGE_SIZE, REPO_ID, SUBTASK_VOCAB

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "models" / "stage_head"
DONE_WINDOW = 10  # frames (1 s at 10 Hz) before a stage boundary that count as "done"
DONE_STREAK = 5  # consecutive "done" calls (0.5 s) before the stage advances at rollout
HELD_OUT_EPISODES = 15
STATE_DIM = 12
IMAGE_KEYS = tuple(f"observation.images.{cam}" for cam in CAMERAS)


class StageHead(nn.Module):
    """Per-camera CNN trunk (shared weights) + joint state + subtask one-hot -> done logit."""

    def __init__(self, n_cameras=len(CAMERAS), n_stages=len(SUBTASK_VOCAB), feat=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, feat, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        per_cam = feat * 4
        self.head = nn.Sequential(
            nn.Linear(per_cam * n_cameras + STATE_DIM + n_stages, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state, onehot, *images):
        feats = [self.trunk(image) for image in images]
        return self.head(torch.cat([*feats, state, onehot], dim=1)).squeeze(1)


# ----------------------------------------------------------------------- data
def done_labels(dataset):
    """1 for frames within DONE_WINDOW of the end of their stage, else 0 (numpy, one per frame)."""
    table = dataset.hf_dataset
    episodes = np.asarray(table["episode_index"])
    onehot = np.stack([np.asarray(x) for x in table["observation.environment_state"]])
    stage = onehot.argmax(1)
    labels = np.zeros(len(episodes), dtype=np.float32)
    start = 0
    for i in range(1, len(episodes) + 1):
        boundary = i == len(episodes) or episodes[i] != episodes[start] or stage[i] != stage[start]
        if boundary:
            labels[max(start, i - DONE_WINDOW):i] = 1.0
            start = i
    return labels, episodes


def batch_tensors(dataset, indices, device):
    items = [dataset[int(i)] for i in indices]
    state = torch.stack([it["observation.state"] for it in items]).to(device)
    onehot = torch.stack([it["observation.environment_state"] for it in items]).to(device)
    images = [torch.stack([it[k] for it in items]).to(device) for k in IMAGE_KEYS]
    return state, onehot, images


def train(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LeRobotDataset(REPO_ID, root=args.dataset_root)
    labels, episodes = done_labels(dataset)
    held = np.unique(episodes)[-HELD_OUT_EPISODES:]
    train_idx = np.where(~np.isin(episodes, held))[0]
    val_idx = np.where(np.isin(episodes, held))[0]
    pos_weight = torch.tensor([(1 - labels[train_idx].mean()) / labels[train_idx].mean()], device=device)
    print(f"{len(train_idx)} train / {len(val_idx)} val frames, done fraction {labels.mean():.3f}, device {device}")

    model = StageHead().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(0)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        idx = rng.choice(train_idx, args.batch_size)
        state, onehot, images = batch_tensors(dataset, idx, device)
        target = torch.as_tensor(labels[idx], device=device)
        loss = loss_fn(model(state, onehot, *images), target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} ({time.perf_counter() - started:.0f} s)", flush=True)
    metrics = evaluate(model, dataset, labels, episodes, val_idx, device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "stage_head.pt")
    (OUT_DIR / "train_report.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps(metrics, indent=1))


@torch.no_grad()
def evaluate(model, dataset, labels, episodes, val_idx, device):
    """Frame accuracy on held-out episodes plus, per stage, how early/late the streak rule fires."""
    model.eval()
    probs = np.zeros(len(val_idx), dtype=np.float32)
    for k in range(0, len(val_idx), 256):
        idx = val_idx[k:k + 256]
        state, onehot, images = batch_tensors(dataset, idx, device)
        probs[k:k + 256] = torch.sigmoid(model(state, onehot, *images)).cpu().numpy()
    truth = labels[val_idx]
    pred = probs > 0.5
    tp = float(np.sum(pred & (truth > 0.5)))
    precision = tp / max(1.0, float(pred.sum()))
    recall = tp / max(1.0, float((truth > 0.5).sum()))
    # Streak rule timing: per (episode, stage) run, first frame with DONE_STREAK consecutive
    # "done" predictions vs the true first "done" frame (positive = fires late).
    stage = np.stack([np.asarray(x) for x in dataset.hf_dataset["observation.environment_state"]]).argmax(1)[val_idx]
    ep = episodes[val_idx]
    offsets, misses, early = [], 0, 0
    start = 0
    for i in range(1, len(val_idx) + 1):
        if i == len(val_idx) or ep[i] != ep[start] or stage[i] != stage[start]:
            run_pred, run_truth = pred[start:i], truth[start:i]
            true_first = int(np.argmax(run_truth > 0.5)) if run_truth.any() else len(run_truth)
            fired = None
            streak = 0
            for j, p in enumerate(run_pred):
                streak = streak + 1 if p else 0
                if streak >= DONE_STREAK:
                    fired = j
                    break
            if fired is None:
                misses += 1
            else:
                offsets.append(fired - true_first)
                early += fired < true_first - DONE_WINDOW
            start = i
    model.train()
    return {"val_frames": int(len(val_idx)), "precision": round(precision, 3), "recall": round(recall, 3),
            "stage_runs": len(offsets) + misses, "streak_never_fired": misses,
            "fired_more_than_1s_early": early,
            "fire_offset_frames_median": float(np.median(offsets)) if offsets else None,
            "fire_offset_frames_max": float(np.max(offsets)) if offsets else None}


# --------------------------------------------------------------------- export
def export(args):
    import nncf
    import openvino as ov
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    model = StageHead()
    model.load_state_dict(torch.load(OUT_DIR / "stage_head.pt", map_location="cpu"))
    model.eval()
    example = (torch.zeros(1, STATE_DIM), torch.zeros(1, len(SUBTASK_VOCAB)),
               *[torch.zeros(1, 3, *IMAGE_SIZE) for _ in CAMERAS])
    with torch.no_grad():
        fp32 = ov.convert_model(model, example_input=example)
    ov.save_model(fp32, OUT_DIR / "stage_head_fp32.xml", compress_to_fp16=False)
    dataset = LeRobotDataset(REPO_ID, root=args.dataset_root)
    idx = np.linspace(0, len(dataset) - 1, 200).astype(int)
    calib = []
    for i in idx:
        state, onehot, images = batch_tensors(dataset, [i], "cpu")
        calib.append([state.numpy(), onehot.numpy(), *[im.numpy() for im in images]])
    int8 = nncf.quantize(fp32, nncf.Dataset(calib), subset_size=len(calib))
    ov.save_model(int8, OUT_DIR / "stage_head_int8.xml")
    meta = {"image_keys": list(IMAGE_KEYS), "done_streak": DONE_STREAK,
            "input_shapes": [list(t.shape) for t in example]}
    for name in ("stage_head_fp32", "stage_head_int8"):
        (OUT_DIR / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"exported to {OUT_DIR}")


class OVStageHead:
    """OpenVINO runtime for the head: ``done(state, onehot, images)`` -> probability."""

    def __init__(self, xml_path, device="CPU"):
        import openvino as ov

        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise SystemExit(f"no stage head at {xml_path}; run: python -m policy.stage_head train && export")
        meta = json.loads(xml_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.image_keys = meta["image_keys"]
        self.streak_needed = meta["done_streak"]
        core = ov.Core()
        model = core.read_model(xml_path)
        model.reshape({port: ov.PartialShape(shape) for port, shape in zip(model.inputs, meta["input_shapes"])})
        self.compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
        self.request = self.compiled.create_infer_request()
        self.streak = 0
        self.infer_ms = []

    def reset(self):
        self.streak = 0

    def done(self, state, onehot, images):
        """Probability that the current subtask is complete; ``fired`` tracks the streak rule."""
        inputs = [np.asarray(state, np.float32)[None], np.asarray(onehot, np.float32)[None]]
        for key in self.image_keys:
            image = np.ascontiguousarray(images[key]).astype(np.float32) / 255.0
            inputs.append(image.transpose(2, 0, 1)[None])
        started = time.perf_counter()
        logit = float(np.asarray(self.request.infer(inputs)[self.compiled.output(0)]).reshape(-1)[0])
        self.infer_ms.append((time.perf_counter() - started) * 1000)
        prob = 1.0 / (1.0 + np.exp(-logit))
        self.streak = self.streak + 1 if prob > 0.5 else 0
        return prob

    @property
    def fired(self):
        return self.streak >= self.streak_needed


def main():
    parser = argparse.ArgumentParser(description="Train / export the stage-completion head.")
    parser.add_argument("command", choices=("train", "export"))
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    (train if args.command == "train" else export)(args)


if __name__ == "__main__":
    main()
