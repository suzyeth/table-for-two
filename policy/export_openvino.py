"""Export the trained ACT policy to OpenVINO IR in FP32, FP16 and INT8, and check accuracy.

  * FP32 - reference conversion of the PyTorch network;
  * FP16 - weight compression, the usual choice for Intel iGPU;
  * INT8 - post-training quantisation with NNCF, calibrated on real frames
    from the training dataset (preprocessed exactly as in training).

The accuracy check runs held-out dataset frames through PyTorch and each IR and
reports the action error in radians after un-normalisation, so optimisation
does not silently degrade the policy.

Run:  .venv\\Scripts\\python.exe -m policy.export_openvino --checkpoint outputs\\act_contact_v3\\checkpoints\\060000\\pretrained_model ^
        --dataset-root data\\dinner_table_contact_v3\\merged --out-dir models\\policy_v3
"""
import argparse
import json
from pathlib import Path

import nncf
import numpy as np
import openvino as ov
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from data.record import REPO_ID
from policy.ov_policy import batch_to_inputs, load_processors

CALIBRATION_FRAMES = 300
CHECK_FRAMES = 100


class ACTCore(torch.nn.Module):
    """ACT network without the LeRobot wrapper: normalised inputs -> normalised action chunk."""

    def __init__(self, policy):
        super().__init__()
        self.model = policy.model

    def forward(self, state, env_state, *images):
        """One image tensor per camera, in the checkpoint's ``image_features`` order."""
        batch = {OBS_STATE: state, OBS_ENV_STATE: env_state, OBS_IMAGES: list(images)}
        return self.model(batch)[0]


def sample_batches(pre, dataset, count, offset):
    """Pre-processed batches from ``count`` frames spread evenly over the dataset."""
    indices = np.linspace(offset, len(dataset) - 1, count).astype(int)
    return [pre(dataset[int(i)]) for i in indices]


def action_error(post, reference, candidate):
    ref = post(torch.from_numpy(reference)).cpu().numpy()
    cand = post(torch.from_numpy(candidate)).cpu().numpy()
    diff = np.abs(ref - cand)
    return float(diff.mean()), float(diff.max())


def build_parser():
    parser = argparse.ArgumentParser(description="Export ACT to OpenVINO FP32/FP16/INT8.")
    # All required: the old defaults were the v1 checkpoint, the old dataset (INT8 calibration) and
    # the v1 export directory, which an export without arguments silently overwrote.
    parser.add_argument("--checkpoint", type=Path, required=True, help="LeRobot pretrained_model directory")
    parser.add_argument("--dataset-root", type=Path, required=True, help="the dataset it was trained on (calibration)")
    parser.add_argument("--out-dir", type=Path, required=True, help="new export directory, e.g. models/policy_v3")
    parser.add_argument("--accuracy-control", action="store_true",
                        help="use NNCF accuracy-aware INT8 (reverts sensitive layers to float)")
    parser.add_argument("--max-drop", type=float, default=0.02,
                        help="allowed increase in mean normalised action error for --accuracy-control")
    return parser


def main():
    args = build_parser().parse_args()

    config, pre, post = load_processors(args.checkpoint)
    policy = ACTPolicy.from_pretrained(str(args.checkpoint))
    policy.to("cpu").eval()
    image_keys = list(policy.config.image_features)
    if not image_keys:
        raise SystemExit("checkpoint has no camera inputs")
    core_model = ACTCore(policy).eval()

    dataset = LeRobotDataset(REPO_ID, root=args.dataset_root)
    calibration = sample_batches(pre, dataset, CALIBRATION_FRAMES, offset=0)
    check = sample_batches(pre, dataset, CHECK_FRAMES, offset=7)
    example = tuple(torch.from_numpy(x) for x in batch_to_inputs(calibration[0], image_keys))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        fp32 = ov.convert_model(core_model, example_input=example)
    # IRs keep dynamic shapes on disk: pinning shapes before saving lets the
    # sinusoidal position-embedding subgraph fold into constants that FP16
    # compression then damages (measured 0.0002 -> 0.085 rad mean error).
    # Runtimes pin the batch-1 shapes from the metadata at load time instead.
    ov.save_model(fp32, args.out_dir / "act_fp32.xml", compress_to_fp16=False)
    ov.save_model(fp32, args.out_dir / "act_fp16.xml", compress_to_fp16=True)

    calib_inputs = [batch_to_inputs(b, image_keys) for b in calibration]
    if args.accuracy_control:
        # Keep quantising only while the mean normalised action error on held-out
        # frames stays within --max-drop of FP32; NNCF reverts the most
        # sensitive layers to floating point until it does.
        validation_items = []
        for batch in check:
            inputs = batch_to_inputs(batch, image_keys)
            with torch.no_grad():
                reference = core_model(*[torch.from_numpy(x) for x in inputs]).numpy()
            validation_items.append((inputs, reference))

        def negative_action_error(compiled, items):
            errors = [np.abs(np.asarray(compiled(inputs)[compiled.output(0)]) - reference).mean()
                      for inputs, reference in items]
            return -float(np.mean(errors))  # NNCF maximises the metric

        int8 = nncf.quantize_with_accuracy_control(
            fp32, nncf.Dataset(calib_inputs), nncf.Dataset(validation_items), negative_action_error,
            max_drop=args.max_drop, drop_type=nncf.DropType.ABSOLUTE, subset_size=len(calib_inputs),
            model_type=nncf.ModelType.TRANSFORMER)
    else:
        int8 = nncf.quantize(fp32, nncf.Dataset(calib_inputs), subset_size=len(calib_inputs),
                             model_type=nncf.ModelType.TRANSFORMER)
    ov.save_model(int8, args.out_dir / "act_int8.xml")

    meta = {"image_keys": image_keys, "chunk_size": policy.config.chunk_size,
            "n_action_steps": policy.config.n_action_steps, "checkpoint": str(args.checkpoint),
            "input_shapes": [list(tensor.shape) for tensor in example]}
    for name in ("act_fp32", "act_fp16", "act_int8"):
        (args.out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    # Accuracy: PyTorch vs each IR on held-out frames, compared after un-normalisation.
    ov_core = ov.Core()
    report = {}
    for name in ("act_fp32", "act_fp16", "act_int8"):
        compiled = ov_core.compile_model(ov_core.read_model(args.out_dir / f"{name}.xml"), "CPU")
        means, maxes = [], []
        for batch in check:
            inputs = batch_to_inputs(batch, image_keys)
            with torch.no_grad():
                reference = core_model(*[torch.from_numpy(x) for x in inputs]).numpy()
            candidate = np.asarray(compiled(inputs)[compiled.output(0)])
            mean_err, max_err = action_error(post, reference, candidate)
            means.append(mean_err)
            maxes.append(max_err)
        report[name] = {"mean_abs_action_error_rad": round(float(np.mean(means)), 5),
                        "max_abs_action_error_rad": round(float(np.max(maxes)), 5),
                        "size_mb": round((args.out_dir / f"{name}.bin").stat().st_size / 1e6, 1)}
    (args.out_dir / "export_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
