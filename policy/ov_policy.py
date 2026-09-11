"""Run the exported ACT policy with OpenVINO.

The OpenVINO IR holds only the ACT network (normalised inputs -> normalised
action chunk). Normalisation, batching and un-normalisation stay in LeRobot's
own processor pipelines saved next to the checkpoint, so inference sees
exactly the preprocessing used in training. Every processor step that has a
``device`` setting is pinned to CPU, which lets the policy run on machines
without CUDA (e.g. an Intel Core Ultra laptop).
"""
import json
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE

PREPROCESSOR_FILE = "policy_preprocessor.json"
POSTPROCESSOR_FILE = "policy_postprocessor.json"


def _cpu_overrides(pretrained_dir, filename):
    """Override every step that carries a ``device`` field so it runs on CPU."""
    path = Path(pretrained_dir) / filename
    if not path.exists():
        return {}
    overrides = {}
    for step in json.loads(path.read_text(encoding="utf-8")).get("steps", []):
        name = step.get("registry_name")
        if name and "device" in step.get("config", {}):
            overrides[name] = {"device": "cpu"}
    return overrides


def load_processors(pretrained_dir):
    """Return (config, preprocessor, postprocessor) for a trained checkpoint, pinned to CPU."""
    config = PreTrainedConfig.from_pretrained(str(pretrained_dir))
    config.device = "cpu"
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides=_cpu_overrides(pretrained_dir, PREPROCESSOR_FILE),
        postprocessor_overrides=_cpu_overrides(pretrained_dir, POSTPROCESSOR_FILE),
    )
    return config, pre, post


def observation_to_item(state, env_state, images):
    """Build one un-batched LeRobot frame from env observations.

    ``images`` maps feature keys (e.g. ``observation.images.overhead``) to HxWx3 uint8 arrays.
    """
    item = {
        OBS_STATE: torch.as_tensor(np.asarray(state, dtype=np.float32)),
        OBS_ENV_STATE: torch.as_tensor(np.asarray(env_state, dtype=np.float32)),
    }
    for key, image in images.items():
        item[key] = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    return item


def batch_to_inputs(batch, image_keys):
    """Order a pre-processed batch as the IR's positional inputs."""
    tensors = [batch[OBS_STATE], batch[OBS_ENV_STATE], *[batch[k] for k in image_keys]]
    return [t.detach().cpu().numpy().astype(np.float32) for t in tensors]


class OVActPolicy:
    """ACT policy on an OpenVINO device with an action queue (re-plans every n_action_steps)."""

    def __init__(self, xml_path, pretrained_dir, device="CPU"):
        xml_path = Path(xml_path)
        meta_path = xml_path.with_suffix(".json")
        if not xml_path.exists() or not meta_path.exists():
            raise SystemExit(f"no OpenVINO policy at {xml_path}: download the release assets into models/policy/ "
                             "or run policy/train.py then policy/export_openvino.py (see README, 'Learned policy')")
        if not Path(pretrained_dir).exists():
            raise SystemExit(f"no checkpoint at {pretrained_dir}: download the release assets into outputs/ or train")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.image_keys = meta["image_keys"]
        self.n_action_steps = meta["n_action_steps"]
        self.device = device
        _, self.pre, self.post = load_processors(pretrained_dir)
        core = ov.Core()
        model = core.read_model(xml_path)
        if meta.get("input_shapes"):  # pin batch-1 shapes at load time (see export_openvino.py)
            model.reshape({port: ov.PartialShape(shape) for port, shape in zip(model.inputs, meta["input_shapes"])})
        started = time.perf_counter()
        self.compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
        self.compile_s = time.perf_counter() - started
        self.request = self.compiled.create_infer_request()
        self.queue = []
        self.infer_ms = []

    def reset(self):
        self.queue.clear()

    def infer_chunk(self, state, env_state, images):
        """Return the un-normalised action chunk, shape (chunk_size, 12)."""
        batch = self.pre(observation_to_item(state, env_state, images))
        inputs = batch_to_inputs(batch, self.image_keys)
        started = time.perf_counter()
        output = self.request.infer(inputs)[self.compiled.output(0)]
        self.infer_ms.append((time.perf_counter() - started) * 1000)
        action = self.post(torch.from_numpy(np.asarray(output)))
        return np.asarray(action.detach().cpu().numpy()).reshape(-1, 12)

    def select_action(self, state, env_state, images):
        if not self.queue:
            self.queue = list(self.infer_chunk(state, env_state, images)[:self.n_action_steps])
        return self.queue.pop(0)
