"""LeRobot training arguments built by policy/train.py."""
from pathlib import Path
from types import SimpleNamespace

from policy.train import build_args


def make(amp):
    return SimpleNamespace(dataset_root=Path("data/x"), device="cuda", output_dir=Path("outputs/y"), steps=10,
                           batch_size=32, num_workers=4, save_freq=5, amp=amp)


def test_amp_off_by_default_path():
    assert "--policy.use_amp=false" in build_args(make(False))


def test_amp_flag_turns_on_mixed_precision():
    args = build_args(make(True))
    assert "--policy.use_amp=true" in args
    assert "--policy.use_amp=false" not in args


def test_core_settings_unchanged():
    args = build_args(make(True))
    assert "--policy.type=act" in args and "--steps=10" in args and "--policy.chunk_size=20" in args
