"""LeRobot training arguments built by policy/train.py."""
import json
from pathlib import Path
from types import SimpleNamespace

from policy.train import EVAL_SPLIT, build_args


def make(amp):
    return SimpleNamespace(dataset_root=Path("data/x"), device="cuda", output_dir=Path("outputs/y"), steps=10,
                           batch_size=32, num_workers=4, save_freq=5, amp=amp)


def test_amp_off_by_default_path():
    assert "--policy.use_amp=false" in build_args(make(False))


def test_amp_flag_turns_on_mixed_precision():
    args = build_args(make(True))
    assert "--policy.use_amp=true" in args
    assert "--policy.use_amp=false" not in args


def test_a_few_episodes_are_held_out_for_validation():
    """Review 2: eval_split was 0.0, so there was no basis for choosing a checkpoint."""
    assert f"--dataset.eval_split={EVAL_SPLIT}" in build_args(make(False))
    assert 0.0 < EVAL_SPLIT <= 0.1


def test_colour_augmentation_is_on_without_shifting_or_turning_the_images():
    """Review 2: image augmentation was off. Colour and sharpness only: a shift or turn of a wrist
    camera image would change what the policy sees of its own fingers."""
    args = build_args(make(False))
    assert "--dataset.image_transforms.enable=true" in args
    transforms = json.loads(next(a for a in args if a.startswith("--dataset.image_transforms.tfs=")).split("=", 1)[1])
    assert set(transforms) == {"brightness", "contrast", "saturation", "hue", "sharpness"}


def test_lerobot_parses_every_training_argument():
    """A first version passed ``...tfs.affine.weight=0``, which LeRobot's parser rejects at the start of
    training - only a real parse catches that."""
    import draccus
    import lerobot.policies.act.configuration_act  # noqa: F401  (registers --policy.type=act, as lerobot_train does)
    from lerobot.configs.train import TrainPipelineConfig

    args = build_args(SimpleNamespace(dataset_root=Path("data/x"), device="cpu", output_dir=Path("outputs/y"),
                                      steps=10, batch_size=32, num_workers=0, save_freq=5, amp=False))
    cfg = draccus.parse(config_class=TrainPipelineConfig, args=args)
    assert cfg.dataset.eval_split == EVAL_SPLIT and cfg.dataset.image_transforms.enable
    assert set(cfg.dataset.image_transforms.tfs) == {"brightness", "contrast", "saturation", "hue", "sharpness"}


def test_pour_frames_are_oversampled_by_default():
    from policy.pour_oversampling import POUR_OVERSAMPLE
    from policy.train import build_parser

    args = build_parser().parse_args(["--dataset-root", "data/d", "--output-dir", "outputs/o"])
    assert args.pour_oversample == POUR_OVERSAMPLE == 2


def test_core_settings_unchanged():
    args = build_args(make(True))
    assert "--policy.type=act" in args and "--steps=10" in args and "--policy.chunk_size=20" in args
