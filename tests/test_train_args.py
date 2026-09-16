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


def test_the_held_out_episodes_are_scored_during_training():
    """The split alone only kept 14 episodes out of training: LeRobot computes their loss only with
    eval_steps > 0 (its default 0 is off), so the v3 run logged no validation loss at all."""
    from policy.train import EVAL_EVERY, EVAL_SAMPLES

    args = build_args(make(False))
    assert f"--eval_steps={EVAL_EVERY}" in args and f"--max_eval_samples={EVAL_SAMPLES}" in args
    assert 0 < EVAL_EVERY <= 5000 and EVAL_SAMPLES > 0


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
    assert cfg.eval_steps > 0 and cfg.max_eval_samples > 0
    assert set(cfg.dataset.image_transforms.tfs) == {"brightness", "contrast", "saturation", "hue", "sharpness"}


def test_pour_frames_are_oversampled_by_default():
    from policy.pour_oversampling import POUR_OVERSAMPLE
    from policy.train import build_parser

    args = build_parser().parse_args(["--dataset-root", "data/d", "--output-dir", "outputs/o"])
    assert args.pour_oversample == POUR_OVERSAMPLE == 2


def test_resuming_continues_a_run_from_its_last_checkpoint():
    """v3 stopped at 40k with its validation loss still falling, so the run has to be continued
    rather than restarted: LeRobot resumes from a checkpoint directory, keeping step, optimizer,
    scheduler and RNG (a local --config_path needs no `last` symlink, which Windows cannot make)."""
    args = SimpleNamespace(dataset_root=Path("data/x"), device="cuda", output_dir=Path("outputs/act_contact_v3"),
                           steps=80000, batch_size=32, num_workers=12, save_freq=5000, amp=True,
                           resume=Path("outputs/act_contact_v3/checkpoints/040000/pretrained_model"))
    built = build_args(args)
    assert "--resume=true" in built
    assert "--config_path=outputs/act_contact_v3/checkpoints/040000/pretrained_model" in built
    assert "--steps=80000" in built


def test_without_resume_no_checkpoint_is_referenced():
    assert not any(a.startswith("--config_path") or a == "--resume=true" for a in build_args(make(False)))


def test_core_settings_unchanged():
    args = build_args(make(True))
    assert "--policy.type=act" in args and "--steps=10" in args and "--policy.chunk_size=20" in args


def test_stage_starts_are_oversampled_by_default():
    from policy.pour_oversampling import START_FRAMES, START_OVERSAMPLE
    from policy.train import build_parser

    args = build_parser().parse_args(["--dataset-root", "data/d", "--output-dir", "outputs/o"])
    assert args.start_oversample == START_OVERSAMPLE > 1 and args.start_frames == START_FRAMES > 0
