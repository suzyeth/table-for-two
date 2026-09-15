"""data/record.py, data/record_dagger.py: recordings use free-space command noise and a wider spread by default.

Sweep (training seeds 100-109, demo gate): spread 1.5 alone kept 9/10; with 0.02 rad free-space
noise added it still kept 9/10 (the same single failure, a pour that spilled 4 beads), while
22% of the control steps carried noise (mean 0.02 rad while on). Uniform noise had kept 1/10.
"""
from data.record import RECORD_NOISE, RECORD_SPREAD, build_parser


ROOT = ["--root", "data/x"]  # required since a forgotten --root could overwrite the old dataset


def test_the_plain_recorder_defaults_to_the_swept_noise_and_spread():
    args = build_parser().parse_args(ROOT)
    assert (args.noise, args.spread) == (RECORD_NOISE, RECORD_SPREAD) == (0.02, 1.5)


def test_the_corrective_recorder_uses_the_same_defaults():
    from data.record_dagger import build_parser as dagger_parser

    args = dagger_parser().parse_args(["--policy", "p.xml", "--checkpoint", "ckpt", *ROOT])
    assert (args.noise, args.spread) == (RECORD_NOISE, RECORD_SPREAD)


def test_both_can_still_record_without_noise_or_spread():
    args = build_parser().parse_args(["--noise", "0", "--spread", "1", *ROOT])
    assert (args.noise, args.spread) == (0.0, 1.0)
