"""policy/*: no command silently falls back to the v1 model or the old dataset.

Review 2: models/policy/act_fp32.json references act_contact/040000, and DEFAULT_CKPT / OUTPUT_DIR
point to outputs/act_contact - so evaluating or exporting without explicit paths quietly used v1,
and training without --dataset-root used the old single-layout dataset.
"""
import pytest

from policy.export_openvino import build_parser as export_parser
from policy.rollout import build_parser as rollout_parser
from policy.train import build_parser as train_parser


@pytest.mark.parametrize("make", [rollout_parser, export_parser, train_parser])
def test_leaving_out_the_model_or_data_paths_is_an_error(make):
    with pytest.raises(SystemExit):
        make().parse_args([])


def test_recorders_and_the_stage_head_need_their_dataset_paths():
    """``data.record --overwrite`` without ``--root`` would have deleted the old dataset at its default path;
    the stage head trained on the old single-layout dataset by default."""
    from data.record import build_parser as record_parser
    from data.record_dagger import build_parser as dagger_parser
    from policy.stage_head import build_parser as head_parser

    for make, needed in ((record_parser, []), (dagger_parser, ["--policy", "p.xml", "--checkpoint", "ckpt"]),
                         (head_parser, ["train"])):
        with pytest.raises(SystemExit):
            make().parse_args(needed)
    assert record_parser().parse_args(["--root", "data/new"]).root.name == "new"
    assert head_parser().parse_args(["train", "--dataset-root", "data/d"]).dataset_root.name == "d"


def test_with_explicit_paths_every_command_parses():
    assert rollout_parser().parse_args(["--policy", "m.xml", "--checkpoint", "ckpt"]).checkpoint.name == "ckpt"
    export = export_parser().parse_args(["--checkpoint", "ckpt", "--dataset-root", "data/d", "--out-dir", "models/m"])
    assert export.checkpoint.name == "ckpt" and export.out_dir.name == "m"
    args = train_parser().parse_args(["--dataset-root", "data/d", "--output-dir", "outputs/o"])
    assert args.dataset_root.name == "d" and args.output_dir.name == "o"
