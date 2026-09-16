"""demo.py: the learned-policy demo names its model and checkpoint and uses temporal ensembling.

It imported policy.export_openvino.DEFAULT_CKPT, removed when the v1 defaults went, so
`demo.py --executor policy` stopped at the import; its --policy default was still v1's INT8 IR.
"""
import pytest

from demo import build_parser


def test_the_scripted_demo_needs_no_model():
    args = build_parser().parse_args(["--seed", "3"])
    assert args.executor == "scripted" and args.policy is None and args.checkpoint is None


def test_the_policy_demo_needs_its_model_and_checkpoint():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--executor", "policy"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--executor", "policy", "--policy", "m.xml"])
    args = build_parser().parse_args(["--executor", "policy", "--policy", "m.xml", "--checkpoint", "ckpt"])
    assert args.checkpoint.name == "ckpt" and args.ensemble == 0.01


def test_only_demonstrated_stages_go_to_the_policy():
    """A spoken "carry the plate, then pour" plans the bottle lift without the fork - a stage the policy
    has no subtask token for; the demo stopped there with a ValueError."""
    from demo import policy_can_run, takeover_note
    from sim.task import DEFAULT_PLAN

    assert all(policy_can_run(stage) for stage in DEFAULT_PLAN)
    assert not policy_can_run([{"skill": "pick_lift", "arm": "right", "object": "bottle"}])
    assert takeover_note(True) == "scripted takeover"
    assert "not a demonstrated stage" in takeover_note(False)


def test_steps_the_policy_knows_only_as_one_parallel_stage_are_merged():
    """The planner wrote "lay the fork" and "lift the bottle" as two stages; the policy was trained on them
    as one two-arm stage (10/10 chained), so the demo packs them back together before executing."""
    from demo import pack_for_policy
    from sim.task import DEFAULT_PLAN

    planned = [[{"skill": "bimanual_place", "object": "plate"}],
               [{"skill": "open_drawer", "arm": "left"}, {"skill": "pick_place", "arm": "right", "object": "mug"}],
               [{"skill": "pick_place", "arm": "left", "object": "fork"}],
               [{"skill": "pick_lift", "arm": "right", "object": "bottle"}],
               [{"skill": "pour", "arm": "right", "into": "mug"}]]
    packed = pack_for_policy(planned)
    assert len(packed) == 4 and packed[2] == planned[2] + planned[3]
    assert packed[:2] == planned[:2] and packed[3] == planned[4]
    assert pack_for_policy(DEFAULT_PLAN) == [list(stage) for stage in DEFAULT_PLAN]
    assert planned[2] == [{"skill": "pick_place", "arm": "left", "object": "fork"}]  # the input is left as it was


def test_stages_known_to_fail_can_be_given_to_the_script_from_the_start():
    """A policy attempt that fails leaves the scene messy (a knocked bottle), and the scripted takeover
    then fails too; for the video those stages can be run by the script from the start, captioned so."""
    from demo import run_by_script

    args = build_parser().parse_args(["--executor", "policy", "--policy", "m.xml", "--checkpoint", "c",
                                      "--scripted-stages", "pour", "return", "handoff"])
    assert args.scripted_stages == ["pour", "return", "handoff"]
    assert build_parser().parse_args(["--seed", "1"]).scripted_stages == []
    pour = [{"skill": "pour", "arm": "right", "into": "mug"}]
    fork_and_bottle = [{"skill": "pick_place", "arm": "left", "object": "fork"},
                       {"skill": "pick_lift", "arm": "right", "object": "bottle"}]
    assert run_by_script(pour, args.scripted_stages) and not run_by_script(fork_and_bottle, args.scripted_stages)
