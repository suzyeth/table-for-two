"""tools/audit_policy.py: classifying releases as placed / gentle / dropped."""
import numpy as np

from tools.audit_policy import PolicyCommands, classify, summarize


def release(obj="mug", resting=False, height=None, speed=None, lost=False):
    event = {"object": obj, "resting_on_support_at_release": resting, "lost_while_squeezing": lost}
    if height is not None:
        event["drop_height_mm"] = height
    if speed is not None:
        event["drop_impact_speed_mps"] = speed
    return event


def test_resting_on_support_is_placed():
    assert classify(release(resting=True)) == "placed"


def test_small_slow_drop_is_gentle():
    assert classify(release(height=4.0, speed=0.08)) == "gentle"


def test_high_or_fast_drop_is_dropped():
    assert classify(release(height=28.0, speed=0.7)) == "dropped"
    assert classify(release(height=3.0, speed=0.3)) == "dropped"
    assert classify(release(height=12.0, speed=0.05)) == "dropped"


def test_lost_while_squeezing_or_still_falling_is_dropped():
    assert classify(release(lost=True, height=1.0, speed=0.01)) == "dropped"
    assert classify(release()) == "dropped"


def test_summary_counts_and_worst_values():
    events = [release("mug", resting=True), release("fork", height=2.0, speed=0.05),
              release("fork", height=30.0, speed=0.8), release("spoon", height=6.0, speed=0.2)]
    summary = summarize(events)
    assert summary["total"] == {"placed": 1, "gentle": 1, "dropped": 2, "tipped": 0, "knocked": 0}
    assert summary["per_object"]["fork"]["worst_drop_mm"] == 30.0
    assert summary["per_object"]["fork"]["worst_impact_mps"] == 0.8


def test_policy_commands_split_the_action_per_arm():
    commands = PolicyCommands()
    commands.update(np.arange(12))
    assert list(commands.skills["left_"].cmd) == [0, 1, 2, 3, 4, 5]
    assert commands.skills["right_"].cmd[5] == 11
