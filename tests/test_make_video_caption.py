"""tools/make_video.py: the demo caption says who executed it, including stages the script did."""
from tools.make_video import demo_caption


def log(executor, stages):
    return {"executor": executor, "seed": 0, "stages": stages}


def test_a_scripted_demo_is_captioned_as_scripted():
    assert demo_caption(log("scripted", [{"stage": "x", "policy_ok": None, "assisted": False}])) \
        == "3 · Execution by the scripted skills — seed 0"


def test_a_policy_demo_names_the_scripted_stages():
    stages = [{"stage": "both: bimanual place plate", "policy_ok": True, "scripted_by_choice": False, "assisted": False},
              {"stage": "right: pour mug", "policy_ok": False, "scripted_by_choice": True, "assisted": False}]
    text = demo_caption(log("policy", stages))
    assert "learned policy" in text and "scripted" in text and "seed 0" in text


def test_a_demo_the_policy_did_alone_says_so():
    stages = [{"stage": "a", "policy_ok": True, "scripted_by_choice": False, "assisted": False}]
    assert demo_caption(log("policy", stages)) == "3 · Execution by the learned policy — seed 0"


def test_footage_can_be_played_faster_and_says_so():
    from tools.make_video import speed_label, speed_up
    assert list(speed_up(iter(range(7)), 2)) == [0, 2, 4, 6]
    assert list(speed_up(iter(range(3)), 1)) == [0, 1, 2]
    assert speed_label("seed 0", 2) == "seed 0 (2× speed)" and speed_label("seed 0", 1) == "seed 0"
