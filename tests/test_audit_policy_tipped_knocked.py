"""tools/audit_policy.py: tipped-over set-downs and knocked (unheld, fast-moving) props."""
from tools.audit_policy import classify, event_kind, summarize


def release(obj="mug", resting=False, height=None, speed=None, tilt=None):
    event = {"object": obj, "resting_on_support_at_release": resting, "lost_while_squeezing": False,
             "tilt_at_release_deg": tilt}
    if height is not None:
        event["drop_height_mm"] = height
    if speed is not None:
        event["drop_impact_speed_mps"] = speed
    return event


def test_upright_set_down_is_placed_but_tipped_one_is_not():
    assert classify(release(resting=True, tilt=3.0)) == "placed"
    assert classify(release(resting=True, tilt=80.0)) == "tipped"


def test_gentle_drop_that_lands_on_its_side_is_tipped():
    assert classify(release(height=2.0, speed=0.05, tilt=45.0)) == "tipped"


def test_a_violent_drop_stays_dropped_even_if_tipped():
    assert classify(release(height=40.0, speed=0.9, tilt=60.0)) == "dropped"


def test_utensils_have_no_tilt_and_are_never_tipped():
    assert classify(release("fork", resting=True, tilt=None)) == "placed"


def test_knocked_events_keep_their_kind_and_are_counted_separately():
    knock = {"object": "fork", "kind": "knocked", "max_speed_mps": 0.9, "height_change_mm": 0.0}
    assert event_kind(knock) == "knocked"
    summary = summarize([release("mug", resting=True, tilt=1.0), knock])
    assert summary["total"]["knocked"] == 1 and summary["total"]["placed"] == 1
    assert summary["per_object"]["fork"]["worst_impact_mps"] == 0.9
    assert summary["per_object"]["fork"]["worst_drop_mm"] == 0.0
