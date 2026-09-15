"""tools/audit_policy.py: tipping is judged on the tilt once the object has settled."""
import numpy as np

import tools.audit_contact as audit_contact
from tools.audit_policy import SETTLE_STEPS, KnockMonitor, classify


def resting(release_tilt, settled_tilt=None):
    event = {"object": "plate", "resting_on_support_at_release": True, "lost_while_squeezing": False,
             "tilt_at_release_deg": release_tilt}
    if settled_tilt is not None:
        event["settled_tilt_deg"] = settled_tilt
    return event


def test_a_plate_rocking_at_release_but_settling_flat_is_placed():
    assert classify(resting(14.3, settled_tilt=2.0)) == "placed"


def test_a_plate_upright_at_release_that_then_falls_over_is_tipped():
    assert classify(resting(3.0, settled_tilt=40.0)) == "tipped"


def test_without_a_settled_reading_the_release_tilt_is_used():
    assert classify(resting(14.3)) == "tipped"


class FakeWatcher:
    def __init__(self):
        self.events, self.hold, self.falling = [], {}, {}

    def touching_arms(self, obj):
        return set()


class FakeEnv:
    def __init__(self):
        self.tilt = 14.0

    def tilt_deg(self, obj):
        return self.tilt if obj == "plate" else 0.0

    def object_frame(self, obj):
        return np.zeros(3), np.eye(3)

    def holder(self, obj):
        return None


def test_monitor_records_the_tilt_one_settle_period_after_the_release(monkeypatch):
    monkeypatch.setattr(audit_contact, "velocity", lambda env, obj: np.zeros(3))
    monkeypatch.setattr(audit_contact, "supports", lambda env, obj: {1})
    env, watcher = FakeEnv(), FakeWatcher()
    monitor = KnockMonitor(env, watcher)
    event = resting(14.0)
    watcher.events.append(event)
    monitor.step("stage")  # the release is noticed on this step
    env.tilt = 2.5  # the plate stops rocking
    for _ in range(SETTLE_STEPS - 1):
        monitor.step("stage")
    assert "settled_tilt_deg" not in event
    monitor.step("stage")
    assert event["settled_tilt_deg"] == 2.5 and classify(event) == "placed"


def test_no_settled_tilt_for_an_object_picked_up_again_before_it_settled(monkeypatch):
    monkeypatch.setattr(audit_contact, "velocity", lambda env, obj: np.zeros(3))
    monkeypatch.setattr(audit_contact, "supports", lambda env, obj: {1})
    env, watcher = FakeEnv(), FakeWatcher()
    monitor = KnockMonitor(env, watcher)
    event = resting(14.0)
    watcher.events.append(event)
    monitor.step("stage")
    watcher.hold["plate"] = {}  # regrasped
    for _ in range(SETTLE_STEPS):
        monitor.step("stage")
    assert "settled_tilt_deg" not in event
