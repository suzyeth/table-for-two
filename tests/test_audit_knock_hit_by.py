"""tools/audit_policy.py: a knock records which bodies were touching the prop (who hit it)."""
from types import SimpleNamespace

import numpy as np

import tools.audit_contact as audit_contact
from tools.audit_policy import KnockMonitor

NAMES = {1: "table", 7: "right_moving_jaw_so101_v1", 9: "bottle", 11: "water_3"}


class FakeModel:
    def body(self, i):
        return SimpleNamespace(name=NAMES[i])


class FakeEnv:
    def __init__(self):
        self.model = FakeModel()
        self.touching = {1, 7, 11}

    def _contact_bodies(self, obj):
        return set(self.touching)

    def object_frame(self, obj):
        return np.zeros(3), np.eye(3)

    def tilt_deg(self, obj):
        return 0.0

    def holder(self, obj):
        return None


class FakeWatcher:
    def __init__(self):
        self.events, self.hold, self.falling = [], {}, {}

    def touching_arms(self, obj):
        return set()


def test_knock_names_the_jaw_that_hit_the_mug_and_ignores_table_and_water(monkeypatch):
    speeds = iter([0.4, 0.3, 0.0])
    monkeypatch.setattr(audit_contact, "velocity", lambda env, obj: np.array([next(speeds, 0.0), 0.0, 0.0])
                        if obj == "mug" else np.zeros(3))
    monkeypatch.setattr(audit_contact, "supports", lambda env, obj: {1})
    env = FakeEnv()
    monitor = KnockMonitor(env, FakeWatcher())
    monitor.step("3:pour")  # knock starts: the jaw is pushing it
    env.touching = {1, 9}  # then the bottle brushes it too
    monitor.step("3:pour")
    monitor.step("3:pour")  # at rest again
    knocks = [e for e in monitor.events if e["object"] == "mug"]
    assert len(knocks) == 1
    assert knocks[0]["hit_by"] == ["bottle", "right_moving_jaw_so101_v1"]
    assert knocks[0]["stage"] == "3:pour"
