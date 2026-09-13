"""tools/audit_policy.py: a twitch from the water beads is 'jostled', a real hit or push is 'knocked'."""
from types import SimpleNamespace

import numpy as np

import tools.audit_contact as audit_contact
from tools.audit_policy import KnockMonitor, motion_kind, summarize

NAMES = {1: "table", 7: "right_moving_jaw_so101_v1", 11: "water_3"}


class FakeEnv:
    def __init__(self):
        self.model = SimpleNamespace(body=lambda i: SimpleNamespace(name=NAMES[i]))
        self.touching = {1, 11}
        self.pos = np.array([0.30, 0.10, 0.45])

    def _contact_bodies(self, obj):
        return set(self.touching)

    def object_frame(self, obj):
        return self.pos.copy(), np.eye(3)

    def tilt_deg(self, obj):
        return 0.0


class FakeWatcher:
    def __init__(self):
        self.events, self.hold, self.falling = [], {}, {}

    def touching_arms(self, obj):
        return set()


def run(monkeypatch, env, moves_mm=0.0):
    speeds = iter([0.2, 0.18, 0.0])
    monkeypatch.setattr(audit_contact, "velocity",
                        lambda e, obj: np.array([next(speeds, 0.0), 0, 0]) if obj == "mug" else np.zeros(3))
    monkeypatch.setattr(audit_contact, "supports", lambda e, obj: {1})
    monitor = KnockMonitor(env, FakeWatcher())
    monitor.step("3:pour")
    env.pos = env.pos + np.array([moves_mm / 1000.0, 0.0, 0.0])
    monitor.step("3:pour")
    monitor.step("3:pour")
    return [e for e in monitor.events if e["object"] == "mug"]


def test_water_only_twitch_without_displacement_is_jostled(monkeypatch):
    events = run(monkeypatch, FakeEnv(), moves_mm=0.3)
    assert len(events) == 1 and events[0]["kind"] == "jostled"
    assert events[0]["water_contact"] is True and events[0]["hit_by"] == []


def test_water_only_but_displaced_is_knocked(monkeypatch):
    events = run(monkeypatch, FakeEnv(), moves_mm=5.0)
    assert events[0]["kind"] == "knocked" and events[0]["moved_mm"] == 5.0


def test_touched_by_a_jaw_is_knocked_even_without_displacement(monkeypatch):
    env = FakeEnv()
    env.touching = {1, 7}
    events = run(monkeypatch, env, moves_mm=0.0)
    assert events[0]["kind"] == "knocked" and events[0]["hit_by"] == ["right_moving_jaw_so101_v1"]


def test_motion_kind_rules():
    assert motion_kind(set(), 0.5) == "jostled"
    assert motion_kind(set(), 2.5) == "knocked"
    assert motion_kind({"bottle"}, 0.0) == "knocked"
    assert motion_kind(set(), None) == "knocked"


def test_summary_counts_jostles_apart_from_knocks():
    events = [{"object": "mug", "kind": "jostled", "max_speed_mps": 0.2},
              {"object": "mug", "kind": "knocked", "max_speed_mps": 0.7}]
    summary = summarize(events)
    assert summary["total"]["jostled"] == 1 and summary["total"]["knocked"] == 1
    assert summary["per_object"]["mug"]["worst_impact_mps"] == 0.7
