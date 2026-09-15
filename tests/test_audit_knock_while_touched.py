"""tools/audit_policy.py: a prop pushed or rocked by a hand that is not holding it is a knock.

The knock monitor used to skip any prop a finger body (fixed jaw / gripper housing, moving jaw)
was touching, so a gripper shoving the mug was never seen - only the rebound after the hand
had left, with an empty ``hit_by``. A mug rocking on its base edge also barely moves its
origin (at the base centre), so tilt must start a knock too, not only speed.
"""
from types import SimpleNamespace

import numpy as np

import tools.audit_contact as audit_contact
from tools.audit_policy import ROCK_TILT_DEG, KnockMonitor, motion_kind

NAMES = {1: "table", 7: "right_moving_jaw_so101_v1", 8: "right_gripper"}


class FakeEnv:
    def __init__(self, touching=(1, 8), holder=None):
        self.model = SimpleNamespace(body=lambda i: SimpleNamespace(name=NAMES[i]))
        self.touching = set(touching)
        self.pos = np.array([0.30, 0.10, 0.40])
        self.tilt = 0.0
        self.grasped_by = holder

    def _contact_bodies(self, obj):
        return set(self.touching) if obj == "mug" else {1}

    def object_frame(self, obj):
        return self.pos.copy(), np.eye(3)

    def tilt_deg(self, obj):
        return self.tilt if obj == "mug" else 0.0

    def holder(self, obj):
        return self.grasped_by if obj == "mug" else None


class FakeWatcher:
    """A hand touches the mug, but it is not held (it still stands on the table)."""

    def __init__(self):
        self.events, self.hold, self.falling = [], {}, {}

    def touching_arms(self, obj):
        return {"right_"} if obj == "mug" else set()


def monitor_with(monkeypatch, env, speeds):
    speeds = iter(speeds)
    monkeypatch.setattr(audit_contact, "velocity",
                        lambda e, obj: np.array([next(speeds, 0.0), 0, 0]) if obj == "mug" else np.zeros(3))
    monkeypatch.setattr(audit_contact, "supports", lambda e, obj: {1})
    return KnockMonitor(env, FakeWatcher())


def mug_events(monitor):
    return [e for e in monitor.events if e["object"] == "mug"]


def test_gripper_shoving_the_mug_while_touching_it_is_knocked_and_named(monkeypatch):
    env = FakeEnv(touching=(1, 8))
    monitor = monitor_with(monkeypatch, env, [0.4, 0.3, 0.0])
    monitor.step("3:pour")
    env.pos = env.pos + np.array([0.02, 0.0, 0.0])
    monitor.step("3:pour")
    monitor.step("3:pour")
    events = mug_events(monitor)
    assert len(events) == 1
    assert events[0]["kind"] == "knocked"
    assert events[0]["hit_by"] == ["right_gripper"]
    assert events[0]["moved_mm"] == 20.0


def test_mug_rocked_on_its_edge_without_fast_motion_is_knocked(monkeypatch):
    env = FakeEnv(touching=(1, 8))
    monitor = monitor_with(monkeypatch, env, [0.05] * 6)  # never above the knock speed; still at the end
    for tilt in (0.0, 0.0, 10.0, 23.0, 12.0, 0.5, 0.5):  # at rest first, so its resting tilt is known
        env.tilt = tilt
        monitor.step("3:pour")
    events = mug_events(monitor)
    assert len(events) == 1
    assert events[0]["kind"] == "knocked"
    assert events[0]["max_tilt_change_deg"] == 23.0
    assert events[0]["hit_by"] == ["right_gripper"]


def test_small_wobble_below_the_rock_threshold_is_not_a_knock(monkeypatch):
    env = FakeEnv(touching=(1,))
    monitor = monitor_with(monkeypatch, env, [0.0] * 10)
    for tilt in (0.0, ROCK_TILT_DEG * 0.5, 0.0):
        env.tilt = tilt
        monitor.step("3:pour")
    assert mug_events(monitor) == []


def test_mug_squeezed_between_one_hands_jaws_is_being_grasped_not_knocked(monkeypatch):
    env = FakeEnv(touching=(1, 7, 8), holder="right_")
    monitor = monitor_with(monkeypatch, env, [0.4, 0.3, 0.0])
    env.tilt = 5.0
    for _ in range(3):
        monitor.step("2:pick mug")
    assert mug_events(monitor) == []


def test_a_shove_that_ends_in_a_grasp_is_still_reported(monkeypatch):
    """Review: the jaw pushes the mug 1 cm and then closes on it - the knock used to be discarded."""
    env = FakeEnv(touching=(1, 8))
    monitor = monitor_with(monkeypatch, env, [0.4, 0.3, 0.3])
    monitor.step("2:pick mug")
    env.pos = env.pos + np.array([0.01, 0.0, 0.0])
    monitor.step("2:pick mug")
    env.grasped_by = "right_"
    monitor.step("2:pick mug")
    events = mug_events(monitor)
    assert len(events) == 1
    assert events[0]["kind"] == "knocked" and events[0]["ended_in_grasp"] is True
    assert events[0]["moved_mm"] == 10.0


def test_a_grasp_that_barely_nudges_the_object_is_not_a_knock(monkeypatch):
    env = FakeEnv(touching=(1, 7, 8))
    monitor = monitor_with(monkeypatch, env, [0.2, 0.0])
    monitor.step("2:pick mug")
    env.pos = env.pos + np.array([0.0005, 0.0, 0.0])
    env.grasped_by = "right_"
    monitor.step("2:pick mug")
    assert [e["kind"] for e in mug_events(monitor)] in ([], ["jostled"])


def test_a_prop_still_toppling_after_the_release_grace_is_not_knocked_by_its_own_fall(monkeypatch):
    """Review: the resting tilt was taken mid-fall at the end of the grace period, then the fall read as a knock."""
    env = FakeEnv(touching=(1,))
    monitor = monitor_with(monkeypatch, env, [0.05] * 5)  # slower than a knock, then at rest
    monitor.watcher.events.append({"object": "mug", "tilt_at_release_deg": None})
    tilts = [2.0 * i for i in range(1, 11)] + [30.0, 40.0, 50.0, 60.0, 70.0] + [70.0] * 5
    for tilt in tilts:
        env.tilt = tilt
        monitor.step("4:return")
    assert [e for e in mug_events(monitor) if e["kind"] == "knocked"] == []


def test_motion_kind_counts_a_large_tilt_as_knocked():
    assert motion_kind(set(), 0.1, tilt_change_deg=ROCK_TILT_DEG + 1) == "knocked"
    assert motion_kind(set(), 0.1, tilt_change_deg=ROCK_TILT_DEG - 1) == "jostled"
