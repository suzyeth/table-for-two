"""policy/rollout.py: a stage ends where the demonstrations end it - once the arms have come to rest.

Review 2: evaluation switched to the next stage the moment the success check passed, while in
the demos a stage ends only after the script has let go and gone home, so every chained stage
started from a state the demos never show (bottle return 10/10 from a scripted start, 6/10
chained). Now the policy keeps running the same stage after the check passes until its
commands have been still for SETTLE_STILL_STEPS (or SETTLE_LIMIT_STEPS have passed).
"""
import numpy as np
import pytest

import policy.rollout as rollout
from policy.rollout import SETTLE_LIMIT_STEPS, SETTLE_STILL_STEPS, run_stage_policy

TOTAL_STEPS = int(rollout.STAGE_TIMEOUT_S * 20 / rollout.RECORD_EVERY)


class FakeEnv:
    def __init__(self):
        self.steps, self.held = 0, 0.0

    def render(self, camera, size):
        return np.zeros((*size, 3), dtype=np.uint8)

    def joint_state(self):
        return np.zeros(12)

    def step(self, action):
        self.steps += 1

    def hold(self, seconds):
        self.held += seconds


class MovingThenStillPolicy:
    """Commands change by 0.05 rad per call for the first ``moving`` calls, then stay put."""

    def __init__(self, moving):
        self.moving, self.calls, self.onehots = moving, 0, []

    def reset(self):
        pass

    def select_action(self, state, onehot, images):
        self.calls += 1
        self.onehots.append(np.asarray(onehot).copy())
        return np.full(12, 0.05 * min(self.calls, self.moving))


def run(monkeypatch, policy, done, settle=True):
    monkeypatch.setattr(rollout, "stage_done", lambda env, stage, elapsed: done(policy.calls))
    stage = [{"skill": "pick_place", "arm": "right", "object": "mug"}]
    if settle is None:
        return run_stage_policy(FakeEnv(), policy, stage, 1)
    return run_stage_policy(FakeEnv(), policy, stage, 1, settle=settle)


def test_by_default_the_stage_switches_as_soon_as_the_check_passes(monkeypatch):
    """Diagnosis: the v2 policy never comes to rest after a stage (its demos never did either) and runs
    straight on into the next skill, so waiting for it only did harm (pour 6/10 -> 3/10). Waiting is
    for policies trained on demos that end every stage with a still hold (sim/task.py)."""
    policy = MovingThenStillPolicy(moving=20)
    assert run(monkeypatch, policy, done=lambda calls: calls >= 3, settle=None) is True
    assert policy.calls == 3


def test_after_the_check_passes_the_same_stage_runs_on_until_the_arms_are_still(monkeypatch):
    policy = MovingThenStillPolicy(moving=20)  # e.g. letting go and going home after the mug is placed
    assert run(monkeypatch, policy, done=lambda calls: calls >= 3) is True
    assert policy.calls == 20 + SETTLE_STILL_STEPS
    assert all(np.array_equal(o, policy.onehots[0]) for o in policy.onehots)  # same subtask throughout


def test_arms_that_never_settle_end_the_stage_after_the_settle_limit(monkeypatch):
    policy = MovingThenStillPolicy(moving=10 ** 6)
    assert run(monkeypatch, policy, done=lambda calls: calls >= 3) is True
    assert policy.calls == 3 + SETTLE_LIMIT_STEPS


def test_a_check_that_no_longer_holds_once_settled_is_not_a_success(monkeypatch):
    policy = MovingThenStillPolicy(moving=1)
    assert run(monkeypatch, policy, done=lambda calls: 3 <= calls <= 4) is False
    assert policy.calls == TOTAL_STEPS


def test_a_stage_that_succeeds_at_the_very_end_still_counts(monkeypatch):
    policy = MovingThenStillPolicy(moving=10 ** 6)
    assert run(monkeypatch, policy, done=lambda calls: calls >= TOTAL_STEPS) is True


class ScheduledPolicy:
    """Commands given by ``schedule(call number)``."""

    def __init__(self, schedule):
        self.schedule, self.calls = schedule, 0

    def reset(self):
        pass

    def select_action(self, state, onehot, images):
        self.calls += 1
        return np.full(12, self.schedule(self.calls))


def test_a_pause_under_way_when_the_check_passes_does_not_end_the_stage_early(monkeypatch):
    """Review: the pour holds the tilted bottle still; if 'poured' came true during that hold the
    stage ended before the bottle was straightened. Stillness only counts from the check on."""
    unwind_end = 30 + SETTLE_STILL_STEPS // 2

    def hold_then_unwind(call):  # still until call 30 (the hold), then 0.05 rad per call until the unwind ends
        return 0.0 if call <= 30 else 0.05 * (min(call, unwind_end) - 30)

    policy = ScheduledPolicy(hold_then_unwind)
    assert run(monkeypatch, policy, done=lambda calls: calls >= 28) is True
    assert policy.calls == unwind_end + SETTLE_STILL_STEPS


def test_settling_takes_longer_than_any_pause_inside_a_demonstrated_stage():
    from sim.env import CONTROL_HZ
    from sim.pour import POUR_HOLD_STEPS

    longest_demo_pause_s = (8 + POUR_HOLD_STEPS) / CONTROL_HZ  # the pour: wait(8) at full tilt, then the hold
    assert SETTLE_STILL_STEPS * rollout.RECORD_EVERY / 20 > longest_demo_pause_s
