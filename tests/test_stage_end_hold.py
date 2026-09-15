"""sim/task.py: every scripted stage ends with the arms held still, so a policy can learn to stop there.

Diagnosis (v2 policy, seeds 6 and 7): after a stage's check passed the policy never came to rest -
it went straight on into the next skill (the left hand moved the fork 9 cm, the right hand carried
the bottle 10 cm toward the mug), because in the demos the next stage began 3 control steps after
the last motion. The hold is recorded under the finished stage's label.
"""
import numpy as np
import pytest

from sim.env import CONTROL_HZ, DinnerTableEnv
from sim.task import STAGE_END_HOLD_S, Executor


@pytest.fixture(scope="module")
def env():
    env = DinnerTableEnv(obs_cameras=())
    yield env
    env.close()


def trailing_identical(actions):
    count = 1
    while count < len(actions) and np.array_equal(actions[-count - 1], actions[-1]):
        count += 1
    return count


def test_each_stage_ends_with_the_arms_still_for_the_hold_under_its_own_label(env):
    env.reset(0)
    executor = Executor(env)
    executor.reset()
    for arm in ("left_", "right_"):
        executor.skills[arm].cmd[:5] += 0.2  # so going home is a real move
    log = []
    plan = [[{"skill": "home", "arm": "left"}], [{"skill": "home", "arm": "right"}]]
    executor.run(plan, on_step=lambda action, label: log.append((label, np.asarray(action, dtype=float).copy())))
    hold = int(round(STAGE_END_HOLD_S * CONTROL_HZ))
    for label in ("0:home", "1:home"):
        actions = [a for stage, a in log if stage == label]
        assert trailing_identical(actions) >= hold, label


def test_the_hold_is_longer_than_the_evaluator_needs_to_see_a_policy_settle():
    from policy.rollout import RECORD_EVERY, SETTLE_STILL_STEPS

    assert STAGE_END_HOLD_S > SETTLE_STILL_STEPS * RECORD_EVERY / 20
