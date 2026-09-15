"""sim/skills.py: command noise rides only on free-space moves, fading in and out.

Sweep (training seeds 100-109, demo gate): uniform command noise broke the scripted demos even
below the policy's own error - 0.005 rad kept 7/10, 0.01 rad 1/10 - because a few motions have
only millimetres to spare (the hand-over line-up, the spoon grasp, the mug set-down). So the
noise now only rides on the big joint-space moves (going home, and the move from high up to a
hover pose), fading in and out over FREE_SPACE_RAMP_STEPS so the command never jumps and the
hand arrives over an object with no noise left.
"""
import numpy as np
import pytest

from sim.env import DinnerTableEnv
from sim.skills import FREE_SPACE_RAMP_STEPS, ArmSkills, free_space_gain


def test_the_gain_ramps_in_and_out_and_is_full_in_between():
    steps = 40
    gains = [free_space_gain(i, steps) for i in range(1, steps + 1)]
    assert gains[0] == pytest.approx(1 / FREE_SPACE_RAMP_STEPS)
    assert max(gains) == 1.0 and gains[-1] == 0.0
    assert all(abs(b - a) <= 1 / FREE_SPACE_RAMP_STEPS + 1e-12 for a, b in zip(gains, gains[1:]))


def test_a_short_move_never_reaches_full_gain_but_still_ends_at_zero():
    gains = [free_space_gain(i, 6) for i in range(1, 7)]
    assert max(gains) < 1.0 and gains[-1] == 0.0


@pytest.fixture(scope="module")
def env():
    env = DinnerTableEnv(obs_cameras=())
    yield env
    env.close()


def gains_while_running(arm, generator):
    gains = []
    for _ in generator:
        gains.append(arm.noise_gain)
    return gains


def test_going_home_lets_the_noise_through_mid_move_and_none_after(env):
    env.reset(0)
    arm = ArmSkills(env, "left_")
    arm.cmd[4] -= np.sign(arm.cmd[4] or 1.0) * 2.0  # a long wrist-roll move home, TCP height unchanged
    gains = gains_while_running(arm, arm.home())
    assert max(gains) >= 0.9 and gains[-1] == 0.0
    assert arm.noise_gain == 0.0


def test_a_straight_cartesian_move_gets_no_noise(env):
    env.reset(0)
    arm = ArmSkills(env, "left_")
    start = arm.point_world(None)
    gains = gains_while_running(arm, arm.line(start + np.array([0.0, 0.0, -0.01]), {"approach": None}))
    assert gains and all(g == 0.0 for g in gains)
