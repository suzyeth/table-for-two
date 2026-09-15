"""data/command_noise.py: recording executes a slowly wandering, noisy command but labels the clean one.

Review 2: the demos were nearly identical and never showed a recovery - each recorded target was
only 0.002-0.011 rad from the arm's position while the policy's own error is 0.008-0.014 rad, so
any drift put it in states it had never seen. With noise on the executed command (DART-style),
the arm drifts off the plan and the clean label shows the way back.
"""
import numpy as np

from data.command_noise import GRIPPER_INDICES, CommandNoise, noisy

SIGMA = 0.02


def noises(noise, steps=4000):
    return np.array([noise(np.zeros(12)) for _ in range(steps)])


def test_grippers_are_never_perturbed():
    offsets = noises(CommandNoise(SIGMA, seed=3))
    assert np.all(offsets[:, list(GRIPPER_INDICES)] == 0.0)


def test_arm_joints_get_noise_of_about_sigma_bounded_at_two_sigma():
    offsets = noises(CommandNoise(SIGMA, seed=3))
    arm = np.delete(offsets, list(GRIPPER_INDICES), axis=1)
    assert 0.5 * SIGMA < arm.std() < 1.3 * SIGMA
    assert np.max(np.abs(arm)) <= 2 * SIGMA + 1e-12
    assert abs(arm.mean()) < 0.3 * SIGMA


def test_the_noise_wanders_slowly_rather_than_jittering():
    offsets = noises(CommandNoise(SIGMA, seed=3))
    per_step = np.abs(np.diff(offsets, axis=0))
    assert per_step.max() <= 0.4 * SIGMA + 1e-12  # knots 0.5 s apart, at most 4 sigma between them


def test_the_same_seed_gives_the_same_noise_and_zero_sigma_none():
    assert np.array_equal(noises(CommandNoise(SIGMA, seed=7), 200), noises(CommandNoise(SIGMA, seed=7), 200))
    assert not np.array_equal(noises(CommandNoise(SIGMA, seed=7), 200), noises(CommandNoise(SIGMA, seed=8), 200))
    assert np.all(noises(CommandNoise(0.0, seed=7), 200) == 0.0)


def test_a_gain_scales_the_noise_per_joint():
    left_only = np.r_[np.ones(6), np.zeros(6)]
    offsets = noises(CommandNoise(SIGMA, seed=3, gain=lambda: left_only), 300)
    assert np.all(offsets[:, 6:] == 0.0) and np.any(offsets[:, :5] != 0.0)
    assert np.all(noises(CommandNoise(SIGMA, seed=3, gain=lambda: np.zeros(12)), 300) == 0.0)


def test_scaling_keeps_the_underlying_noise_stream_the_same():
    full = noises(CommandNoise(SIGMA, seed=3), 300)
    half = noises(CommandNoise(SIGMA, seed=3, gain=lambda: np.full(12, 0.5)), 300)
    assert np.allclose(half, 0.5 * full)


def test_noisy_perturbs_what_the_env_executes_and_restores_env_step():
    class Env:
        def __init__(self):
            self.executed = []

        def step(self, action):
            self.executed.append(np.asarray(action, dtype=float).copy())

    env, noise = Env(), CommandNoise(SIGMA, seed=1)
    clean = np.full(12, 0.3)
    with noisy(env, noise):
        env.step(clean)
    assert "step" not in vars(env)  # the class method is back
    assert not np.array_equal(env.executed[0], clean)
    assert np.array_equal(clean, np.full(12, 0.3))  # the caller's (label) action is untouched
