"""Noise on the executed command while recording; the clean command stays the label (DART-style).

The scripted targets are exact, so plain demos never leave the planned path and never show a
correction: each recorded target was only 0.002-0.011 rad from the arm's position, while the
policy's own error is 0.008-0.014 rad, so any drift put it in states it had never seen (review 2).
Here the command the arm executes wanders slowly around the scripted one - Gaussian knots every
KNOT_SECONDS, linearly interpolated, clipped at CLIP_SIGMAS - while the dataset records the clean
target, which from the drifted state is the way back. The grippers are never perturbed.

Uniform noise broke the scripted demos even below the policy's own error (training seeds
100-109: 0.005 rad kept 7/10, 0.01 rad 1/10) - a few motions have only millimetres to spare - so
the recorders scale it per arm with ``free_space_gains``: full only in the middle of the big
joint-space moves (going home, moving from high up to a hover pose), zero near objects.
"""
import contextlib

import numpy as np

from sim.env import ARMS, CONTROL_HZ, JOINTS

KNOT_SECONDS = 0.5
CLIP_SIGMAS = 2.0
GRIPPER_INDICES = tuple(i * len(JOINTS) + JOINTS.index("gripper") for i in range(len(ARMS)))
NOISE_STREAM = 7919  # mixed into the seed so the noise does not reuse the scene randomisation's draws


class CommandNoise:
    """``noise(action)`` -> a new array: ``action`` plus this control step's offset (radians).

    ``gain`` (optional): a function returning a per-joint factor (12 values or a scalar) that
    scales the offset at each step, e.g. ``free_space_gains(executor)``.
    """

    def __init__(self, sigma, seed, knot_seconds=KNOT_SECONDS, control_hz=CONTROL_HZ, gain=None):
        self.sigma = float(sigma)
        self.gain = gain
        self.rng = np.random.default_rng([int(seed), NOISE_STREAM])
        self.knot_steps = max(1, int(round(knot_seconds * control_hz)))
        self.size = len(ARMS) * len(JOINTS)
        self._phase = 0
        self._from, self._to = self._knot(), self._knot()

    def _knot(self):
        limit = CLIP_SIGMAS * self.sigma
        knot = np.clip(self.rng.normal(0.0, self.sigma, self.size), -limit, limit)
        knot[list(GRIPPER_INDICES)] = 0.0
        return knot

    def __call__(self, action):
        offset = self._from + (self._to - self._from) * (self._phase / self.knot_steps)
        self._phase += 1
        if self._phase == self.knot_steps:
            self._phase, self._from, self._to = 0, self._to, self._knot()
        if self.gain is not None:
            offset = offset * np.asarray(self.gain(), dtype=float)
        return np.asarray(action, dtype=float) + offset


def free_space_gains(executor):
    """Gain function for ``CommandNoise``: each arm's current ``noise_gain`` (sim/skills.py), per joint."""
    return lambda: np.repeat([executor.skills[arm].noise_gain for arm in ARMS], len(JOINTS))


@contextlib.contextmanager
def noisy(env, noise):
    """Within the block ``env.step`` executes ``noise(action)``; the caller's action is left untouched.

    A wrapper already set on ``env`` (e.g. the demo gate's) stays underneath and is put back afterwards.
    """
    raw_step = env.step
    previous = vars(env).get("step")

    def step(action):
        return raw_step(noise(action))

    env.step = step
    try:
        yield env
    finally:
        if previous is None:
            del env.step
        else:
            env.step = previous
