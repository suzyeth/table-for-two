"""data/record.py, sim/env.py: the policy sees joint velocities as well as positions.

Review 2: the policy saw one frame and joint positions only, so it could not time the pour's
tilt against the water flow; the pour was its only true skill gap (2-3/10 from a scripted start).
"""
import numpy as np
import pytest

from data.record import JOINT_NAMES, STATE_NAMES, dataset_features, policy_state
from sim.env import ARMS, DinnerTableEnv


class FakeEnv:
    def joint_state(self):
        return np.arange(12, dtype=float)

    def joint_velocity(self):
        return np.arange(12, dtype=float) + 100.0


def test_the_policy_state_is_positions_then_velocities():
    state = policy_state(FakeEnv())
    assert state.shape == (24,) and state.dtype == np.float32
    assert np.array_equal(state[:12], np.arange(12)) and np.array_equal(state[12:], np.arange(12) + 100)


def test_the_dataset_state_feature_names_all_24_values():
    feature = dataset_features()["observation.state"]
    assert feature["shape"] == (24,) and feature["names"] == STATE_NAMES
    assert STATE_NAMES[:12] == JOINT_NAMES and STATE_NAMES[12:] == [f"{name}.vel" for name in JOINT_NAMES]


@pytest.fixture(scope="module")
def env():
    env = DinnerTableEnv(obs_cameras=())
    yield env
    env.close()


def test_the_env_reports_the_arm_joint_velocities(env):
    env.reset(0)
    assert env.joint_velocity().shape == (12,)
    command = np.concatenate([env.home[arm] for arm in ARMS])
    command[0] += 0.3  # swing the left shoulder
    env.step(command)
    assert abs(env.joint_velocity()[0]) > 1e-3
