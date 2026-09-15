"""sim/env.py: a seeded scene really is randomised - and the arms still start at home.

Found while adding the placement spread: every seed put every prop exactly where seed 0 does
(plate at (50, 0) mm, mug at (0, -150) mm, bottle at (120, -220) mm for seeds 0, 7, 100, 101):
the jitter was written into the state and then overwritten when the mass constants were
refreshed, so only mass, friction and lighting ever varied between scenes.
"""
import numpy as np
import pytest

from sim.env import ARMS, PROPS, UTENSILS, XY_JITTER, DinnerTableEnv

SEEDS = range(100, 110)


@pytest.fixture(scope="module")
def env():
    env = DinnerTableEnv(obs_cameras=())
    yield env
    env.close()


@pytest.fixture(scope="module")
def nominal(env):
    env.reset(0)
    return {obj: env.object_frame(obj) for obj in PROPS}


def test_seeded_scenes_place_the_props_off_nominal_and_within_the_jitter(env, nominal):
    moved = {obj: 0.0 for obj in PROPS}
    for seed in SEEDS:
        env.reset(seed)
        for obj in PROPS:
            offset = env.object_frame(obj)[0][:2] - nominal[obj][0][:2]
            assert np.all(np.abs(offset) <= XY_JITTER[obj] + 0.002), (seed, obj, offset)  # + settling
            moved[obj] = max(moved[obj], float(np.max(np.abs(offset))))
    assert all(moved[obj] > 0.25 * XY_JITTER[obj] for obj in PROPS), moved


def test_seeded_scenes_turn_the_utensils(env, nominal):
    headings = {obj: set() for obj in UTENSILS}
    for seed in SEEDS:
        env.reset(seed)
        for obj in UTENSILS:
            x_axis = env.object_frame(obj)[1][:, 0]
            headings[obj].add(round(float(np.degrees(np.arctan2(x_axis[1], x_axis[0]))), 1))
    assert all(len(h) > 5 for h in headings.values()), headings


def test_the_arms_start_at_home_in_every_scene(env):
    for seed in (0, 101):
        env.reset(seed)
        for arm in ARMS:
            assert np.allclose(env.arm_qpos(arm)[:5], env.home[arm][:5], atol=0.02), (seed, arm)
