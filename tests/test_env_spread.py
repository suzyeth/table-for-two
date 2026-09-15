"""sim/env.py: recordings can spread the props wider than the evaluation scenes.

Review 2: position jitter was 12 mm for the plate, mug and bottle and 4 mm for the utensils. The
evaluation seeds must keep their layouts, so the wider spread is a reset option (default 1.0).
"""
import numpy as np
import pytest

from sim.env import PROPS, XY_JITTER, DinnerTableEnv


@pytest.fixture(scope="module")
def env():
    env = DinnerTableEnv(obs_cameras=())
    yield env
    env.close()


def prop_offsets(env, seed, **kwargs):
    env.reset(0)
    nominal = {obj: env.data.qpos[env.free_qadr[obj]:env.free_qadr[obj] + 2].copy() for obj in PROPS}
    env.reset(seed, **kwargs)
    return {obj: env.data.qpos[env.free_qadr[obj]:env.free_qadr[obj] + 2] - nominal[obj] for obj in PROPS}


def test_the_default_spread_leaves_every_seeded_scene_exactly_as_it_was(env):
    env.reset(5)
    before = env.data.qpos.copy()
    env.reset(5, spread=1.0)
    assert np.array_equal(env.data.qpos, before)


TABLE_PROPS = ("plate", "mug", "bottle")


def test_a_wider_spread_moves_the_table_props_further_but_within_the_scaled_range(env):
    widest = {obj: 0.0 for obj in TABLE_PROPS}
    for seed in range(100, 140):
        offsets = prop_offsets(env, seed, spread=2.0)
        for obj in TABLE_PROPS:
            assert np.all(np.abs(offsets[obj]) <= 2.0 * XY_JITTER[obj] + 1e-9)
            widest[obj] = max(widest[obj], float(np.max(np.abs(offsets[obj]))))
    assert all(widest[obj] > XY_JITTER[obj] for obj in TABLE_PROPS)


def test_the_utensils_keep_their_range_because_the_drawer_leaves_only_6_mm_beside_them(env):
    for seed in (101, 117, 133):
        wide, normal = prop_offsets(env, seed, spread=2.0), prop_offsets(env, seed)
        for obj in ("spoon", "fork"):  # same draw; the settling physics differs by nanometres, not millimetres
            assert np.allclose(wide[obj], normal[obj], atol=1e-6)
