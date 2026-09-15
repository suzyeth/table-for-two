"""scene/build_scene.py: props slide on the table like ceramic and glass on wood.

Review 1: the table and every prop had mu = 1.0, and MuJoCo takes the larger of two contacting
geoms' friction, so prop-on-table friction was always 1.0-1.3 (the per-seed scale only raised it)
and the beads met the bottle and mug walls at 1.0 as well; ceramic or glass on wood is ~0.4.
"""
import numpy as np
import pytest

from scene.build_scene import PAD_FRICTION, PROP_FRICTION
from sim.env import PROPS, DinnerTableEnv

SLIDING = 0.4


@pytest.fixture(scope="module")
def env():
    env = DinnerTableEnv(obs_cameras=())
    env.reset(0)  # seed 0: no per-seed friction scaling
    yield env
    env.close()


def test_the_table_and_every_prop_slide_like_ceramic_or_glass_on_wood(env):
    assert PROP_FRICTION[0] == pytest.approx(SLIDING)
    model = env.model
    assert model.geom_friction[model.geom("table_top").id, 0] == pytest.approx(SLIDING)
    for obj in PROPS:
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == env.body_ids[obj]]
        assert geoms and np.allclose(model.geom_friction[geoms, 0], SLIDING), obj


def test_the_finger_pads_still_grip_harder_than_props_slide():
    assert PAD_FRICTION[0] > PROP_FRICTION[0] * 1.3  # also above the highest per-seed prop scale

