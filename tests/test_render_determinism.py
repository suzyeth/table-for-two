"""scene/build_scene.py: the cameras render the same pixels for the same state, every time.

Review 3: the same checkpoint, seed and settings gave different rollouts. The physics and the
OpenVINO inference are bit-exact; the wrist-camera images were not - with 4x multisample
anti-aliasing the first render of a state differed from later ones and a render now and then came
out different again (1-5 pixels, one grey level). The closed loop grows that into a different
trajectory. Without multisampling every render of a state is identical. The flicker came and went
with the machine's load, so the repeat test below can pass even with multisampling on; the
setting itself is what the first test pins.
"""
import hashlib

import numpy as np
import pytest

from data.record import CAMERAS, IMAGE_SIZE, RECORD_EVERY
from sim.env import DinnerTableEnv

REPEATS = 10


@pytest.fixture(scope="module")
def env():
    env = DinnerTableEnv(obs_cameras=())
    env.reset(0)
    for _ in range(RECORD_EVERY):  # one control step, as in the rollout where the differences showed up
        env.step(env.data.ctrl.copy())
    yield env
    env.close()


def test_the_scene_renders_without_multisampling(env):
    assert env.model.vis.quality.offsamples == 0


@pytest.mark.parametrize("camera", CAMERAS)
def test_every_render_of_a_state_is_identical(env, camera):
    digests = set()
    for _ in range(REPEATS):
        for cam in CAMERAS:  # interleaved as in the rollout loop
            image = env.render(cam, IMAGE_SIZE)
            if cam == camera:
                digests.add(hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())
    assert len(digests) == 1
