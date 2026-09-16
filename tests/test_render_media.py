"""tools/render_media.py: stills and videos for people are anti-aliased; what a policy sees is not.

The scene renders without multisampling so a seed replays exactly (tests/test_render_determinism.py).
Media from the scripted pipeline never feeds a policy, so it keeps the smoother 4x-multisampled edges.
"""
from sim.env import DinnerTableEnv
from tools.render_media import MEDIA_OFFSAMPLES, media_env


def test_media_renders_are_anti_aliased():
    env = media_env()
    try:
        assert env.model.vis.quality.offsamples == MEDIA_OFFSAMPLES > 0
    finally:
        env.close()


def test_the_simulation_itself_still_renders_without_multisampling():
    env = media_env()
    plain = DinnerTableEnv(obs_cameras=())
    try:
        assert plain.model.vis.quality.offsamples == 0
    finally:
        env.close()
        plain.close()
