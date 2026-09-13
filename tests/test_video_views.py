"""policy/video_views.py: tiling several camera views into one video frame."""
import numpy as np
import pytest

from policy.video_views import compose_views, tile_frames


def solid(value, h=4, w=6):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_four_tiles_make_a_two_by_two_grid_in_row_order():
    sheet = tile_frames([solid(10), solid(20), solid(30), solid(40)])
    assert sheet.shape == (8, 12, 3)
    assert sheet[0, 0, 0] == 10 and sheet[0, 11, 0] == 20
    assert sheet[7, 0, 0] == 30 and sheet[7, 11, 0] == 40


def test_three_tiles_leave_the_last_cell_black():
    sheet = tile_frames([solid(50), solid(60), solid(70)])
    assert sheet.shape == (8, 12, 3)
    assert sheet[7, 11].sum() == 0


def test_explicit_columns():
    assert tile_frames([solid(1), solid(2), solid(3)], cols=3).shape == (4, 18, 3)


def test_mismatched_sizes_and_labels_are_rejected():
    with pytest.raises(ValueError):
        tile_frames([solid(1), solid(2, h=5)])
    with pytest.raises(ValueError):
        tile_frames([solid(1), solid(2)], labels=["only one"])
    with pytest.raises(ValueError):
        tile_frames([])


class FakeEnv:
    def __init__(self):
        self.calls = []

    def render(self, camera, size=(480, 640)):
        self.calls.append((camera, size))
        return np.zeros((*size, 3), dtype=np.uint8)


def test_single_camera_is_rendered_at_full_size_untouched():
    env = FakeEnv()
    frame = compose_views(env, ["operator"])
    assert frame.shape == (480, 640, 3) and env.calls == [("operator", (480, 640))]


def test_several_cameras_render_the_same_step_as_small_tiles():
    env = FakeEnv()
    frame = compose_views(env, ["front_high", "overhead", "left_wrist_cam", "right_wrist_cam"], tile_size=(36, 48))
    assert frame.shape == (72, 96, 3)
    assert [c for c, _ in env.calls] == ["front_high", "overhead", "left_wrist_cam", "right_wrist_cam"]
