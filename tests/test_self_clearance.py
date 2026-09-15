"""sim/skills.py: an arm pose where the wrist camera mount runs into the arm's own shoulder is detected.

Measured on the hand-over: the receiver's hover pose put the right camera mount 5 mm into its
own shoulder; with the camera on the other side the same hover was 45 mm clear.
"""
import numpy as np
import pytest

from sim.env import DinnerTableEnv
from sim.skills import ArmSkills

JAMMED_HOVER = np.array([-0.68, -0.67, 0.70, 1.55, 1.99])
FLIPPED_HOVER = np.array([-0.67, -0.67, 0.69, 1.54, -1.14])


@pytest.fixture(scope="module")
def right_arm():
    env = DinnerTableEnv(obs_cameras=())
    env.reset(0)
    yield ArmSkills(env, "right_")
    env.close()


def test_camera_mount_pressed_into_the_shoulder_has_negative_clearance(right_arm):
    assert right_arm.self_clearance(JAMMED_HOVER) < 0.0


def test_the_same_hover_with_the_camera_on_the_other_side_is_clear(right_arm):
    assert right_arm.self_clearance(FLIPPED_HOVER) > 0.03


def test_home_pose_is_clear(right_arm):
    assert right_arm.self_clearance(right_arm.env.home["right_"][:5]) > 0.0
