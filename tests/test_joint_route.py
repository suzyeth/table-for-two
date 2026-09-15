"""sim/skills.py: a joint-space move (e.g. going home) is routed so the arm does not run into itself.

Measured on the plate carry: going home after setting the plate down, the straight joint-space
move swung wrist_roll 2.0 -> 1.3 rad with the shoulder low, and the left wrist camera mount
scraped the left shoulder (5.9 N on seed 0, 14.7 N on seed 4 - that demo was rejected).
"""
import numpy as np

from sim.skills import joint_route

START = np.array([0.5, -0.66, 0.62, 1.51, 2.02])
GOAL = np.array([0.06, -1.75, 0.5, 1.66, 2.74])


def camera_hits_shoulder(q):
    """Fake self clearance: the camera mount reaches the shoulder only with the wrist mid-roll while the arm is low."""
    return -0.001 if 1.0 < q[4] < 2.5 and q[1] > -1.0 else 0.02


def test_a_clear_straight_move_is_left_exactly_as_it_was():
    route, clearance = joint_route(START, GOAL, lambda q: 0.02, needed=0.005)
    assert len(route) == 1 and np.array_equal(route[0], GOAL) and clearance == 0.02


def test_a_straight_move_through_the_arm_rolls_the_wrist_last_up_at_the_goal_posture():
    start = np.array([0.5, -0.66, 0.62, 1.51, 0.5])  # rolling at the low start would hit; rolling up high is clear
    route, clearance = joint_route(start, GOAL, camera_hits_shoulder, needed=0.005)
    assert len(route) == 2 and np.array_equal(route[-1], GOAL)
    assert np.allclose(route[0][:4], GOAL[:4]) and route[0][4] == start[4]
    assert clearance >= 0.005


def test_rolling_first_is_used_when_only_that_is_clear():
    def only_roll_first_clear(q):
        # arm low with the roll still at the start value is blocked; roll already at the goal value is clear
        return -0.001 if q[1] > -1.0 and q[4] < 2.6 and q[1] < -0.7 else 0.02

    start = np.array([0.5, -0.66, 0.62, 1.51, 2.02])
    route, clearance = joint_route(start, GOAL, only_roll_first_clear, needed=0.005)
    assert len(route) == 2 and route[0][4] == GOAL[4] and np.allclose(route[0][:4], start[:4])
    assert clearance >= 0.005


def test_a_start_already_close_to_the_arm_does_not_hide_a_clear_route():
    """Review: every route was scored including the start pose, so a start within the margin tied all
    three and the straight move (the 5 rad wrist sweep) won."""
    start = np.array([0.5, -0.66, 0.62, 1.51, 2.02])

    def tight_start(q):
        if np.allclose(q, start):
            return 0.003
        if q[4] == start[4] or q[1] == GOAL[1]:  # on the roll-last route
            return 0.02
        return 0.004

    route, clearance = joint_route(start, GOAL, tight_start, needed=0.005)
    assert len(route) == 2 and route[0][4] == start[4] and clearance >= 0.005


def test_if_no_route_is_clear_the_clearest_is_returned_with_its_clearance():
    def always_tight(q):
        return -0.002 + 0.0001 * q[4]

    route, clearance = joint_route(START, GOAL, always_tight, needed=0.005)
    assert clearance < 0.005 and np.array_equal(route[-1], GOAL)
