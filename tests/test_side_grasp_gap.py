"""sim/pour.py: before squeezing the bottle, the hand brings the fixed pad to ~1 mm from it.

Measured with real placement jitter: the side grasp tipped the bottle 4.6-8.5 deg in 6 of 20
seeds (every demo-gate reject). The moving jaw pushed the 40 g, 7 cm bottle ~6 mm before the
fixed pad caught it, 3-4.4 cm above its base. Sliding the hand first, so the fixed pad sat
1-2 mm off, left all 9 seeds tried untouched (no tilt before the squeeze, no knock after).
"""
import numpy as np

from sim.env import GRIPPER_SQUEEZE
from sim.pour import SIDE_GRASP_GAP, PourMixin


class FakeSideArm(PourMixin):
    """``side_pick_bottle`` with the motions recorded instead of executed."""

    def __init__(self, gap):
        self.gap, self.calls, self.cmd = gap, [], np.zeros(6)
        self.arm, self.warnings = "right_", []
        self.free_speed, self.squeeze_settle = 1.2, 8
        self.closing = np.array([0.0, 1.0, 0.0])
        self.plan = {"orient": {"approach": None, "lateral": np.array([0.0, 0.0, 1.0]), "point": np.zeros(3)},
                     "centre": np.array([0.3, -0.2, 0.43]), "pre": np.array([0.3, -0.2, 0.5]),
                     "q_pre": np.zeros(5), "sign": 1.0}

    def _plan_side_grasp(self):
        return self.plan

    def _fixed_pad_gap(self, obj):
        return self.gap

    def frame(self, q=None):
        return np.zeros(3), np.column_stack([self.closing, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def point_world(self, point, q=None):
        return self.plan["centre"].copy()

    def rise(self):
        self.calls.append(("rise", None))
        return iter(())

    def _to(self, q, grip=None, speed=None, *args, **kwargs):
        self.calls.append(("to", None))
        return iter(())

    def line(self, end, orient, steps=6, *args, **kwargs):
        self.calls.append(("line", np.asarray(end, dtype=float).copy()))
        return iter(())

    def grip(self, value, *args, **kwargs):
        self.calls.append(("grip", value))
        return iter(())

    def wait(self, steps):
        self.calls.append(("wait", None))
        return iter(())


def move_before_squeeze(arm):
    list(arm.side_pick_bottle())
    names = [name for name, _ in arm.calls]
    squeeze = names.index("grip")
    assert arm.calls[squeeze][1] == GRIPPER_SQUEEZE
    return arm.calls[squeeze - 1]


def test_the_fixed_pad_is_brought_to_the_gap_before_the_squeeze():
    arm = FakeSideArm(gap=0.006)
    name, target = move_before_squeeze(arm)
    assert name == "line"
    assert np.allclose(target, arm.plan["centre"] + arm.closing * (0.006 - SIDE_GRASP_GAP))


def test_no_extra_move_when_the_fixed_pad_is_already_that_close():
    arm = FakeSideArm(gap=SIDE_GRASP_GAP * 0.5)
    name, target = move_before_squeeze(arm)
    assert name == "line" and np.allclose(target, arm.plan["centre"])  # just the descent onto the bottle
    assert sum(name == "line" for name, _ in arm.calls) == 2  # the descent and the lift, nothing else
