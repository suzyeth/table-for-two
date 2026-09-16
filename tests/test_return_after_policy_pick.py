"""sim/pour.py: the scripted return also works when the learned policy picked the bottle up.

``return_bottle`` used to do nothing unless the scripted ``side_pick_bottle`` had run, so in a
policy demo with a scripted return the bottle stayed in the hand (seeds 0-3: all 7 but
"bottle returned"). The side grasp is now read off the hand holding it.
"""
import numpy as np

from sim.pour import PourMixin


class HoldingArm(PourMixin):
    """A right hand holding the bottle with its jaw-width (+y) axis pointing ``lateral_z``."""

    def __init__(self, lateral_z, holding=True):
        self.arm, self.warnings, self.calls = "right_", [], []
        self.free_speed, self.release_backoff = 1.2, 0.004
        self.lateral_z, self.holding = lateral_z, holding

    def holds(self, obj):
        return self.holding

    def frame(self, q=None):
        return np.zeros(3), np.column_stack([[1.0, 0.0, 0.0], [0.0, 0.0, self.lateral_z], [0.0, 1.0, 0.0]])

    def held_point(self, obj):
        return np.zeros(3)

    def point_world(self, point, q=None):
        return np.zeros(3)

    def line(self, end, orient, steps=6, *args, **kwargs):
        self.calls.append(("line", orient["lateral"].copy()))
        return iter(())

    def lower_until_supported(self, obj, rest, carry):
        self.calls.append(("lower", None))
        return iter(())

    def grip(self, value, *args, **kwargs):
        self.calls.append(("grip", value))
        return iter(())

    def wait(self, steps):
        return iter(())


def test_the_bottle_is_returned_when_the_policy_picked_it_up():
    arm = HoldingArm(lateral_z=-1.0)
    list(arm.return_bottle(np.array([0.3, -0.2])))
    assert ("lower", None) in arm.calls
    laterals = [lat for kind, lat in arm.calls if kind == "line"]
    assert laterals and all(lat[2] < 0 for lat in laterals)  # keeps the jaw orientation it holds with
    assert arm.side is None


def test_an_empty_hand_still_returns_nothing():
    arm = HoldingArm(lateral_z=1.0, holding=False)
    list(arm.return_bottle(np.array([0.3, -0.2])))
    assert arm.calls == [] and arm.warnings
