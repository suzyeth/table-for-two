"""sim/env.py: the water beads inside a bottle are not something it stands on.

``supported()`` used to count every non-finger contact as support. With a couple of beads
left in the bottle after the pour, the set-down stopped lowering at once and opened the jaws
~47 mm above the table.
"""
from types import SimpleNamespace

import numpy as np

from sim.env import DinnerTableEnv

TABLE, LEFT_JAWS, WATER = 1, {3, 4}, (20, 21)


def fake_env(touching):
    return SimpleNamespace(_contact_bodies=lambda obj: set(touching),
                           finger_bodies={"left_": set(LEFT_JAWS), "right_": {5, 6}},
                           water_ids=np.array(WATER))


def test_held_bottle_touching_only_its_water_is_not_supported():
    assert DinnerTableEnv.supported(fake_env(LEFT_JAWS | set(WATER)), "bottle") is False


def test_bottle_on_the_table_is_supported_with_water_inside():
    assert DinnerTableEnv.supported(fake_env(LEFT_JAWS | {TABLE, WATER[0]}), "bottle") is True


def test_held_bottle_touching_nothing_else_is_not_supported():
    assert DinnerTableEnv.supported(fake_env(LEFT_JAWS), "bottle") is False
