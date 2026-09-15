"""sim/ik.py: the hand-written 3-vector cross product gives exactly numpy's result (it only saves time).

Profiling the pour planner: np.cross on 3-vectors was ~30% of all IK time.
"""
import numpy as np

from sim.ik import _cross3


def test_cross3_matches_numpy_bit_for_bit():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        a, b = rng.normal(size=3), rng.normal(size=3) * 10.0 ** rng.integers(-6, 6)
        assert np.array_equal(_cross3(a, b), np.cross(a, b))
