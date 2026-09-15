"""sim/skills.py: a sideways line-up (hand-over receiver) is chosen so the arm stays clear of itself.

Measured on the hand-over: lining up 5 cm out along the configured direction put the right wrist
camera mount 5 mm into its own shoulder (9 mm on the way there); the same 5 cm turned 40 deg was
11-20 mm clear. Candidates keep the configured offset first, then turn it, then shorten it.
"""
import numpy as np

from sim.skills import approach_candidates, first_clear


def test_the_first_candidate_that_keeps_the_arm_clear_is_chosen():
    clearance = {0: -0.005, 1: 0.004, 2: 0.012, 3: 0.03}
    assert first_clear(range(4), lambda i: clearance[i], needed=0.008) == (2, 0.012)


def test_if_none_is_clear_enough_the_one_with_the_most_clearance_is_used():
    clearance = [-0.009, -0.001, -0.004]
    assert first_clear(range(3), lambda i: clearance[i], needed=0.008) == (1, -0.001)


def test_candidates_keep_the_given_offset_first_then_turn_it_then_shorten_it():
    given = np.array([0.0, -0.05, 0.0])
    candidates = approach_candidates(given)
    assert np.allclose(candidates[0], given)
    same_length = [c for c in candidates if abs(np.linalg.norm(c) - 0.05) < 1e-9]
    assert len(same_length) > 1 and all(np.allclose(a, b) for a, b in zip(candidates, same_length))
    assert np.linalg.norm(candidates[-1]) < 0.05
    assert all(abs(c[2]) < 1e-12 for c in candidates)
