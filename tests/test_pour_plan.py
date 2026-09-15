"""sim/pour.py: the pour plan never jumps a joint, never skips tilts silently, keeps the hand and
bottle clear of the mug, and unwinds lifted and at the forward speed.

Review 1 measured the old plan: tilts 70/80/88 had no solution at the tight clearance and were
skipped without a warning, so wrist_flex jumped 2.21 rad from 60 to 95 deg; the unwind played
that jump back in 0.3 s (10.6 rad/s commanded) and the bottle and gripper housing tipped the mug
21-31 deg.
"""
from types import SimpleNamespace

import numpy as np

from sim.pour import MAX_POUR_JOINT_STEP, POUR_JOINT_SPEED, POUR_MAX_CLEARANCE, TILTS_DEG, PourMixin, PourStep

RIM = np.array([0.0, 0.0, 0.46])


class FakeArm(PourMixin):
    """Pour planner with a fake IK: joint 3 follows the tilt (rad), joint 4 records the clearance."""

    def __init__(self, solvable=lambda tilt_deg, clearance: True, clear=lambda q: True):
        self.cmd = np.zeros(6)
        self.arm = "right_"
        self.env = SimpleNamespace(arm_base=lambda arm: np.array([-0.3, 0.0, 0.4]),
                                   object_frame=lambda obj: (np.zeros(3), np.eye(3)))
        self.ik = SimpleNamespace(last_rot_err=0.0)
        self.warnings = []
        self.solvable, self.clear = solvable, clear
        self.timed = []

    def _bottle_in_gripper(self):
        return [], [], np.array([0.0, 1.0, 0.0])

    def _pour_pose(self, q_seed, tilt, lean, rim, lip, outline, axis_g, clearance, restarts):
        if not self.solvable(float(np.degrees(tilt)), float(clearance)):
            return np.asarray(q_seed, dtype=float), 1.0
        return np.array([0.0, 0.0, 0.0, tilt, clearance]), 0.0

    def _pour_pose_clear(self, q, margin=None):
        return self.clear(q)


def max_step(plan):
    qs = [np.zeros(5)] + [s.q for s in plan]
    return max(float(np.max(np.abs(b - a))) for a, b in zip(qs, qs[1:]))


def test_plan_reaches_the_last_tilt_without_moving_a_joint_more_than_the_step_limit():
    plan = FakeArm()._plan_pour(RIM)
    assert plan[-1].tilt == TILTS_DEG[-1]
    assert max_step(plan) <= MAX_POUR_JOINT_STEP + 1e-9


def test_tilts_unreachable_at_the_tight_clearance_use_more_clearance_instead_of_being_skipped():
    def tight_fails(tilt, clearance):
        return not (70 <= tilt <= 88 and clearance <= POUR_MAX_CLEARANCE + 1e-9)

    plan = FakeArm(solvable=tight_fails)._plan_pour(RIM)
    by_tilt = {round(s.tilt, 3): s for s in plan}
    assert all(t in by_tilt and by_tilt[t].clearance > POUR_MAX_CLEARANCE for t in (70, 80, 88))
    assert plan[-1].tilt == TILTS_DEG[-1]
    assert max_step(plan) <= MAX_POUR_JOINT_STEP + 1e-9


def test_plan_stops_before_a_gap_it_cannot_cross_clear_of_the_mug_and_warns():
    """Every way across the 65-90 deg gap passes poses that would touch the mug: stop, don't jump."""
    arm = FakeArm(solvable=lambda tilt, clearance: not 65 < tilt < 90,
                  clear=lambda q: not np.deg2rad(65.5) < q[3] < np.deg2rad(89.5))
    plan = arm._plan_pour(RIM)
    assert plan[-1].tilt <= 65
    assert max_step(plan) <= MAX_POUR_JOINT_STEP + 1e-9
    assert any("stops at" in w for w in arm.warnings)


def two_branches(tilt, clearance):
    """Fake IK with a gap: joints [0,0,0,tilt,10c] up to 80 deg, [1.5,0,0,tilt,10c] from 105 deg."""
    if tilt <= 80:
        return np.array([0.0, 0.0, 0.0, np.deg2rad(tilt), 10 * clearance])
    if tilt >= 105:
        return np.array([1.5, 0.0, 0.0, np.deg2rad(tilt), 10 * clearance])
    return None


def mug_below_5cm_mid_switch(q):
    """Half-way between the branches the hand passes over the mug: only 5 cm or more above the rim is clear."""
    return not (0.2 < q[0] < 1.3 and q[4] < 0.5 - 1e-9)


def branch_arm():
    arm = FakeArm(clear=mug_below_5cm_mid_switch)

    def pose(q_seed, tilt, lean, rim, lip, outline, axis_g, clearance, restarts):
        q = two_branches(float(np.degrees(tilt)), float(clearance))
        return (np.asarray(q_seed, dtype=float), 1.0) if q is None else (q, 0.0)

    arm._pour_pose = pose
    return arm


def test_a_gap_between_ik_branches_is_crossed_once_on_a_path_clear_of_the_mug():
    plan = branch_arm()._plan_pour(RIM)
    assert plan[-1].tilt == TILTS_DEG[-1]
    jumps = [(a, b) for a, b in zip(plan, plan[1:]) if np.max(np.abs(b.q - a.q)) > MAX_POUR_JOINT_STEP + 1e-9]
    assert len(jumps) == 1
    a, b = jumps[0]
    assert all(mug_below_5cm_mid_switch(a.q + (b.q - a.q) * s) for s in np.linspace(0, 1, 41))


def lifted_switch_hits_mug(q):
    """Half-way between the branches the pour-height switch is clear, but 4-9 cm up the hand would hit the mug."""
    return not (0.2 < q[0] < 1.3 and 0.4 < q[4] < 0.9)


def test_the_way_back_retraces_the_checked_switch_and_every_move_stays_clear():
    """Review: the unwind went between the branches through the lifted poses, a path never checked."""
    arm = branch_arm()
    arm.clear = lifted_switch_hits_mug
    moves = []

    def timed(q, grip, seconds):
        moves.append(np.asarray(q, dtype=float).copy())
        return iter(())

    arm._timed = timed
    arm.wait = lambda n: iter(())
    list(arm.pour_into("mug"))
    previous, crossings = arm.cmd[:5].copy(), 0
    for q in moves:
        assert all(lifted_switch_hits_mug(previous + (q - previous) * s) for s in np.linspace(0, 1, 41))
        crossings += (previous[0] < 0.75) != (q[0] < 0.75)  # joint 0 tells the two IK branches apart
        previous = q
    assert crossings == 2  # the switch there and back, nothing else changes branch


def test_no_pour_move_turns_a_joint_faster_than_the_pour_speed():
    arm = branch_arm()
    targets = []

    def timed(q, grip, seconds):
        targets.append((np.asarray(q, dtype=float).copy(), seconds))
        return iter(())

    arm._timed = timed
    arm.wait = lambda n: iter(())
    list(arm.pour_into("mug"))
    previous = arm.cmd[:5]
    for q, seconds in targets:
        assert np.max(np.abs(q - previous)) / seconds <= POUR_JOINT_SPEED + 1e-9
        previous = q


def test_poses_where_the_hand_or_bottle_would_touch_the_mug_are_not_used():
    plan = FakeArm(clear=lambda q: q[4] >= 0.025 - 1e-9)._plan_pour(RIM)
    assert plan and all(s.clearance >= 0.025 - 1e-9 for s in plan)


def test_every_waypoint_gets_a_lifted_pose_even_when_the_lift_itself_is_a_big_move():
    """Measured: lifting 3.5 cm moves a joint 0.36-0.73 rad; only the lift may be that big, not the way back."""
    arm = FakeArm()
    arm._pour_pose = lambda q_seed, tilt, lean, rim, lip, outline, axis_g, clearance, restarts: (
        np.array([0.0, clearance * 20, 0.0, tilt, 0.0]), 0.0)
    plan = arm._plan_pour(RIM)
    assert all(not np.allclose(s.q, s.q_raised) for s in plan)
    raised = [s.q_raised for s in plan]
    assert max(float(np.max(np.abs(b - a))) for a, b in zip(raised, raised[1:])) <= MAX_POUR_JOINT_STEP + 1e-9


def test_unwind_lifts_first_then_straightens_through_lifted_poses_at_the_forward_speed():
    arm = FakeArm()
    steps = [PourStep(60.0, 0.01, np.full(5, 0.1), np.full(5, 0.2)),
             PourStep(95.0, 0.02, np.full(5, 0.3), np.full(5, 0.4)),
             PourStep(100.0, 0.02, np.full(5, 0.5), np.full(5, 0.6))]
    arm._plan_pour = lambda rim: steps

    def timed(q, grip, seconds):
        arm.timed.append((round(float(q[0]), 2), seconds))
        return iter(())

    arm._timed = timed
    arm.wait = lambda n: iter(())
    list(arm.pour_into("mug"))
    assert arm.timed == [(0.1, 0.5), (0.3, 0.8), (0.5, 0.8),  # tilt forward
                         (0.6, 0.8),  # lift at full tilt
                         (0.4, 0.8), (0.2, 0.8)]  # straighten: each segment as long as it took forward
