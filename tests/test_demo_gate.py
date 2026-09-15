"""tools/demo_gate.py: a demonstration is kept only if it did nothing the policy should not copy.

Review 1: the recorder kept any episode that ended in success, so demos that knocked the mug,
dropped a bottle or jammed a wrist camera into the arm's own shoulder went into the dataset.
"""
import mujoco
import numpy as np
import pytest

from sim.env import DinnerTableEnv
from tools.demo_gate import SelfContactMonitor, demo_defects

JAMMED_HOVER = np.array([-0.68, -0.67, 0.70, 1.55, 1.99])  # right camera mount 5 mm into its shoulder


def event(obj, kind, stage="3:pour"):
    return {"object": obj, "kind": kind, "stage": stage}


def test_a_clean_demo_has_no_defects():
    events = [event("mug", "placed"), event("fork", "gentle"), event("mug", "jostled")]
    assert demo_defects(events, self_contacts=[], warnings=[]) == []


@pytest.mark.parametrize("kind", ["dropped", "tipped", "knocked"])
def test_drops_tips_and_knocks_are_defects(kind):
    defects = demo_defects([event("bottle", kind, "4:return")], self_contacts=[], warnings=[])
    assert len(defects) == 1 and kind in defects[0] and "bottle" in defects[0] and "4:return" in defects[0]


def test_a_one_step_graze_of_the_arm_is_tolerated_but_a_jam_is_not():
    graze = {"pair": "left_camera_box2/left_shoulder", "stage": "0:bimanual_place", "steps": 1, "max_force_n": 5.9}
    jam = {"pair": "right_camera_box2/right_shoulder", "stage": "5:handoff", "steps": 26, "max_force_n": 33.1}
    assert demo_defects([], self_contacts=[graze], warnings=[]) == []
    defects = demo_defects([], self_contacts=[graze, jam], warnings=[])
    assert len(defects) == 1 and "right_camera_box2/right_shoulder" in defects[0]


def test_a_short_but_hard_hit_of_the_arm_is_a_defect():
    hit = {"pair": "right_camera_box2/right_shoulder", "stage": "5:handoff", "steps": 1, "max_force_n": 12.0}
    assert len(demo_defects([], self_contacts=[hit], warnings=[])) == 1


def test_planner_warnings_about_unsafe_motion_are_defects_but_small_ik_misses_are_not():
    unsafe = ["right_ set bottle down without it touching a support",
              "right_ line-up for spoon only 3.1 mm clear of the arm itself",
              "right_ no reachable side grasp on the bottle"]
    for warning in unsafe:
        assert len(demo_defects([], self_contacts=[], warnings=[warning])) == 1, warning
    assert demo_defects([], self_contacts=[], warnings=["left_ IK 9.9 mm / 0.1 deg at [-0.012, 0.002, 0.49]"]) == []


def test_a_pour_that_did_not_tilt_all_the_way_is_judged_by_whether_it_poured_not_by_the_warning():
    """Review: 'pour plan reaches only N deg' / 'stops at N deg' mean a shorter pour, not an unsafe one;
    the episode's 'poured' check already decides whether it counts."""
    for warning in ("right_ pour plan reaches only 97 deg", "right_ pour plan stops at 90 deg: no pose further"):
        assert demo_defects([], self_contacts=[], warnings=[warning]) == [], warning


def test_watched_puts_back_a_step_wrapper_that_was_already_there():
    """Review: ``del env.step`` removed an outer wrapper (e.g. the audit's) instead of restoring it."""
    from types import SimpleNamespace

    from tools.demo_gate import watched

    class Env:
        def step(self, action):
            return "raw"

    env = Env()

    def outer(action):
        return "outer"

    env.step = outer
    with watched(env, SimpleNamespace(step=lambda: None)):
        assert env.step(0) == "outer"
    assert env.step is outer


@pytest.fixture(scope="module")
def env():
    env = DinnerTableEnv(obs_cameras=())
    env.reset(0)
    yield env
    env.close()


def test_monitor_records_the_wrist_camera_pressed_into_the_shoulder(env):
    adr = [env.model.jnt_qposadr[env.model.joint("right_" + j).id]
           for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]
    env.data.qpos[adr] = JAMMED_HOVER
    mujoco.mj_forward(env.model, env.data)
    monitor = SelfContactMonitor(env)
    monitor.step("5:handoff")
    monitor.step("5:handoff")
    contacts = monitor.summary()
    jam = [c for c in contacts if "right_camera_box2" in c["pair"]]
    assert jam and jam[0]["steps"] == 2 and jam[0]["stage"] == "5:handoff" and jam[0]["max_force_n"] > 0.0
    assert all(any(all(name.startswith(arm) for name in c["pair"].split("/")) for arm in ("left_", "right_"))
               for c in contacts)  # same arm only
