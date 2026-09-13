"""Audit how the scripted plan holds and puts down every object.

For each object and seed it measures, from the physical state at every control
step:
  * balance in the hand - how far the object tilts (utensils: how much the head
    sags) away from the attitude it had when it was picked up, and how far it
    slides relative to the fingers;
  * touch-down - the object's downward speed when it first touches a support
    (table, plate, drawer...) while still held;
  * release - whether the object was resting on a support when the fingers let
    go, or was dropped (and then from what height and at what speed it landed);
  * unintended drops - the fingers lose the object while still squeezing.

The bottle's own pour tilt is excluded from its balance figure.

Run:  .venv\\Scripts\\python.exe -m tools.audit_contact --seeds 0 1 2
"""
import argparse
import json

import mujoco
import numpy as np

from sim.env import ARMS, GRIPPER_SQUEEZE, PROPS, DinnerTableEnv
from sim.task import DEFAULT_PLAN, Executor
from tools.grasp_lab import ROOT

UTENSIL_OBJECTS = ("spoon", "fork")
SQUEEZING = GRIPPER_SQUEEZE + 0.05  # a commanded jaw below this means "still gripping"


def attitude(env, obj):
    """Unit vector that defines the object's attitude: long axis for utensils, up axis otherwise."""
    rot = env.object_frame(obj)[1]
    return rot[:, 0] if obj in UTENSIL_OBJECTS else rot[:, 2]


def robot_and_water_bodies(env):
    """Bodies that never count as a support: every link of both arms, and the water beads.

    Beads sit inside the full bottle (and later the mug) and touch its walls, and an object
    can lean on the other arm's wrist; neither means the object has been set down.
    """
    cached = getattr(env, "_not_support_ids", None)
    if cached is None:
        model = env.model
        robot = {i for i in range(model.nbody) if (model.body(i).name or "").startswith(ARMS)}
        cached = frozenset(robot | {int(b) for b in env.water_ids})
        env._not_support_ids = cached
    return cached


def supports(env, obj):
    """Bodies other than the robot and the water touching ``obj`` (table, plate, drawer, other props)."""
    return env._contact_bodies(obj) - robot_and_water_bodies(env)


def velocity(env, obj):
    res = np.zeros(6)
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, env.body_ids[obj], res, 0)
    return res[3:]


class Watcher:
    """Tracks every "hold": from the moment an object leaves its support while a finger touches it
    until no finger touches it any more. Works for one hand, two hands on one object, and a
    hand-over (the set of touching arms changes while the object is in the air)."""

    def __init__(self, env, executor):
        self.env, self.executor = env, executor
        self.hold = {}  # obj -> dict for the current held phase
        self.falling = {}  # obj -> dict for an object in free fall after release
        self.events = []

    def touching_arms(self, obj):
        bodies = self.env._contact_bodies(obj)
        return {arm for arm in ARMS if bodies & self.env.finger_bodies[arm]}

    def step(self, stage):
        env = self.env
        for obj in PROPS:
            arms = self.touching_arms(obj)
            supported = bool(supports(env, obj))
            if obj in self.hold:
                if not arms:
                    self._release(obj, stage, supported)
                else:
                    self._during_hold(obj, stage, arms, supported)
            elif arms and not supported:
                self._start_hold(obj, arms, stage)
            if obj in self.falling:
                self._during_fall(obj)

    def _gripper_frame(self, arm):
        body = self.env.ik[arm].gripper_body
        return self.env.data.xpos[body].copy(), self.env.data.xmat[body].reshape(3, 3).copy()

    def _local(self, obj, arm):
        pos, rot = self._gripper_frame(arm)
        return rot.T @ (self.env.object_frame(obj)[0] - pos)

    def _start_hold(self, obj, arms, stage):
        self.hold[obj] = {"arms": set(arms), "first_arms": set(arms), "stage": stage,
                          "ref_attitude": attitude(self.env, obj), "ref_local": {a: self._local(obj, a) for a in arms},
                          "max_tilt": 0.0, "max_slip": 0.0, "supported": False, "touchdown_speed": None,
                          "handed_over": False}

    def _during_hold(self, obj, stage, arms, supported):
        h = self.hold[obj]
        env = self.env
        if arms != h["arms"]:
            if arms - h["first_arms"]:  # a new hand took it
                h["handed_over"] = True
            for a in arms - set(h["ref_local"]):
                h["ref_local"][a] = self._local(obj, a)
            h["arms"] = set(arms)
        if obj in UTENSIL_OBJECTS:
            # Sag: change in the long axis' elevation (turning it in the horizontal plane is intended).
            now = float(np.degrees(np.arcsin(np.clip(attitude(env, obj)[2], -1, 1))))
            ref = float(np.degrees(np.arcsin(np.clip(h["ref_attitude"][2], -1, 1))))
            h["max_tilt"] = max(h["max_tilt"], abs(now - ref))
        elif not (obj == "bottle" and h["stage"] != stage):  # the bottle's own pour tilt is intended
            cos = np.clip(attitude(env, obj) @ h["ref_attitude"], -1, 1)
            h["max_tilt"] = max(h["max_tilt"], float(np.degrees(np.arccos(cos))))
        for a in arms:
            slip = float(np.linalg.norm(self._local(obj, a) - h["ref_local"][a]))
            h["max_slip"] = max(h["max_slip"], slip)
        if supported and not h["supported"]:
            speed = float(-velocity(env, obj)[2])
            h["touchdown_speed"] = speed if h["touchdown_speed"] is None else max(h["touchdown_speed"], speed)
        h["supported"] = supported

    def _release(self, obj, stage, resting):
        env = self.env
        h = self.hold.pop(obj)
        squeezing = any(self.executor.skills[a].cmd[5] < SQUEEZING for a in h["arms"])
        event = {"object": obj, "arm": "+".join(sorted(h["first_arms"])) + ("->" + "+".join(sorted(h["arms"])) if h["handed_over"] else ""),
                 "picked_in": h["stage"], "released_in": stage,
                 "in_hand_tilt_deg": round(h["max_tilt"], 1), "in_hand_slip_mm": round(h["max_slip"] * 1000, 1),
                 "touchdown_speed_mps": None if h["touchdown_speed"] is None else round(h["touchdown_speed"], 3),
                 "resting_on_support_at_release": bool(resting), "handed_over": h["handed_over"],
                 "lost_while_squeezing": bool(squeezing and not resting),
                 "tilt_at_release_deg": round(env.tilt_deg(obj), 1) if obj not in UTENSIL_OBJECTS else None}
        self.events.append(event)
        if not resting:
            self.falling[obj] = {"event": event, "z0": float(env.object_frame(obj)[0][2]), "max_speed": 0.0}

    def _during_fall(self, obj):
        f = self.falling[obj]
        speed = float(np.linalg.norm(velocity(self.env, obj)))
        f["max_speed"] = max(f["max_speed"], speed)
        if supports(self.env, obj) or self.touching_arms(obj):
            z = float(self.env.object_frame(obj)[0][2])
            f["event"]["drop_height_mm"] = round((f["z0"] - z) * 1000, 1)
            f["event"]["drop_impact_speed_mps"] = round(f["max_speed"], 3)
            del self.falling[obj]


def audit(seeds):
    env = DinnerTableEnv(obs_cameras=())
    executor = Executor(env)
    report = {}
    for seed in seeds:
        env.reset(seed)
        executor.reset()
        watcher = Watcher(env, executor)
        executor.run(DEFAULT_PLAN, on_step=lambda _a, stage: watcher.step(stage))
        report[seed] = {"events": watcher.events, "success": env.success()}
        print(f"seed {seed}: success={env.success()['all']}")
        for e in watcher.events:
            fall = (f" DROPPED {e.get('drop_height_mm', '?')} mm at {e.get('drop_impact_speed_mps', '?')} m/s"
                    if not e["resting_on_support_at_release"] else "")
            lost = " LOST WHILE SQUEEZING" if e["lost_while_squeezing"] else ""
            print(f"  {e['object']:6s} {e['arm']:13s} {e['picked_in']:28s} -> {e['released_in']:28s} "
                  f"tilt {e['in_hand_tilt_deg']:5.1f} deg  slip {e['in_hand_slip_mm']:5.1f} mm  "
                  f"touchdown {e['touchdown_speed_mps']} m/s  rest-tilt {e['tilt_at_release_deg']}{fall}{lost}")
    env.close()
    return report


def main():
    parser = argparse.ArgumentParser(description="Balance / touch-down / drop audit of the scripted plan.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(ROOT / "out" / "audit_contact.json"))
    args = parser.parse_args()
    report = audit(args.seeds)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
