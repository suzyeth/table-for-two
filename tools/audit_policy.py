"""Audit how objects are put down: set down gently, or dropped, tipped or knocked.

Runs the learned policy through the plan (pure policy, no scripted takeover) -
or, with ``--scripted``, the scripted pipeline the demonstrations came from - and
attaches the same ``Watcher`` that ``tools/audit_contact.py`` uses. Every physics
control step is observed, including the pauses the evaluator holds to confirm a
stage, so no landing is missed.

Each release is classified:
  placed   the object was already resting on a support when the last finger left it;
  gentle   it fell at most GENTLE_DROP_MM and landed no faster than GENTLE_IMPACT_MPS;
  dropped  anything else (including an object lost while the jaws were still squeezing);
  tipped   came to rest more than UPRIGHT_TOL_DEG off upright (plate, mug, bottle).
Separately, a prop that is not held, not being grasped and not falling from a release, but
moves faster than KNOCK_SPEED_MPS or tilts more than ROCK_TILT_DEG off its resting tilt, is
followed until it is at rest again - also while a hand is touching it, so a gripper shoving
the mug is seen while it shoves (and named in ``hit_by``):
  knocked  something other than the table and the water touched it (``hit_by``: an arm, the
           bottle, another prop), it ended up more than KNOCK_MOVED_MM from where it was, or it
           rocked more than ROCK_TILT_DEG (a mug rocking on its base edge barely moves its origin);
  jostled  only the table and the water beads touched it and it barely moved - the beads
           landing in the mug make it twitch, which is not the robot hitting anything.

Supports never include the robot or the water beads (see ``tools.audit_contact.supports``).

Run:  .venv\\Scripts\\python.exe -m tools.audit_policy --policy models/policy_v2/act_fp32.xml ^
        --checkpoint outputs/act_contact_v2/checkpoints/060000/pretrained_model --ensemble 0.01 --seeds 0 1 2
      .venv\\Scripts\\python.exe -m tools.audit_policy --scripted --seeds 0 1 2
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data.record import EVAL_SEEDS, SUBTASK_VOCAB, stage_signature
from sim.env import ARMS, PROPS, UPRIGHT_TOL_DEG, UTENSILS
from sim.task import DEFAULT_PLAN

ROOT = Path(__file__).resolve().parent.parent
GENTLE_DROP_MM = 5.0  # released at most this far above its support ...
GENTLE_IMPACT_MPS = 0.10  # ... and landing no faster than this counts as a gentle set-down
CONFIRM_SECONDS = 1.0
KNOCK_SPEED_MPS = 0.15  # an unheld prop moving faster than this is followed as a possible knock
KNOCK_REST_MPS = 0.02  # ... until it is supported and slower than this
KNOCK_MOVED_MM = 2.0  # ... and it counts as knocked if it ended up further than this from where it was
ROCK_TILT_DEG = 2.0  # an unheld plate / mug / bottle tilting this far off its resting tilt is followed too
SETTLED_TILT_STEP_DEG = 0.1  # ... and is at rest only once its tilt changes less than this per control step
RELEASE_GRACE_STEPS = 10  # motion within 0.5 s of a release belongs to that release, not a knock
SETTLE_STEPS = 20  # 1 s after a release the object has come to rest; its tilt is judged then
KINDS = ("placed", "gentle", "dropped", "tipped", "knocked", "jostled")
MOTION_KINDS = ("knocked", "jostled")


class PolicyCommands:
    """Stands in for the scripted executor inside ``Watcher``: each arm's last commanded joint targets."""

    def __init__(self):
        self.skills = {arm: SimpleNamespace(cmd=np.zeros(6)) for arm in ARMS}

    def update(self, action):
        action = np.asarray(action, dtype=float)
        for i, arm in enumerate(ARMS):
            self.skills[arm].cmd = action[6 * i:6 * i + 6].copy()


def classify(event, max_drop_mm=GENTLE_DROP_MM, max_impact_mps=GENTLE_IMPACT_MPS):
    """'placed', 'gentle', 'dropped' or 'tipped' for one release event from ``Watcher``.

    Tipping is judged on ``settled_tilt_deg`` (measured SETTLE_STEPS after the release, once the
    object has come to rest) when available: at the instant of release a plate still rocks
    between two hands and reads a few degrees more than where it ends up.
    """
    tilt = event.get("settled_tilt_deg", event.get("tilt_at_release_deg"))
    tipped = tilt is not None and tilt > UPRIGHT_TOL_DEG
    if event["resting_on_support_at_release"]:
        return "tipped" if tipped else "placed"
    if event.get("lost_while_squeezing"):
        return "dropped"
    height, speed = event.get("drop_height_mm"), event.get("drop_impact_speed_mps")
    if height is None or speed is None:  # still falling when the run ended
        return "dropped"
    if height > max_drop_mm or speed > max_impact_mps:
        return "dropped"
    return "tipped" if tipped else "gentle"


def motion_kind(hit_by, moved_mm, tilt_change_deg=0.0):
    """'knocked' if something other than table/water touched it, it was displaced or it rocked; else 'jostled'."""
    if hit_by or moved_mm is None or moved_mm > KNOCK_MOVED_MM or tilt_change_deg > ROCK_TILT_DEG:
        return "knocked"
    return "jostled"


class KnockMonitor:
    """Props that move or rock without being held, grasped or falling from a release.

    It also records each released object's tilt SETTLE_STEPS after the release
    (``settled_tilt_deg``), which ``classify`` uses to judge tipping.
    """

    def __init__(self, env, watcher):
        self.env, self.watcher = env, watcher
        self.moving = {}  # obj -> {"stage", "z0", "xy0", "max_speed", "max_tilt_change", "last_tilt", "hit_by", "water"}
        self.rest_tilt = {}  # obj -> tilt while it was last at rest (None for utensils)
        self.last_tilt = {}  # obj -> tilt at the previous control step it was watched
        self.events = []
        self.steps = 0
        self.last_release = {}  # obj -> control step of its latest release
        self.seen_releases = 0
        self.pending_tilt = []  # (release event, control step at which to measure its settled tilt)

    def step(self, stage):
        self.steps += 1
        for event in self.watcher.events[self.seen_releases:]:
            self.last_release[event["object"]] = self.steps
            if event.get("tilt_at_release_deg") is not None:  # utensils have no upright to lose
                self.pending_tilt.append((event, self.steps + SETTLE_STEPS))
        self.seen_releases = len(self.watcher.events)
        for event, due in self.pending_tilt:
            if due <= self.steps and event["object"] not in self.watcher.hold:  # regrasped: keep release tilt
                event["settled_tilt_deg"] = round(float(self.env.tilt_deg(event["object"])), 1)
        self.pending_tilt = [(e, due) for e, due in self.pending_tilt if due > self.steps]
        for obj in PROPS:
            self._step_prop(obj, stage)

    def _step_prop(self, obj, stage):
        from tools.audit_contact import supports, velocity

        # A hand merely touching a prop does not hide it: only holding, grasping (both jaws of one
        # arm on it) or falling from a release does. A shove that ends in a grasp is still reported.
        falling = obj in self.watcher.falling
        if falling or obj in self.watcher.hold or self.env.holder(obj) is not None:
            moving = self.moving.pop(obj, None)
            if moving is not None and not falling:
                self.events.append(self._event(obj, moving, self.env.object_frame(obj)[0], still_moving=False,
                                               ended_in_grasp=True))
            self.rest_tilt.pop(obj, None)
            self.last_tilt.pop(obj, None)
            return
        tilt = None if obj in UTENSILS else float(self.env.tilt_deg(obj))
        seen_before, previous_tilt = obj in self.last_tilt, self.last_tilt.get(obj)
        self.last_tilt[obj] = tilt
        if self.steps - self.last_release.get(obj, -RELEASE_GRACE_STEPS) < RELEASE_GRACE_STEPS:
            self.rest_tilt.pop(obj, None)
            return
        speed = float(np.linalg.norm(velocity(self.env, obj)))
        pos = self.env.object_frame(obj)[0]
        if obj not in self.rest_tilt and obj not in self.moving:
            # The resting tilt is taken only once the prop is still: after a release it may still be toppling.
            still_tilt = tilt is None or (seen_before and abs(tilt - previous_tilt) < SETTLED_TILT_STEP_DEG)
            if still_tilt and speed < KNOCK_SPEED_MPS:
                self.rest_tilt[obj] = tilt
        rest_tilt = self.rest_tilt.get(obj)
        tilt_change = 0.0 if tilt is None or rest_tilt is None else abs(tilt - rest_tilt)
        if obj in self.moving:
            moving = self.moving[obj]
            moving["max_speed"] = max(moving["max_speed"], speed)
            moving["max_tilt_change"] = max(moving["max_tilt_change"], tilt_change)
            hit, water = self._touching(obj)
            moving["hit_by"] |= hit
            moving["water"] = moving["water"] or water
            settled = tilt is None or abs(tilt - moving["last_tilt"]) < SETTLED_TILT_STEP_DEG
            moving["last_tilt"] = tilt
            if speed < KNOCK_REST_MPS and settled and supports(self.env, obj):
                self.events.append(self._event(obj, moving, pos, still_moving=False))
                del self.moving[obj]
                self.rest_tilt[obj] = tilt
        elif speed > KNOCK_SPEED_MPS or tilt_change > ROCK_TILT_DEG:
            hit, water = self._touching(obj)
            self.moving[obj] = {"stage": stage, "z0": float(pos[2]), "xy0": np.array(pos[:2], dtype=float),
                                "max_speed": speed, "max_tilt_change": tilt_change, "last_tilt": tilt,
                                "hit_by": hit, "water": water}

    def _touching(self, obj):
        """(names of everything touching ``obj`` except the table and the water, water touching?)."""
        model = self.env.model
        names = {model.body(b).name for b in self.env._contact_bodies(obj)}
        water = any(n and n.startswith("water_") for n in names)
        return {n for n in names if n and n != "table" and not n.startswith("water_")}, water

    def _event(self, obj, moving, pos, still_moving, ended_in_grasp=False):
        """A finished (or unfinished) motion. One that ended in a grasp is judged only on how far it
        moved or rocked: the hand touching it is expected then."""
        moved_mm = float(np.linalg.norm(np.asarray(pos[:2], dtype=float) - moving["xy0"]) * 1000)
        tilt_change = moving["max_tilt_change"]
        kind = motion_kind(set() if ended_in_grasp else moving["hit_by"], moved_mm, tilt_change)
        event = {"object": obj, "kind": kind, "stage": moving["stage"],
                 "max_speed_mps": round(moving["max_speed"], 3), "moved_mm": round(moved_mm, 1),
                 "max_tilt_change_deg": round(tilt_change, 1),
                 "height_change_mm": round((float(pos[2]) - moving["z0"]) * 1000, 1),
                 "hit_by": sorted(moving["hit_by"]), "water_contact": bool(moving["water"])}
        if still_moving:
            event["still_moving_at_end"] = True
        if ended_in_grasp:
            event["ended_in_grasp"] = True
        return event

    def finish(self):
        """Motions not yet at rest when the run ends are reported too."""
        for obj, moving in self.moving.items():
            self.events.append(self._event(obj, moving, self.env.object_frame(obj)[0], still_moving=True))
        self.moving = {}


def event_kind(event):
    kind = event.get("kind")
    return kind if kind in MOTION_KINDS else classify(event)


def summarize(events):
    """Counts per object and overall, plus the worst drop height and worst impact / knock speed."""
    per_object = {}
    for event in events:
        kind = event_kind(event)
        row = per_object.setdefault(event["object"], {**{k: 0 for k in KINDS},
                                                      "worst_drop_mm": 0.0, "worst_impact_mps": 0.0})
        row[kind] += 1
        if kind == "jostled":
            continue
        if kind == "knocked":
            row["worst_impact_mps"] = max(row["worst_impact_mps"], event.get("max_speed_mps") or 0.0)
            continue
        row["worst_drop_mm"] = max(row["worst_drop_mm"], event.get("drop_height_mm") or 0.0)
        row["worst_impact_mps"] = max(row["worst_impact_mps"], event.get("drop_impact_speed_mps") or 0.0)
    total = {kind: sum(r[kind] for r in per_object.values()) for kind in KINDS}
    return {"per_object": per_object, "total": total}


def audit_seed(env, policy, seed, plan=DEFAULT_PLAN, executor=None):
    """Run one seed with every control step watched; return (events, per-stage ok, final success).

    With ``executor`` (a ``sim.task.Executor``) the scripted pipeline runs the plan instead of
    ``policy``; per-stage results are then not available and an empty list is returned.
    """
    from tools.audit_contact import Watcher

    env.reset(seed)
    if executor is not None:
        executor.reset()
    commands = executor if executor is not None else PolicyCommands()
    watcher = Watcher(env, commands)
    knocks = KnockMonitor(env, watcher)
    current = {"label": ""}
    raw_step = env.step

    def watched_step(action):
        result = raw_step(action)
        if executor is None:
            commands.update(action)
        watcher.step(current["label"])
        knocks.step(current["label"])
        return result

    env.step = watched_step  # env.hold() calls self.step, so confirmation pauses are watched too
    try:
        stage_ok = []
        if executor is not None:
            executor.run(plan, on_step=lambda _action, label: current.update(label=label))
        else:
            from policy.rollout import run_stage_policy

            for stage in plan:
                current["label"] = stage_signature(stage)
                ok = run_stage_policy(env, policy, stage, SUBTASK_VOCAB.index(current["label"]))
                stage_ok.append(bool(ok))
        env.hold(CONFIRM_SECONDS)
    finally:
        del env.step
    knocks.finish()
    return watcher.events + knocks.events, stage_ok, env.success()


def save_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")


def main():
    from sim.env import DinnerTableEnv
    from sim.task import Executor

    parser = argparse.ArgumentParser(description="Set-down vs drop / tip / knock audit.")
    parser.add_argument("--scripted", action="store_true", help="audit the scripted pipeline instead of a policy")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--ensemble", type=float, default=None, metavar="M")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "audit_policy.json")
    args = parser.parse_args()

    env = DinnerTableEnv(obs_cameras=())
    if args.scripted:
        policy, executor, source = None, Executor(env), "scripted"
    else:
        if args.policy is None or args.checkpoint is None:
            raise SystemExit("--policy and --checkpoint are required unless --scripted")
        from policy.ov_policy import OVActPolicy

        policy = OVActPolicy(args.policy, args.checkpoint, device=args.device, ensemble_m=args.ensemble)
        executor, source = None, str(args.policy)
    report, all_events = {"source": source, "ensemble": args.ensemble, "seeds": {}}, []
    for seed in args.seeds:
        events, stage_ok, success = audit_seed(env, policy, seed, executor=executor)
        for event in events:
            event["kind"] = event_kind(event)
        all_events.extend(events)
        report["seeds"][seed] = {"events": events, "stage_ok": stage_ok, "success": success}
        drops = [f"{e['object']} {e.get('drop_height_mm', '?')} mm @ {e.get('drop_impact_speed_mps', '?')} m/s"
                 for e in events if e["kind"] == "dropped"]
        knocked = [f"{e['object']} {e['moved_mm']} mm / {e['max_tilt_change_deg']} deg by {e['hit_by'] or 'nothing'}"
                   for e in events if e["kind"] == "knocked"]
        jostled = sum(e["kind"] == "jostled" for e in events)
        tipped = [e["object"] for e in events if e["kind"] == "tipped"]
        releases = sum(e["kind"] not in MOTION_KINDS for e in events)
        stages = f"{sum(stage_ok)}/{len(stage_ok)}" if stage_ok else "scripted"
        print(f"seed {seed}: stages {stages} full={success['all']} releases {releases} dropped {drops} "
              f"tipped {tipped} knocked {knocked} jostled-by-water {jostled}", flush=True)
        report["summary"] = summarize(all_events)
        save_report(report, args.out)  # after every seed, so a crash keeps what was already measured
    env.close()
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
