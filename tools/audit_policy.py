"""Audit how the learned policy puts objects down: set down gently, or dropped.

Runs the learned policy through the plan (pure policy, no scripted takeover) and
attaches the same ``Watcher`` that ``tools/audit_contact.py`` uses for the
scripted pipeline. Every physics control step is observed, including the pauses
the evaluator holds to confirm a stage, so no landing is missed.

Each release is classified:
  placed   the object was already resting on a support when the last finger left it;
  gentle   it fell at most GENTLE_DROP_MM and landed no faster than GENTLE_IMPACT_MPS;
  dropped  anything else (including an object lost while the jaws were still squeezing).

Run:  .venv\\Scripts\\python.exe -m tools.audit_policy --policy models/policy_v2/act_fp32.xml ^
        --checkpoint outputs/act_contact_v2/checkpoints/060000/pretrained_model --ensemble 0.01 --seeds 0 1 2
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data.record import SUBTASK_VOCAB, stage_signature
from sim.env import ARMS, PROPS
from sim.task import DEFAULT_PLAN

ROOT = Path(__file__).resolve().parent.parent
GENTLE_DROP_MM = 5.0  # released at most this far above its support ...
GENTLE_IMPACT_MPS = 0.10  # ... and landing no faster than this counts as a gentle set-down
CONFIRM_SECONDS = 1.0


class PolicyCommands:
    """Stands in for the scripted executor inside ``Watcher``: each arm's last commanded joint targets."""

    def __init__(self):
        self.skills = {arm: SimpleNamespace(cmd=np.zeros(6)) for arm in ARMS}

    def update(self, action):
        action = np.asarray(action, dtype=float)
        for i, arm in enumerate(ARMS):
            self.skills[arm].cmd = action[6 * i:6 * i + 6].copy()


def classify(event, max_drop_mm=GENTLE_DROP_MM, max_impact_mps=GENTLE_IMPACT_MPS):
    """'placed', 'gentle' or 'dropped' for one release event from ``Watcher``."""
    if event["resting_on_support_at_release"]:
        return "placed"
    if event.get("lost_while_squeezing"):
        return "dropped"
    height, speed = event.get("drop_height_mm"), event.get("drop_impact_speed_mps")
    if height is None or speed is None:  # still falling when the run ended
        return "dropped"
    return "gentle" if height <= max_drop_mm and speed <= max_impact_mps else "dropped"


def summarize(events):
    """Counts per object and overall, plus the worst drop height and impact speed."""
    per_object = {}
    for event in events:
        kind = classify(event)
        row = per_object.setdefault(event["object"], {"placed": 0, "gentle": 0, "dropped": 0,
                                                      "worst_drop_mm": 0.0, "worst_impact_mps": 0.0})
        row[kind] += 1
        row["worst_drop_mm"] = max(row["worst_drop_mm"], event.get("drop_height_mm") or 0.0)
        row["worst_impact_mps"] = max(row["worst_impact_mps"], event.get("drop_impact_speed_mps") or 0.0)
    total = {kind: sum(r[kind] for r in per_object.values()) for kind in ("placed", "gentle", "dropped")}
    return {"per_object": per_object, "total": total}


def audit_seed(env, policy, seed, plan=DEFAULT_PLAN):
    """Run one seed with every control step watched; return (events, per-stage ok, final success)."""
    from policy.rollout import run_stage_policy
    from tools.audit_contact import Watcher

    env.reset(seed)
    commands = PolicyCommands()
    watcher = Watcher(env, commands)
    current = {"label": ""}
    raw_step = env.step

    def watched_step(action):
        result = raw_step(action)
        commands.update(action)
        watcher.step(current["label"])
        return result

    env.step = watched_step  # env.hold() calls self.step, so confirmation pauses are watched too
    try:
        stage_ok = []
        for stage in plan:
            current["label"] = stage_signature(stage)
            stage_ok.append(bool(run_stage_policy(env, policy, stage, SUBTASK_VOCAB.index(current["label"]))))
        env.hold(CONFIRM_SECONDS)
    finally:
        del env.step
    return watcher.events, stage_ok, env.success()


def main():
    from policy.ov_policy import OVActPolicy
    from sim.env import DinnerTableEnv

    parser = argparse.ArgumentParser(description="Set-down vs drop audit of the learned policy.")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--ensemble", type=float, default=None, metavar="M")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "audit_policy.json")
    args = parser.parse_args()

    policy = OVActPolicy(args.policy, args.checkpoint, device=args.device, ensemble_m=args.ensemble)
    env = DinnerTableEnv(obs_cameras=())
    report, all_events = {"policy": str(args.policy), "ensemble": args.ensemble, "seeds": {}}, []
    for seed in args.seeds:
        events, stage_ok, success = audit_seed(env, policy, seed)
        for event in events:
            event["kind"] = classify(event)
        all_events.extend(events)
        report["seeds"][seed] = {"events": events, "stage_ok": stage_ok, "success": success}
        drops = [f"{e['object']} {e.get('drop_height_mm', '?')} mm @ {e.get('drop_impact_speed_mps', '?')} m/s"
                 for e in events if e["kind"] == "dropped"]
        print(f"seed {seed}: stages {sum(stage_ok)}/{len(stage_ok)} full={success['all']} "
              f"releases {len(events)} dropped {len(drops)} {drops}", flush=True)
    env.close()
    report["summary"] = summarize(all_events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
