"""Keep a demonstration only if it did nothing a policy should not copy.

A recorded episode is rejected when, anywhere in it:
  * an object was dropped, tipped over or knocked (pushed, rocked more than 2 deg or hit - see
    ``tools/audit_policy.py``), including a shove that ended in a grasp;
  * an arm ran into itself for SELF_CONTACT_MAX_STEPS control steps or more, or with
    SELF_CONTACT_MAX_N or more (a one-step graze is tolerated);
  * the scripted skills reported an unsafe motion (UNSAFE_WARNINGS) - small IK misses are not counted.

Success alone is not enough: before this gate the recorder kept demos that knocked the mug
21-31 deg during the pour and jammed the right wrist camera into its own shoulder.

Run:  .venv\\Scripts\\python.exe -m tools.demo_gate --seeds 0 1 2
"""
import argparse
import contextlib

import mujoco
import numpy as np

from sim.env import ARMS

REJECT_KINDS = ("dropped", "tipped", "knocked")
SELF_CONTACT_MAX_STEPS = 3
SELF_CONTACT_MAX_N = 10.0
# A pour that did not tilt all the way ("pour plan reaches only / stops at") is not listed: it is a
# shorter pour, and the episode's 'poured' check already decides whether it counts.
UNSAFE_WARNINGS = ("without it touching a support", "line-up", "no reachable")


def demo_defects(events, self_contacts, warnings):
    """Reasons to reject a demo (empty if it may be kept).

    ``events``: release and motion events from ``Watcher`` / ``KnockMonitor``; ``self_contacts``:
    ``SelfContactMonitor.summary()``; ``warnings``: the executor's warnings.
    """
    from tools.audit_policy import event_kind

    defects = []
    for event in events:
        kind = event.get("kind") or event_kind(event)
        if kind in REJECT_KINDS:
            defects.append(f"{kind} {event['object']} in {event.get('stage') or event.get('released_in', '?')}")
    defects += [f"arm hit itself: {c['pair']} in {c['stage']} ({c['steps']} steps, {c['max_force_n']} N)"
                for c in self_contacts if c["steps"] >= SELF_CONTACT_MAX_STEPS or c["max_force_n"] >= SELF_CONTACT_MAX_N]
    defects += [f"unsafe motion: {w}" for w in warnings if any(token in w for token in UNSAFE_WARNINGS)]
    return defects


class SelfContactMonitor:
    """Contacts between two links of the same arm that are not joined to each other, per stage and pair."""

    def __init__(self, env):
        self.env = env
        model = env.model
        self._bodies = [model.body(b).name for b in range(model.nbody)]
        self._geoms = [model.geom(g).name or self._bodies[model.geom_bodyid[g]] for g in range(model.ngeom)]
        self._contacts = {}  # (stage, pair) -> {"steps", "max_force_n"}

    def step(self, stage):
        model, data = self.env.model, self.env.data
        force = np.zeros(6)
        seen = set()
        for k in range(data.ncon):
            contact = data.contact[k]
            b1, b2 = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
            n1, n2 = self._bodies[b1], self._bodies[b2]
            if not any(n1.startswith(arm) and n2.startswith(arm) for arm in ARMS):
                continue
            if model.body_parentid[b1] == b2 or model.body_parentid[b2] == b1:
                continue
            mujoco.mj_contactForce(model, data, k, force)
            key = (stage, "/".join(sorted((self._geoms[contact.geom1], self._geoms[contact.geom2]))))
            record = self._contacts.setdefault(key, {"steps": 0, "max_force_n": 0.0})
            if key not in seen:
                record["steps"] += 1
                seen.add(key)
            record["max_force_n"] = max(record["max_force_n"], float(force[0]))

    def summary(self):
        return [{"stage": stage, "pair": pair, "steps": r["steps"], "max_force_n": round(r["max_force_n"], 2)}
                for (stage, pair), r in sorted(self._contacts.items())]


class DemoGate:
    """Watches one episode (create it after ``env.reset`` / ``executor.reset``) and says if it may be kept."""

    def __init__(self, env, commands):
        from tools.audit_contact import Watcher
        from tools.audit_policy import KnockMonitor

        self.env = env
        self.watcher = Watcher(env, commands)
        self.knocks = KnockMonitor(env, self.watcher)
        self.self_contacts = SelfContactMonitor(env)
        self.label = ""

    def step(self):
        self.watcher.step(self.label)
        self.knocks.step(self.label)
        self.self_contacts.step(self.label)

    def defects(self, warnings):
        self.knocks.finish()
        return demo_defects(self.watcher.events + self.knocks.events, self.self_contacts.summary(), warnings)


@contextlib.contextmanager
def watched(env, gate):
    """Within the block every ``env.step`` (also those ``env.hold`` makes) is followed by ``gate.step()``.

    A wrapper already set on ``env`` (e.g. the audit's) is kept inside and put back afterwards.
    """
    raw_step = env.step
    previous = vars(env).get("step")

    def step(action):
        result = raw_step(action)
        gate.step()
        return result

    env.step = step
    try:
        yield gate
    finally:
        if previous is None:
            del env.step
        else:
            env.step = previous


def main():
    from sim.env import DinnerTableEnv
    from sim.task import DEFAULT_PLAN, Executor

    from data.command_noise import CommandNoise, free_space_gains, noisy

    parser = argparse.ArgumentParser(description="Which scripted demos the recorder would keep, and why not.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--noise", type=float, default=0.0,
                        help="command noise sigma (rad) on free-space moves, as data.record")
    parser.add_argument("--spread", type=float, default=1.0, help="placement spread of the table props, as data.record")
    args = parser.parse_args()
    env = DinnerTableEnv(obs_cameras=())
    executor = Executor(env)
    kept = 0
    for seed in args.seeds:
        env.reset(seed, spread=args.spread)
        executor.reset()
        gate = DemoGate(env, executor)
        with watched(env, gate):
            with noisy(env, CommandNoise(args.noise, seed, gain=free_space_gains(executor))):
                executor.run(DEFAULT_PLAN, on_step=lambda _action, label: setattr(gate, "label", label))
            env.hold(1.0)  # scored at rest, without the noise
        success = env.success()["all"]
        defects = gate.defects(executor.warnings())
        keep = success and not defects
        kept += keep
        print(f"seed {seed}: {'KEEP  ' if keep else 'REJECT'} success={success} {defects}", flush=True)
    env.close()
    print(f"kept {kept} of {len(args.seeds)}")


if __name__ == "__main__":
    main()
