"""Where does the scripted contact pipeline break? Push each perturbation past its
randomisation range, one at a time, and report success per level.

The seed randomisation covers pad friction x0.7-1.3, mass x0.8-1.2 and position
+-1.2 cm. This sweep forces a single factor to a fixed value (the others stay at
their nominal, seed-0 values) over several seeds and records which sub-goals
survive. Writes out/robustness_envelope.json and prints a table.

Run:  python -m tools.robustness_envelope --seeds 1 2 3
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from sim.env import PROPS, DinnerTableEnv
from sim.task import DEFAULT_PLAN, Executor, run_seed

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "robustness_envelope.json"

SWEEPS = {
    "pad_friction_scale": [0.4, 0.5, 0.6, 1.0, 1.4, 1.6],
    "mass_scale": [0.5, 0.7, 1.0, 1.5, 2.0],
    "position_jitter_cm": [0.0, 1.5, 2.0, 2.5, 3.0],
}


class ForcedEnv(DinnerTableEnv):
    """Env whose reset applies one forced factor instead of drawing it from the seed."""

    forced = {}

    def reset(self, seed=0):
        obs = super().reset(seed)
        m, d = self.model, self.data
        if "pad_friction_scale" in self.forced:
            m.geom_friction[self.pad_geoms, 0] = self._friction0[self.pad_geoms, 0] * self.forced["pad_friction_scale"]
        if "mass_scale" in self.forced:
            for obj in PROPS:
                body = self.body_ids[obj]
                m.body_mass[body] = self._mass0[body] * self.forced["mass_scale"]
                m.body_inertia[body] = self._inertia0[body] * self.forced["mass_scale"]
            mujoco.mj_setConst(m, d)
        if "position_jitter_cm" in self.forced:
            rng = np.random.default_rng(10_000 + seed)
            jitter = self.forced["position_jitter_cm"] / 100
            for obj in PROPS:
                adr = self.free_qadr[obj]
                scale = 0.3 if obj in ("spoon", "fork") else 1.0  # utensils are boxed in by the drawer
                old = d.qpos[adr:adr + 2].copy()
                new = self._qpos0[adr:adr + 2] + rng.uniform(-jitter, jitter, 2) * scale
                d.qpos[adr:adr + 2] = new
                if obj == "bottle":
                    for bead in self.water_ids:
                        bead_adr = m.jnt_qposadr[m.body_jntadr[bead]]
                        d.qpos[bead_adr:bead_adr + 2] += new - old
            mujoco.mj_forward(m, d)
            for _ in range(500):
                mujoco.mj_step(m, d)
        return obs


def main():
    parser = argparse.ArgumentParser(description="Perturbation envelope of the scripted pipeline.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--factors", nargs="+", default=list(SWEEPS))
    args = parser.parse_args()
    env = ForcedEnv(obs_cameras=())
    executor = Executor(env)
    report = {}
    for factor in args.factors:
        report[factor] = {}
        for level in SWEEPS[factor]:
            ForcedEnv.forced = {factor: level}
            results = []
            for seed in args.seeds:
                result, _ = run_seed(env, executor, seed, DEFAULT_PLAN)
                results.append(result)
            rate = float(np.mean([r["all"] for r in results]))
            failed = sorted({k for r in results for k, v in r.items() if not v and k != "all"})
            report[factor][str(level)] = {"full_task": rate, "failed_subgoals": failed}
            print(f"{factor:22s} {level:5}: full task {int(rate * len(args.seeds))}/{len(args.seeds)}"
                  + (f"  failing: {failed}" if failed else ""), flush=True)
    env.close()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"seeds": args.seeds, "sweeps": report}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
