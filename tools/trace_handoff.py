"""Debug helper: step through the fork handoff stage and report each sub-part.

Run:  .venv\\Scripts\\python.exe -m tools.trace_handoff --seed 0
"""
import argparse

import numpy as np

from sim.env import ARMS, DinnerTableEnv
from sim.task import DEFAULT_PLAN, Executor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = DinnerTableEnv(obs_cameras=())
    ex = Executor(env)
    env.reset(args.seed)
    ex.reset()
    handoff_index = next(i for i, stage in enumerate(DEFAULT_PLAN) if stage[0]["skill"] == "handoff")
    ex.run(DEFAULT_PLAN[:handoff_index])
    print("before handoff: fork", np.round(env.grasp_point("fork"), 3),
          "drawer", round(float(env.data.qpos[env.drawer_qadr]), 3), "intent", env.grasp_intent)
    for part, sub in enumerate(ex.expand(DEFAULT_PLAN[handoff_index])):
        gens = {}
        for arm_name, factory in sub.items():
            gens[arm_name] = factory()
        while gens:
            for arm_name in list(gens):
                try:
                    next(gens[arm_name])
                except StopIteration:
                    del gens[arm_name]
            if gens:
                env.step(np.concatenate([ex.skills[name].cmd for name in ARMS]))
        fork = env.grasp_point("fork")
        print(f"part {part} ({'+'.join(sub)}): held={env.held} fork={np.round(fork, 3)} "
              f"L_tcp={np.round(env.tcp('left_'), 3)} R_tcp={np.round(env.tcp('right_'), 3)} "
              f"R_dist={np.linalg.norm(env.tcp('right_') - fork):.3f} intent={env.grasp_intent}")
    target = env.data.site_xpos[env.targets["fork"]]
    print("fork target", np.round(target, 3), "placed", env.success()["fork_placed"])


if __name__ == "__main__":
    main()
