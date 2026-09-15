"""Record one dataset with several recorder processes at once, then merge the shards.

Each shard runs ``data.record`` on its own seeds (``--shard K/N``: start+K, start+K+N, ...) into
``<root>/shard_K``; ``tools.run_parallel`` keeps them running side by side and
``tools.aggregate_contact`` merges them into ``<root>/merged``. Every seed stays below
``--seed-limit`` (the DAgger recorder starts at 500; the evaluation seeds are 0-9 and 1000-1029).
A recorded episode is ~145 s of simulation plus rendering, so one process needs many hours for
300 of them; the scene is step-driven, so running shards side by side changes no outcome.

Run:  .venv\\Scripts\\python.exe -m tools.record_parallel --episodes 300 --shards 8 --root data\\dinner_table_contact_v3
"""
import argparse
import sys
from pathlib import Path

from data.record import FIRST_TRAIN_SEED
from data.record_dagger import FIRST_DAGGER_SEED

SEEDS_PER_EPISODE = 1.1  # the demo gate keeps about 9 in 10 seeds: refuse a range with less room than this


def plan_shards(episodes, shards, start_seed, seed_limit, root):
    """One dict per shard: index, episodes, start_seed, max_tries, root and the recorder command."""
    seeds = seed_limit - start_seed
    if shards < 1 or seeds < episodes * SEEDS_PER_EPISODE:
        raise ValueError(f"{seeds} seeds ({start_seed}..{seed_limit - 1}) are too few for {episodes} episodes "
                         f"at {SEEDS_PER_EPISODE} seeds per kept episode")
    tries = seeds // shards
    base, extra = divmod(episodes, shards)
    plan = []
    for index in range(shards):
        count = base + (1 if index < extra else 0)
        shard_root = Path(root) / f"shard_{index}"
        command = (f'"{sys.executable}" -m data.record --episodes {count} --start-seed {start_seed} '
                   f'--shard {index}/{shards} --max-tries {tries} --root "{shard_root}" --overwrite')
        plan.append({"index": index, "episodes": count, "start_seed": start_seed, "max_tries": tries,
                     "root": str(shard_root), "command": command})
    return plan


def main():
    from tools.aggregate_contact import aggregate
    from tools.run_parallel import run_all

    parser = argparse.ArgumentParser(description="Record a dataset with parallel shards and merge them.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--shards", type=int, default=8, help="recorder processes running at once")
    parser.add_argument("--start-seed", type=int, default=FIRST_TRAIN_SEED)
    parser.add_argument("--seed-limit", type=int, default=FIRST_DAGGER_SEED, help="first seed not to use")
    parser.add_argument("--root", type=Path, required=True, help="shards go to ROOT/shard_K, the result to ROOT/merged")
    args = parser.parse_args()

    plan = plan_shards(args.episodes, args.shards, args.start_seed, args.seed_limit, args.root)
    results = run_all([(f"shard_{s['index']}", s["command"]) for s in plan], args.shards, args.root / "logs")
    failed = [name for name, (code, _) in results.items() if code != 0]
    if failed:
        raise SystemExit(f"shards failed: {failed}; see {args.root / 'logs'}")
    sources, episodes, frames = aggregate([s["root"] for s in plan], args.root / "merged", overwrite=True)
    print(f"merged {sources} -> {episodes} episodes, {frames} frames at {args.root / 'merged'}")
    if episodes < args.episodes:
        print(f"WARNING: {args.episodes - episodes} episodes short - some shard ran out of seeds "
              f"(see {args.root / 'logs'})")


if __name__ == "__main__":
    main()
