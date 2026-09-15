"""Merge two locally recorded contact datasets into one LeRobot dataset.

Both source datasets were written by ``data/record.py`` with the same features
and task vocabulary, only from different seed ranges.

Run:  .venv\\Scripts\\python.exe -m tools.aggregate_contact --roots data/dinner_table_contact data/dinner_table_contact_b --out data/dinner_table_contact_300
"""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def aggregate(roots, out, overwrite=False):
    """Merge the LeRobot datasets at ``roots`` into ``out``.

    Returns (episodes per source, total episodes, total frames).
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    from data.record import REPO_ID
    from lerobot.datasets.aggregate import aggregate_datasets

    roots, out = [Path(root) for root in roots], Path(out)
    for root in roots:
        if not (root / "meta" / "info.json").exists():
            raise SystemExit(f"no LeRobot dataset at {root}")
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    aggregate_datasets(repo_ids=[REPO_ID] * len(roots), aggr_repo_id=REPO_ID + "_aggr", roots=roots, aggr_root=out)
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    sources = [json.loads((r / "meta" / "info.json").read_text(encoding="utf-8"))["total_episodes"] for r in roots]
    if info["total_episodes"] != sum(sources):
        raise SystemExit(f"episode count mismatch: {sources} -> {info['total_episodes']}")
    return sources, info["total_episodes"], info["total_frames"]


def main():
    parser = argparse.ArgumentParser(description="Aggregate recorded LeRobot datasets.")
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    sources, episodes, frames = aggregate(args.roots, args.out, args.overwrite)
    print(f"aggregated {sources} -> {episodes} episodes, {frames} frames at {args.out}")


if __name__ == "__main__":
    main()
