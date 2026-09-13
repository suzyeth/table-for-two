"""Merge two locally recorded contact datasets into one LeRobot dataset.

Both source datasets were written by ``data/record.py`` with the same features
and task vocabulary, only from different seed ranges.

Run:  .venv\Scripts\python.exe -m tools.aggregate_contact --roots data/dinner_table_contact data/dinner_table_contact_b --out data/dinner_table_contact_300
"""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Aggregate recorded LeRobot datasets.")
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    from data.record import REPO_ID
    from lerobot.datasets.aggregate import aggregate_datasets

    for root in args.roots:
        if not (root / "meta" / "info.json").exists():
            raise SystemExit(f"no LeRobot dataset at {root}")
    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")
        shutil.rmtree(args.out)

    aggregate_datasets(repo_ids=[REPO_ID] * len(args.roots), aggr_repo_id=REPO_ID + "_aggr",
                       roots=list(args.roots), aggr_root=args.out)
    info = json.loads((args.out / "meta" / "info.json").read_text(encoding="utf-8"))
    sources = [json.loads((r / "meta" / "info.json").read_text(encoding="utf-8"))["total_episodes"] for r in args.roots]
    print(f"aggregated {sources} -> {info['total_episodes']} episodes, {info['total_frames']} frames at {args.out}")
    if info["total_episodes"] != sum(sources):
        raise SystemExit("episode count mismatch")


if __name__ == "__main__":
    main()
