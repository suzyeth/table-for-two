"""Copy evaluation results into the slide data and print the README results table.

Reads the reports named on the command line (all required - the old defaults were v1's files):
  --policy-report      chained, learned policy only (policy/rollout.py --mode policy)
  --hybrid-report      chained, scripted takeover on timeout (--mode hybrid)
  --stagewise-report   each stage started from a scripted state (--stagewise)
  --export-dir         the export whose export_report.json gives the INT8 action error
Writes the matching fields of docs/slides/deck_data.json (nothing else); --print-only leaves it alone.

Run:  .venv\\Scripts\\python.exe -m tools.fill_results --policy-report out/v3_80k_rollout_policy_ens_seeds0-9.json ^
        --hybrid-report out/v3_80k_rollout_hybrid_ens_seeds0-9.json ^
        --stagewise-report out/v3_80k_rollout_stagewise_ens_seeds0-9.json --export-dir models/policy_v3_80k
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECK_DATA = ROOT / "docs" / "slides" / "deck_data.json"
# Order of env.success() sub-goals as shown in README and deck_data.json (not plan order).
SUBGOAL_KEYS = ["drawer_open", "mug_placed", "plate_placed", "fork_placed", "spoon_placed", "poured",
                "bottle_returned"]
SUBGOAL_NAMES = ["Drawer opened", "Mug placed", "Plate on placemat", "Fork placed", "Spoon handed over + placed",
                 "Poured (≥ 60 % of the water)", "Bottle put back"]
# Which plan stage produces each sub-goal (for the hybrid "policy-solved + assisted" cells).
SUBGOAL_STAGE = {"drawer_open": "open_drawer", "mug_placed": "pick_place:mug", "plate_placed": "bimanual_place:plate",
                 "fork_placed": "pick_place:fork", "spoon_placed": "handoff:spoon", "poured": "pour:mug",
                 "bottle_returned": "return:bottle"}


def load(path):
    path = path if Path(path).is_absolute() else ROOT / path
    if not path.exists():
        raise SystemExit(f"no report at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def counts(report):
    """Sub-goal successes out of the seeds in a chained rollout report."""
    if not report:
        return None, None
    seeds = len(report["seeds"])
    fractions = report["summary"]["subgoal_success"]
    return [round(fractions[key] * seeds) for key in SUBGOAL_KEYS], seeds


def hybrid_cells(report):
    """Per sub-goal: 'final count (assisted)' - how many runs ended with it done, and in how many
    the script had to finish the stage that produces it.

    A takeover can also undo earlier work (knock a placed mug), so the final count is what the
    table shows, not the number of stages solved.
    """
    if not report:
        return None
    seeds = len(report["seeds"])
    cells = []
    for key in SUBGOAL_KEYS:
        final = round(report["summary"]["subgoal_success"][key] * seeds)
        assisted = sum(bool(stage["assisted"]) for episode in report["episodes"] for stage in episode["stages"]
                       if SUBGOAL_STAGE[key] in stage["stage"])
        cells.append(f"{final} ({assisted} assisted)")
    return cells


def fill(deck, policy, hybrid, export):
    """A copy of ``deck`` with the policy, hybrid and INT8 fields taken from the reports."""
    new = dict(deck)
    policy_counts, seeds = counts(policy)
    new["policy_10_seeds"] = policy_counts
    new["hybrid_10_seeds"] = hybrid_cells(hybrid)
    new["eval_seeds"] = seeds
    for key, rollout in (("policy_full_task", policy), ("hybrid_full_task", hybrid)):
        new[key] = round(rollout["summary"]["task_success_rate"] * len(rollout["seeds"])) if rollout else None
    if hybrid:
        new["hybrid_assisted_per_episode"] = round(hybrid["summary"]["assisted_stages_per_episode"], 1)
    if export:
        new["int8_action_error_rad"] = export["act_int8"]["mean_abs_action_error_rad"]
    return new


def build_parser():
    parser = argparse.ArgumentParser(description="Fill slide data and print README tables from evaluation JSON.")
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--hybrid-report", type=Path, required=True)
    parser.add_argument("--stagewise-report", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True, help="export whose INT8 action error to report")
    parser.add_argument("--print-only", action="store_true", help="print the tables without writing the deck data")
    return parser


def print_tables(deck, policy, hybrid, stagewise, export):
    seeds = deck["eval_seeds"] or 10
    print("\n| Sub-goal | Scripted | Learned policy | Hybrid |\n|---|---|---|---|")
    for i, name in enumerate(SUBGOAL_NAMES):
        pol = f"{deck['policy_10_seeds'][i]}/{seeds}" if deck["policy_10_seeds"] else "—"
        hyb = deck["hybrid_10_seeds"][i] if deck["hybrid_10_seeds"] else "—"
        print(f"| {name} | {deck['scripted_10_seeds'][i]}/10 | {pol} | {hyb} |")
    for label, report in (("Learned policy", policy), ("Hybrid", hybrid)):
        summary = report["summary"]
        ci = summary.get("task_success_ci95")  # reports written before the intervals have none
        interval = f" (95% {ci[0] * 100:.0f}–{ci[1] * 100:.0f}%)" if ci else ""
        print(f"\n{label}: full task {summary['task_success_rate'] * 100:.0f}%{interval} · "
              f"assisted stages per episode {summary['assisted_stages_per_episode']:.1f} · "
              f"policy call {summary['policy_infer_ms_mean']} ms")
    print("\n| Stage (started from a scripted state) | Policy success | 95% interval |\n|---|---|---|")
    intervals = stagewise.get("per_stage_ci95", {})
    for stage, rate in stagewise["per_stage_success"].items():
        ci = intervals.get(stage)
        span = f"{ci[0] * 100:.0f}–{ci[1] * 100:.0f}%" if ci else "—"
        print(f"| `{stage}` | {rate * 100:.0f}% | {span} |")
    print("\n| IR | Mean / max action error (rad) | Size (MB) |\n|---|---|---|")
    for name, row in export.items():
        print(f"| {name} | {row['mean_abs_action_error_rad']} / {row['max_abs_action_error_rad']} | {row['size_mb']} |")


def main():
    args = build_parser().parse_args()
    policy, hybrid, stagewise = load(args.policy_report), load(args.hybrid_report), load(args.stagewise_report)
    export = load(args.export_dir / "export_report.json")
    deck = fill(json.loads(DECK_DATA.read_text(encoding="utf-8")), policy, hybrid, export)
    if not args.print_only:
        DECK_DATA.write_text(json.dumps(deck, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"updated {DECK_DATA}")
    print_tables(deck, policy, hybrid, stagewise, export)


if __name__ == "__main__":
    main()
