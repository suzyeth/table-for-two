"""Copy evaluation results into the slide data and print the README results table.

Reads:
  out/rollout_policy_fp32.json      chained, learned policy only
  out/rollout_hybrid_fp32.json      chained, scripted takeover on timeout
  out/rollout_stagewise_fp32.json   each stage started from a scripted state
  models/<dir>/export_report.json   INT8 action error (--int8-dir picks which export)
Writes the matching fields of docs/slides/deck_data.json; nothing else is touched.

Run:  .venv\\Scripts\\python.exe -m tools.fill_results --int8-dir models/policy_ac
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
    path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def counts(report):
    """Sub-goal successes out of the seeds in a chained rollout report."""
    if not report:
        return None, None
    seeds = len(report["seeds"])
    fractions = report["summary"]["subgoal_success"]
    return [round(fractions[key] * seeds) for key in SUBGOAL_KEYS], seeds


def hybrid_cells(report):
    """Per sub-goal: 'policy-solved + assisted' counts of the stage that produces it.

    A hybrid run's sub-goal success is mostly the script's work, so the deck shows how
    many episodes the policy solved that stage itself and how many the script finished.
    """
    if not report:
        return None
    cells = []
    for key in SUBGOAL_KEYS:
        solved = assisted = 0
        for episode in report["episodes"]:
            stage = next((s for s in episode["stages"] if SUBGOAL_STAGE[key] in s["stage"]), None)
            if stage:
                solved += bool(stage["policy_ok"])
                assisted += bool(stage["assisted"])
        cells.append(f"{solved} + {assisted} assisted")
    return cells


def main():
    parser = argparse.ArgumentParser(description="Fill slide data and print README tables from evaluation JSON.")
    parser.add_argument("--int8-dir", default="models/policy", help="export directory whose INT8 error to report")
    args = parser.parse_args()

    policy = load("out/rollout_policy_fp32.json")
    hybrid = load("out/rollout_hybrid_fp32.json")
    stagewise = load("out/rollout_stagewise_fp32.json")
    export = load(f"{args.int8_dir}/export_report.json")

    deck = json.loads(DECK_DATA.read_text(encoding="utf-8"))
    policy_counts, seeds = counts(policy)
    hybrid_counts = hybrid_cells(hybrid)
    deck["policy_10_seeds"] = policy_counts
    deck["hybrid_10_seeds"] = hybrid_counts
    deck["eval_seeds"] = seeds
    for key, report in (("policy_full_task", policy), ("hybrid_full_task", hybrid)):
        deck[key] = round(report["summary"]["task_success_rate"] * len(report["seeds"])) if report else None
    if hybrid:
        deck["hybrid_assisted_per_episode"] = round(hybrid["summary"]["assisted_stages_per_episode"], 1)
    if export:
        deck["int8_action_error_rad"] = export["act_int8"]["mean_abs_action_error_rad"]
    DECK_DATA.write_text(json.dumps(deck, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"updated {DECK_DATA}")

    denominator = seeds or 10
    print("\n| Sub-goal | Scripted | Learned policy | Hybrid |\n|---|---|---|---|")
    for i, name in enumerate(SUBGOAL_NAMES):
        pol = f"{policy_counts[i]}/{denominator}" if policy_counts else "—"
        hyb = hybrid_counts[i] if hybrid_counts else "—"
        print(f"| {name} | {deck['scripted_10_seeds'][i]}/10 | {pol} | {hyb} |")
    for label, report in (("Learned policy", policy), ("Hybrid", hybrid)):
        if report:
            summary = report["summary"]
            print(f"\n{label}: full task {summary['task_success_rate'] * 100:.0f}% · "
                  f"assisted stages per episode {summary['assisted_stages_per_episode']:.1f} · "
                  f"policy call {summary['policy_infer_ms_mean']} ms")
    if stagewise:
        print("\n| Stage (started from a scripted state) | Policy success |\n|---|---|")
        for stage, rate in stagewise["per_stage_success"].items():
            print(f"| `{stage}` | {rate * 100:.0f}% |")
    if export:
        print("\n| IR | Mean / max action error (rad) | Size (MB) |\n|---|---|---|")
        for name, row in export.items():
            print(f"| {name} | {row['mean_abs_action_error_rad']} / {row['max_abs_action_error_rad']} | {row['size_mb']} |")


if __name__ == "__main__":
    main()
