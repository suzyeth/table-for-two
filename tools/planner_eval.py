"""How reliably does the language planner understand varied instructions?

Runs a set of paraphrased instructions (different wording, order, synonyms,
partial requests, named arms) through the OpenVINO planner and checks each
plan two ways:
  * valid   - parses, passes the syntax and hand-state checks (check_semantics);
  * correct - contains exactly the set of skills the instruction asks for
              (e.g. "pour a drink" must yield pick_lift/pour/return of the
              bottle and a placed mug, and nothing about the drawer).

Writes out/planner_eval.json and prints a table. Runs on CPU by default.

Run:  python -m tools.planner_eval [--device GPU]
"""
import argparse
import json
import time
from pathlib import Path

from planner.planner import Planner, check_semantics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "planner_eval.json"

# (instruction, set of (skill, object-or-None) that must appear, set of skills that must not)
CASES = [
    ("Carry the plate to the placemat with both hands, open the drawer and put the mug in its place, lay the fork, "
     "pour the water from the bottle into the mug, then hand the spoon from the left arm to the right arm.",
     {("bimanual_place", "plate"), ("open_drawer", None), ("pick_place", "mug"), ("pick_place", "fork"),
      ("pick_lift", "bottle"), ("pour", None), ("return", "bottle"), ("handoff", "spoon")}, set()),
    ("Set the table and pour me a drink.",
     {("bimanual_place", "plate"), ("open_drawer", None), ("pick_place", "mug"), ("pour", None)}, set()),
    ("Put the plate on the placemat.", {("bimanual_place", "plate")}, {"open_drawer", "pour", "pick_lift"}),
    ("Just pour some water into the cup.", {("pick_place", "mug"), ("pick_lift", "bottle"), ("pour", None),
                                            ("return", "bottle")}, {"open_drawer", "handoff", "bimanual_place"}),
    ("Take the fork out of the drawer and put it next to the plate.",
     {("open_drawer", None), ("pick_place", "fork")}, {"pour", "bimanual_place"}),
    ("Pass the spoon from the left hand to the right hand and set it down.",
     {("open_drawer", None), ("handoff", "spoon")}, {"pour", "bimanual_place"}),
    ("Open the drawer.", {("open_drawer", None)}, {"pour", "handoff", "pick_place", "bimanual_place"}),
    ("Put the cup where it belongs.", {("pick_place", "mug")}, {"open_drawer", "pour"}),
    ("Fill the mug from the bottle, then put the bottle back where it was.",
     {("pick_place", "mug"), ("pick_lift", "bottle"), ("pour", None), ("return", "bottle")}, {"open_drawer"}),
    ("Lay out the cutlery: fork on the left, spoon on the right.",
     {("open_drawer", None), ("pick_place", "fork")}, {"pour", "bimanual_place"}),
    ("Use both arms to move the dish onto the mat, then give me a glass of water.",
     {("bimanual_place", "plate"), ("pick_place", "mug"), ("pour", None)}, set()),
    ("Hand the spoon over to the right arm.", {("open_drawer", None), ("handoff", "spoon")}, {"pour"}),
    ("Pour water into the glass and then open the drawer.", {("pour", None), ("open_drawer", None)}, set()),
    ("Set the dish on the placemat and put the fork beside it.",
     {("bimanual_place", "plate"), ("open_drawer", None), ("pick_place", "fork")}, {"pour"}),
    ("Drawer, mug, plate, fork, drink, spoon - the whole table please.",
     {("open_drawer", None), ("pick_place", "mug"), ("bimanual_place", "plate"), ("pick_place", "fork"),
      ("pour", None), ("handoff", "spoon")}, set()),
    ("Could you get a spoon for me? The drawer is closed.", {("open_drawer", None)}, {"pour", "bimanual_place"}),
    ("Serve water.", {("pick_place", "mug"), ("pour", None)}, {"open_drawer", "handoff"}),
    ("Place the plate, then pour, then pass the spoon across.",
     {("bimanual_place", "plate"), ("pour", None), ("handoff", "spoon")}, set()),
    ("Open the drawer with the left arm and put the cup down with the right arm.",
     {("open_drawer", None), ("pick_place", "mug")}, {"pour", "bimanual_place"}),
    ("I'd like the fork laid on the left of the plate and a drink poured.",
     {("pick_place", "fork"), ("pour", None)}, {"handoff"}),
]


def skills_in(plan):
    found = set()
    for stage in plan:
        for sub in stage:
            skill = sub["skill"]
            obj = sub.get("object") if skill in ("pick_place", "bimanual_place", "handoff", "pick_lift", "return") else None
            found.add((skill, obj))
    return found


def judge(plan, required, forbidden):
    found = skills_in(plan)
    names = {skill for skill, _ in found}
    missing = {r for r in required if r not in found}
    extra = {skill for skill in forbidden if skill in names}
    return not missing and not extra, sorted(missing), sorted(extra)


def main():
    parser = argparse.ArgumentParser(description="Planner instruction-understanding evaluation.")
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args()
    planner = Planner(device=args.device)
    rows, valid, correct, llm_source, latency = [], 0, 0, 0, []
    for text, required, forbidden in CASES:
        started = time.perf_counter()
        plan, info = planner.plan(text)
        latency.append(time.perf_counter() - started)
        is_valid = not check_semantics(plan)
        ok, missing, extra = judge(plan, required, forbidden)
        valid += is_valid
        correct += ok and is_valid
        llm_source += info["source"].startswith("llm")
        rows.append({"instruction": text, "valid": is_valid, "correct": bool(ok and is_valid), "missing": missing,
                     "unwanted": extra, "source": info["source"], "attempts": info["attempts"],
                     "latency_s": round(latency[-1], 2), "plan": plan})
        mark = "OK " if ok and is_valid else "BAD"
        print(f"{mark} [{info['source']:14s} {latency[-1]:4.1f}s] {text[:70]}"
              + (f"  missing={missing} unwanted={extra}" if not ok else ""))
    summary = {"cases": len(CASES), "valid": valid, "correct": correct, "from_llm": llm_source,
               "mean_latency_s": round(sum(latency) / len(latency), 2), "device": args.device}
    print(json.dumps(summary))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
