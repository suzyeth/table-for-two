"""tools/fill_results.py: slide numbers come from the reports named on the command line.

It used to read fixed v1 file names (out/rollout_policy_fp32.json, ...), so the v3 reports could not
reach the deck without renaming files - the kind of silent default the review asked to remove.
"""
import pytest

from tools.fill_results import SUBGOAL_KEYS, build_parser, fill

STAGES = ["bimanual_place:plate", "open_drawer:|pick_place:mug", "pick_lift:bottle|pick_place:fork", "pour:mug",
          "return:bottle", "handoff:spoon"]


def report(subgoal, task=0.0, assisted=0.0, stage_flags=None):
    episodes = [{"stages": [{"stage": s, "policy_ok": ok, "assisted": helped} for s, (ok, helped) in
                            zip(STAGES, flags)]} for flags in (stage_flags or [])]
    return {"seeds": list(range(10)), "episodes": episodes,
            "summary": {"subgoal_success": subgoal, "task_success_rate": task,
                        "assisted_stages_per_episode": assisted, "policy_infer_ms_mean": 100.0}}


def test_the_report_paths_have_to_be_given():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    args = build_parser().parse_args(["--policy-report", "p.json", "--hybrid-report", "h.json",
                                      "--stagewise-report", "s.json", "--export-dir", "models/m"])
    assert args.policy_report.name == "p.json" and args.export_dir.name == "m" and not args.print_only


def test_the_deck_gets_counts_out_of_the_seeds_and_the_int8_error():
    deck = {"scripted_10_seeds": [10] * 7, "title": "kept"}
    policy = report({k: (0.9 if k == "plate_placed" else 0.0) for k in SUBGOAL_KEYS}, task=0.1)
    hybrid = report({k: (0.7 if k == "plate_placed" else 1.0) for k in SUBGOAL_KEYS}, assisted=2.64,
                    stage_flags=[[(True, False), (True, False), (True, False), (False, True), (False, True),
                                  (False, True)]] * 10)
    export = {"act_int8": {"mean_abs_action_error_rad": 0.0058, "max_abs_action_error_rad": 0.44, "size_mb": 38.6}}
    new = fill(deck, policy, hybrid, export)
    assert new["policy_10_seeds"][SUBGOAL_KEYS.index("plate_placed")] == 9
    assert new["policy_full_task"] == 1 and new["hybrid_full_task"] == 0 and new["eval_seeds"] == 10
    # a hybrid cell is the sub-goal's final count, with how many of those runs the script finished that stage
    assert new["hybrid_10_seeds"][SUBGOAL_KEYS.index("poured")] == "10 (10 assisted)"
    assert new["hybrid_10_seeds"][SUBGOAL_KEYS.index("plate_placed")] == "7 (0 assisted)"
    assert new["hybrid_assisted_per_episode"] == 2.6 and new["int8_action_error_rad"] == 0.0058
    assert new["title"] == "kept" and "policy_10_seeds" not in deck  # the input is not changed
