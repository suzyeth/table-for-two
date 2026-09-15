"""policy/rollout.py: success rates come with a 95% confidence interval.

Review 2: the same checkpoint, ensembling setting and seeds scored the pour 8/10 in one run and
3/10 in another. With 10 scenes a single rate says little; a Wilson interval shows how little.
"""
import json
from types import SimpleNamespace

import pytest

from policy.rollout import summarise, wilson_interval


@pytest.mark.parametrize("k, n, low, high", [(5, 10, 0.2366, 0.7634), (0, 10, 0.0, 0.2775),
                                              (10, 10, 0.7225, 1.0), (27, 30, 0.7438, 0.9654)])
def test_wilson_interval_matches_hand_computed_values(k, n, low, high):
    lo, hi = wilson_interval(k, n)
    assert lo == pytest.approx(low, abs=5e-4) and hi == pytest.approx(high, abs=5e-4)


def test_no_episodes_says_nothing():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_the_summary_gives_an_interval_for_the_task_and_every_scored_stage():
    per_seed = [{"seed": s, "success": {"poured": s < 3, "all": False},
                 "stages": [{"stage": "pour:mug", "scored": True, "policy_ok": s < 3,
                             "policy_ok_from_clean_state": s < 3, "assisted": False}]} for s in range(10)]
    summary = summarise(per_seed, SimpleNamespace(device="CPU", infer_ms=[], compile_s=0.0))
    assert summary["policy_stage_success"]["pour:mug"] == pytest.approx(0.3)
    assert summary["policy_stage_success_ci95"]["pour:mug"] == pytest.approx(wilson_interval(3, 10))
    assert summary["task_success_ci95"] == pytest.approx(wilson_interval(0, 10))
    again = json.loads(json.dumps(summary))  # the report is written as JSON
    assert again["policy_stage_success_ci95"]["pour:mug"] == pytest.approx(list(wilson_interval(3, 10)))


def test_by_default_the_policy_is_scored_on_30_scenes_no_demo_was_recorded_on():
    """10 scenes gave a 95% interval ~0.5 wide; 30 bring it to ~0.3. Demos use 100-449, DAgger 500 on."""
    from data.record import FIRST_TRAIN_SEED
    from data.record_dagger import FIRST_DAGGER_SEED
    from policy.rollout import build_parser

    seeds = build_parser().parse_args(["--policy", "m.xml", "--checkpoint", "ckpt"]).seeds
    assert seeds == list(range(1000, 1030))
    assert FIRST_TRAIN_SEED < FIRST_DAGGER_SEED < min(seeds)
