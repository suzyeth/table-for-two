"""Unit tests for plan parsing, validation, repair and the keyword fallback (no LLM needed)."""
import pytest

from planner.planner import (
    EXAMPLE_PLAN,
    PlanValidationError,
    check_semantics,
    keyword_plan,
    parse_plan_text,
    repair,
)


def test_example_plan_is_semantically_clean():
    assert check_semantics(EXAMPLE_PLAN) == []


def test_parse_compact_plan_with_parallel_stage():
    plan = parse_plan_text("open_drawer left\npick_hold left mug ; pick_lift right bottle")
    assert plan == [
        [{"skill": "open_drawer", "arm": "left"}],
        [{"skill": "pick_hold", "arm": "left", "object": "mug"},
         {"skill": "pick_lift", "arm": "right", "object": "bottle"}],
    ]


def test_synonyms_map_cup_to_mug():
    plan = parse_plan_text("pick_hold right cup")
    assert plan[0][0]["object"] == "mug"


def test_pour_naming_the_bottle_is_redirected_to_the_mug():
    plan = parse_plan_text("pour right bottle")
    assert plan[0][0]["into"] == "mug"


@pytest.mark.parametrize("text", [
    "fly left",                               # unknown skill
    "pick_place left",                        # missing argument
    "pick_place left spoon ; home left",      # two subtasks for one arm
    "handoff fork left left",                 # giver == receiver
    "handoff plate left right",               # only utensils can be handed off
    "pick_place middle plate",                # unknown arm
])
def test_malformed_plans_are_rejected(text):
    with pytest.raises(PlanValidationError):
        parse_plan_text(text)


def test_semantics_flags_utensil_before_drawer():
    problems = check_semantics(parse_plan_text("pick_place left spoon\nhome left"))
    assert any("open the drawer" in message for _, _, message in problems)


def test_semantics_flags_pour_without_bottle_and_unfinished_hold():
    plan = parse_plan_text("pick_hold left mug\npour right mug")
    messages = [message for _, _, message in check_semantics(plan)]
    assert any("pour needs the bottle" in m for m in messages)
    assert any("ends with the left arm holding the mug" in m for m in messages)


def test_repair_drops_a_stray_place_step():
    plan = parse_plan_text("pick_place right plate\nplace right plate\nhome right")
    repaired = repair(plan)
    assert repaired == [[{"skill": "pick_place", "arm": "right", "object": "plate"}],
                        [{"skill": "home", "arm": "right"}]]


def test_scene_state_skips_the_drawer_when_already_open():
    problems = check_semantics(parse_plan_text("open_drawer left"), {"done": ["drawer_open"]})
    assert problems and "already open" in problems[0][2]


def test_keyword_fallback_honours_named_arms_for_pouring():
    plan = keyword_plan("Hold the cup with the right arm and pour with the left arm")
    hold = next(s for stage in plan for s in stage if s["skill"] == "pick_hold")
    pour = next(s for stage in plan for s in stage if s["skill"] == "pour")
    assert (hold["arm"], pour["arm"]) == ("right", "left")
    assert check_semantics(plan) == []


def test_keyword_fallback_opens_drawer_before_utensils():
    plan = keyword_plan("Set the spoon and the fork")
    assert plan[0][0]["skill"] == "open_drawer"
    assert check_semantics(plan) == []
