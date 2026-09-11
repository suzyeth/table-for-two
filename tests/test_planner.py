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


def test_example_plan_matches_the_executor_default_plan():
    from sim.task import DEFAULT_PLAN
    assert EXAMPLE_PLAN == DEFAULT_PLAN


def test_parse_compact_plan_with_parallel_stage():
    plan = parse_plan_text("open_drawer left ; pick_place right mug\npick_lift right bottle")
    assert plan == [
        [{"skill": "open_drawer", "arm": "left"}, {"skill": "pick_place", "arm": "right", "object": "mug"}],
        [{"skill": "pick_lift", "arm": "right", "object": "bottle"}],
    ]


def test_synonyms_map_cup_to_mug():
    plan = parse_plan_text("pick_place right cup")
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


def test_semantics_flags_pour_without_bottle():
    messages = [m for _, _, m in check_semantics(parse_plan_text("pick_place right mug\npour right mug"))]
    assert any("pour needs the bottle" in m for m in messages)


def test_semantics_flags_pour_before_the_mug_is_in_place_and_unreturned_bottle():
    messages = [m for _, _, m in check_semantics(parse_plan_text("pick_lift right bottle\npour right mug"))]
    assert any("set the mug down" in m for m in messages)
    assert any("ends with the right arm holding the bottle" in m for m in messages)


def test_scene_state_counts_a_mug_already_in_place():
    plan = parse_plan_text("pick_lift right bottle\npour right mug\nreturn right bottle")
    assert check_semantics(plan, {"done": ["mug_placed"]}) == []


def test_bimanual_place_needs_both_hands_free_and_its_own_line():
    with pytest.raises(PlanValidationError):
        parse_plan_text("bimanual_place plate ; pick_place left fork")
    with pytest.raises(PlanValidationError):
        parse_plan_text("bimanual_place mug")
    messages = [m for _, _, m in check_semantics(parse_plan_text("pick_lift right bottle\nbimanual_place plate"))]
    assert any("still holding the bottle" in m for m in messages)


def test_keyword_fallback_carries_the_plate_with_both_hands():
    plan = keyword_plan("Put the plate on the placemat")
    assert [{"skill": "bimanual_place", "object": "plate"}] in plan
    assert check_semantics(plan) == []


def test_repair_drops_a_stray_place_step():
    plan = parse_plan_text("pick_place right plate\nplace right plate\nhome right")
    repaired = repair(plan)
    assert repaired == [[{"skill": "pick_place", "arm": "right", "object": "plate"}],
                        [{"skill": "home", "arm": "right"}]]


def test_scene_state_skips_the_drawer_when_already_open():
    problems = check_semantics(parse_plan_text("open_drawer left"), {"done": ["drawer_open"]})
    assert problems and "already open" in problems[0][2]


def test_keyword_fallback_pours_after_setting_the_mug():
    plan = keyword_plan("Pour some water into the cup")
    skills = [(s["skill"], s.get("object", s.get("into"))) for stage in plan for s in stage]
    assert skills.index(("pick_place", "mug")) < skills.index(("pour", "mug"))
    assert check_semantics(plan) == []


def test_keyword_fallback_hands_the_spoon_over_by_default():
    plan = keyword_plan("Set the spoon and the fork")
    assert plan[0][0]["skill"] == "open_drawer"
    assert any(s["skill"] == "handoff" and s["object"] == "spoon" for stage in plan for s in stage)
    assert check_semantics(plan) == []
