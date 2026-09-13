"""Caption text for demo video stages: every skill shape the planner can emit."""
from demo import stage_text


def test_single_arm_skill():
    assert stage_text([{"skill": "pick_place", "arm": "right", "object": "mug"}]) == "right: pick place mug"


def test_pour_uses_its_target():
    assert stage_text([{"skill": "pour", "arm": "right", "into": "mug"}]) == "right: pour mug"


def test_two_handed_carry_has_no_arm_key():
    assert stage_text([{"skill": "bimanual_place", "object": "plate"}]) == "both: bimanual place plate"


def test_handoff_names_giver_and_receiver():
    stage = [{"skill": "handoff", "object": "spoon", "giver": "left", "receiver": "right"}]
    assert stage_text(stage) == "hand spoon left -> right"


def test_parallel_subtasks_are_joined():
    stage = [{"skill": "open_drawer", "arm": "left"}, {"skill": "pick_place", "arm": "right", "object": "mug"}]
    assert stage_text(stage) == "left: open drawer  |  right: pick place mug"
