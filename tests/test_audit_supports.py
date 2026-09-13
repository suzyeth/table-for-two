"""tools/audit_contact.supports: the robot and the water beads never count as a support."""
from sim.env import DinnerTableEnv
from tools.audit_contact import robot_and_water_bodies, supports


def test_full_bottle_at_rest_is_supported_by_the_table_not_by_its_water():
    env = DinnerTableEnv(obs_cameras=())
    try:
        env.reset(0)
        env.hold(0.5)
        excluded = robot_and_water_bodies(env)
        beads = {int(b) for b in env.water_ids}
        assert beads <= excluded
        assert env._contact_bodies("bottle") & beads, "the beads should be touching the full bottle"
        found = supports(env, "bottle")
        assert found and not (found & excluded)
        names = {env.model.body(b).name for b in found}
        assert not any(n.startswith(("left_", "right_", "water_")) for n in names)
    finally:
        env.close()


def test_robot_links_are_excluded_but_props_are_not():
    env = DinnerTableEnv(obs_cameras=())
    try:
        names = {env.model.body(b).name for b in robot_and_water_bodies(env)}
        assert any(n.startswith("left_") for n in names) and any(n.startswith("right_") for n in names)
        excluded = robot_and_water_bodies(env)
        for prop in ("plate", "mug", "bottle", "spoon", "fork", "drawer"):
            assert env.body_ids[prop] not in excluded
    finally:
        env.close()
