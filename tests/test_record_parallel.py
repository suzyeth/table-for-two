"""tools/record_parallel.py: one recording split across shards that stay in the plain-demo seed range."""
import pytest

from tools.record_parallel import plan_shards


def test_the_episodes_are_split_evenly_and_add_up():
    shards = plan_shards(episodes=300, shards=8, start_seed=100, seed_limit=500, root="data/x")
    assert [s["episodes"] for s in shards] == [38, 38, 38, 38, 37, 37, 37, 37]
    assert sum(s["episodes"] for s in shards) == 300


def test_every_shard_stays_below_the_seed_limit():
    shards = plan_shards(episodes=300, shards=8, start_seed=100, seed_limit=500, root="data/x")
    for shard in shards:
        assert shard["start_seed"] == 100 and shard["max_tries"] == 50
        assert 100 + shard["index"] + 8 * (shard["max_tries"] - 1) < 500


def test_each_shard_records_its_own_seeds_into_its_own_root():
    shards = plan_shards(episodes=16, shards=2, start_seed=100, seed_limit=500, root="data/x")
    assert shards[1]["root"].endswith("shard_1")
    command = shards[1]["command"]
    for part in ("-m data.record", "--episodes 8", "--start-seed 100", "--shard 1/2", "--max-tries 200",
                 "--overwrite", "shard_1"):
        assert part in command


def test_a_seed_range_too_small_for_the_episodes_is_refused():
    with pytest.raises(ValueError):
        plan_shards(episodes=300, shards=8, start_seed=100, seed_limit=400, root="data/x")


def test_the_margin_allows_for_the_gate_keeping_only_4_in_5_seeds():
    """At prop friction 0.4 the demo gate kept 16 of 20 seeds (the two-handed plate pinch)."""
    with pytest.raises(ValueError):
        plan_shards(episodes=330, shards=8, start_seed=100, seed_limit=500, root="data/x")
    assert sum(s["episodes"] for s in plan_shards(280, 8, 100, 500, "data/x")) == 280
