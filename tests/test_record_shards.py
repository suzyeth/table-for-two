"""data/record.py: parallel recorders split the seeds so no two shards ever record the same scene."""
import pytest

from data.record import parse_shard, shard_seeds


def test_a_shard_takes_every_nth_seed_from_its_offset():
    assert shard_seeds(start=100, tries=4, shard=(2, 8)) == [102, 110, 118, 126]


def test_shards_are_disjoint_and_together_cover_the_range():
    shards = [shard_seeds(start=100, tries=50, shard=(k, 8)) for k in range(8)]
    seeds = [seed for shard in shards for seed in shard]
    assert len(seeds) == len(set(seeds)) == 400
    assert set(seeds) == set(range(100, 500))


def test_a_single_shard_is_the_plain_consecutive_range():
    assert shard_seeds(start=100, tries=5, shard=(0, 1)) == [100, 101, 102, 103, 104]


@pytest.mark.parametrize("text, expected", [("0/1", (0, 1)), ("3/8", (3, 8))])
def test_parse_shard(text, expected):
    assert parse_shard(text) == expected


@pytest.mark.parametrize("text", ["8/8", "-1/4", "2", "a/b", "1/0"])
def test_parse_shard_rejects_nonsense(text):
    with pytest.raises(ValueError):
        parse_shard(text)
