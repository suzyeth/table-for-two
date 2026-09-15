"""policy/pour_oversampling.py: the pour frames are seen more than once per training epoch.

Review 2: the pour was the policy's only true skill gap (2-3/10 from a scripted start). LeRobot's
EpisodeAwareSampler has no per-frame weighting (its loss weighting needs a per-sample forward ACT
does not have), so training swaps in a sampler that repeats the pour frames.
"""
import collections
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from policy.pour_oversampling import POUR_STAGE, OversamplingSampler, install, pour_frame_indices

FROM, TO = [0, 10, 20], [10, 20, 30]  # three 10-frame episodes


def counts(sampler):
    return collections.Counter(iter(sampler))


def test_every_frame_once_and_the_extra_copies_on_top_each_epoch():
    sampler = OversamplingSampler(FROM, TO, shuffle=True, seed=1, extra_indices=[3, 3, 14])
    assert len(sampler) == 33
    seen = counts(sampler)
    assert sum(seen.values()) == 33 and set(seen) == set(range(30))
    assert seen[3] == 3 and seen[14] == 2 and seen[0] == 1


def test_extra_copies_of_held_out_episodes_are_not_trained_on():
    sampler = OversamplingSampler(FROM, TO, episode_indices_to_use=[0, 1], shuffle=True, seed=1, extra_indices=[3, 25])
    seen = counts(sampler)
    assert 25 not in seen and seen[3] == 2 and len(sampler) == 21


def test_the_order_is_reproducible_and_resumes_mid_epoch():
    full = list(iter(OversamplingSampler(FROM, TO, shuffle=True, seed=7, extra_indices=[3, 14])))
    resumed = OversamplingSampler(FROM, TO, shuffle=True, seed=7, extra_indices=[3, 14])
    resumed.load_state_dict({"epoch": 0, "start_index": 12})
    assert list(iter(resumed)) == full[12:]


def write_frames(root, stages):
    folder = root / "data" / "chunk-000"
    folder.mkdir(parents=True)
    one_hots = [[1.0 if i == stage else 0.0 for i in range(7)] for stage in stages]
    pq.write_table(pa.table({"index": list(range(len(stages))), "observation.environment_state": one_hots}),
                   folder / "file-000.parquet")


def test_the_pour_frames_are_found_from_the_stage_one_hot(tmp_path):
    write_frames(tmp_path, [0, 0, POUR_STAGE, POUR_STAGE, 5])
    assert pour_frame_indices(tmp_path).tolist() == [2, 3]


def test_install_makes_lerobot_build_the_oversampling_sampler(tmp_path):
    write_frames(tmp_path, [0] * 10 + [POUR_STAGE] * 5 + [0] * 15)
    module = SimpleNamespace(EpisodeAwareSampler=None)
    install(module, tmp_path, factor=3)
    sampler = module.EpisodeAwareSampler(FROM, TO, shuffle=True, seed=0)
    assert isinstance(sampler, OversamplingSampler) and len(sampler) == 30 + 5 * 2


def test_a_factor_of_one_leaves_lerobot_alone(tmp_path):
    module = SimpleNamespace(EpisodeAwareSampler="original")
    install(module, tmp_path, factor=1)
    assert module.EpisodeAwareSampler == "original"
