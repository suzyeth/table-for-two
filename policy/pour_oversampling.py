"""Show the pour frames more than once per training epoch.

Review 2: the pour was the policy's only true skill gap (2-3/10 even from a scripted start). LeRobot
can weight per-sample losses, but only through ``policy.forward(batch, reduction="none")``, which ACT
does not implement, and its ``EpisodeAwareSampler`` has no per-frame weights. So training swaps in a
sampler that repeats every pour frame ``factor`` times per epoch. In the new demos the pour stage is
~28% of the frames (821 of 2922 control steps); the default factor 2 makes it ~44% of what training sees.

``install(lerobot_train, dataset_root, factor)`` swaps the sampler class LeRobot's training script
builds, the way ``policy/train.py`` swaps its checkpoint pointer helper.
"""
import functools
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.sampler import EpisodeAwareSampler

from data.record import SUBTASK_VOCAB, stage_signature

POUR_STAGE = SUBTASK_VOCAB.index(stage_signature([{"skill": "pour", "arm": "right", "into": "mug"}]))
POUR_OVERSAMPLE = 2
# Every demo stage ends with a still hold (sim/task.py STAGE_END_HOLD_S: ~26 frames that say "stay
# put"), while a static arm starting to move shows up only in the first frames of the next stage. v3
# learned the hold and froze at the start of the pour (1/10 even from a scripted start, the arm all but
# still for the whole stage). Showing the first START_FRAMES of every stage START_OVERSAMPLE times
# rebalances the two.
START_FRAMES = 10
START_OVERSAMPLE = 5


class OversamplingSampler(EpisodeAwareSampler):
    """``EpisodeAwareSampler`` plus one extra visit of each of ``extra_indices`` (absolute frame indices) per epoch.

    Copies of frames outside the episodes in use (e.g. the held-out validation episodes) are dropped.
    Shuffling, reproducibility and mid-epoch resume work exactly as in the parent: the positions past
    its own frames map to the extra copies.
    """

    def __init__(self, *args, extra_indices=(), **kwargs):
        super().__init__(*args, **kwargs)
        extras = np.asarray(extra_indices, dtype=np.int64)
        lengths = np.diff(np.concatenate([[0], self._cum_lengths]))
        episode = np.clip(np.searchsorted(self._starts, extras, side="right") - 1, 0, None)
        inside = (extras >= self._starts[episode]) & (extras < self._starts[episode] + lengths[episode])
        self._extras = extras[inside]
        self._base_frames = self._num_frames
        self._num_frames = self._base_frames + len(self._extras)

    def _frame_index(self, position):
        if position < self._base_frames:
            return super()._frame_index(position)
        absolute = int(self._extras[position - self._base_frames])
        if self._absolute_to_relative is not None:
            return self._absolute_to_relative[absolute]
        return absolute


def pour_frame_indices(dataset_root, stage=POUR_STAGE):
    """Absolute indices of the frames recorded under ``stage`` (only two columns are read, not the images)."""
    found = []
    for path in sorted(Path(dataset_root).glob("data/**/*.parquet")):
        table = pq.read_table(path, columns=["index", "observation.environment_state"])
        one_hot = np.asarray(table.column("observation.environment_state").to_pylist(), dtype=float)
        index = np.asarray(table.column("index").to_pylist(), dtype=np.int64)
        found.append(index[one_hot[:, stage] > 0.5])
    return np.concatenate(found) if found else np.zeros(0, dtype=np.int64)


def stage_start_indices(dataset_root, frames=START_FRAMES):
    """Absolute indices of the first ``frames`` frames of every stage of every episode."""
    found = []
    for path in sorted(Path(dataset_root).glob("data/**/*.parquet")):
        table = pq.read_table(path, columns=["index", "episode_index", "observation.environment_state"])
        index = np.asarray(table.column("index").to_pylist(), dtype=np.int64)
        episode = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        stage = np.asarray(table.column("observation.environment_state").to_pylist(), dtype=float).argmax(axis=1)
        starts_run = np.ones(len(index), dtype=bool)
        starts_run[1:] = (stage[1:] != stage[:-1]) | (episode[1:] != episode[:-1])
        run_start = np.flatnonzero(starts_run)
        offset = np.arange(len(index)) - run_start[np.cumsum(starts_run) - 1]
        found.append(index[offset < frames])
    return np.concatenate(found) if found else np.zeros(0, dtype=np.int64)


def install(lerobot_train, dataset_root, factor=POUR_OVERSAMPLE, start_factor=1, start_frames=START_FRAMES):
    """Make LeRobot's training script build an ``OversamplingSampler`` that shows each pour frame ``factor``
    times and the first ``start_frames`` frames of every stage ``start_factor`` times."""
    parts = []
    if factor > 1:
        parts.append(np.repeat(pour_frame_indices(dataset_root), int(factor) - 1))
    if start_factor > 1:
        parts.append(np.repeat(stage_start_indices(dataset_root, start_frames), int(start_factor) - 1))
    if not parts:
        return
    extras = np.concatenate(parts)
    lerobot_train.EpisodeAwareSampler = functools.partial(OversamplingSampler, extra_indices=extras)
