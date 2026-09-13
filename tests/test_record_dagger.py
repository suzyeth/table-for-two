"""Pure parts of the DAgger-style corrective recording."""
from data.record import SUBTASK_VOCAB
from data.record_dagger import split_stage, takeover_stage, vocab_index


def test_split_cycles_over_the_middle_stages_only():
    splits = {split_stage(seed) for seed in range(500, 600)}
    assert splits == {1, 2, 5}  # empty-handed boundaries only; never stage 0 or the home stage
    assert split_stage(517) == split_stage(517)
    assert split_stage(7, splits=(3, 4)) == 4


def test_takeover_is_the_split_unless_the_policy_fails_earlier():
    assert takeover_stage(split=3, first_failed=None) == 3
    assert takeover_stage(split=3, first_failed=1) == 1
    assert takeover_stage(split=3, first_failed=5) == 3


def test_vocab_index_offsets_the_executor_label_by_the_takeover_stage():
    # Executor.run numbers stages relative to the plan it was given.
    assert vocab_index("0:pour", offset=3) == 3
    assert vocab_index("2:home+home", offset=4) == len(SUBTASK_VOCAB) - 1
