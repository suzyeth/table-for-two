"""tools/val_loss.py: which checkpoints and which held-out frames are scored."""
from tools.val_loss import checkpoints, evenly_spaced


def test_frames_are_spread_over_all_held_out_episodes_not_taken_from_the_start():
    picked = evenly_spaced(20000, 5)
    assert picked == [0, 5000, 10000, 14999, 19999]
    assert len(set(evenly_spaced(20000, 3000))) == 3000


def test_fewer_frames_than_asked_are_all_scored():
    assert evenly_spaced(4, 10) == [0, 1, 2, 3]


def test_checkpoints_are_found_in_step_order_and_the_pointer_file_is_skipped(tmp_path):
    for step in ("040000", "005000", "010000"):
        (tmp_path / "checkpoints" / step / "pretrained_model").mkdir(parents=True)
    (tmp_path / "checkpoints" / "last_checkpoint.txt").write_text("040000", encoding="utf-8")
    assert [p.parent.name for p in checkpoints(tmp_path)] == ["005000", "010000", "040000"]
