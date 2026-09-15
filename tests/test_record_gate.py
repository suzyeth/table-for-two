"""data/record.py: an episode is kept only if it succeeded and the demo gate found nothing wrong."""
from data.record import skip_reason

SUCCESS = {"drawer_open": True, "poured": True, "all": True}


def test_a_successful_clean_episode_is_kept():
    assert skip_reason(SUCCESS, defects=[]) is None


def test_a_failed_episode_is_skipped_and_says_what_failed():
    reason = skip_reason({"drawer_open": True, "poured": False, "all": False}, defects=[])
    assert reason is not None and "poured" in reason and "drawer_open" not in reason


def test_a_successful_episode_with_defects_is_skipped_and_says_why():
    reason = skip_reason(SUCCESS, defects=["knocked mug in 3:pour"])
    assert reason is not None and "knocked mug in 3:pour" in reason
