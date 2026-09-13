"""tools/run_parallel.py: job-file parsing and bounded concurrent execution."""
import sys
import time

import pytest

from tools.run_parallel import parse_jobs, run_all

PY = f'"{sys.executable}"'


def test_parse_skips_comments_and_blank_lines():
    jobs = parse_jobs("# header\n\na | echo one\n  b|echo two  \n")
    assert jobs == [("a", "echo one"), ("b", "echo two")]


def test_parse_rejects_missing_separator_and_duplicates():
    with pytest.raises(ValueError):
        parse_jobs("just a command")
    with pytest.raises(ValueError):
        parse_jobs("a | echo 1\na | echo 2")


def test_jobs_run_concurrently_and_report_exit_codes(tmp_path):
    sleep = f'{PY} -c "import time; time.sleep(1.5)"'
    fail = f'{PY} -c "import sys; print(42); sys.exit(3)"'
    started = time.perf_counter()
    results = run_all([("s1", sleep), ("s2", sleep), ("bad", fail)], max_parallel=3, log_dir=tmp_path, poll_s=0.1)
    elapsed = time.perf_counter() - started
    assert elapsed < 2.9  # three jobs side by side, not 1.5 + 1.5 + ...
    assert results["s1"][0] == 0 and results["s2"][0] == 0 and results["bad"][0] == 3
    assert "42" in (tmp_path / "bad.log").read_text(encoding="utf-8")


def test_max_parallel_one_runs_in_sequence(tmp_path):
    sleep = f'{PY} -c "import time; time.sleep(0.6)"'
    started = time.perf_counter()
    run_all([("a", sleep), ("b", sleep)], max_parallel=1, log_dir=tmp_path, poll_s=0.05)
    assert time.perf_counter() - started >= 1.2
