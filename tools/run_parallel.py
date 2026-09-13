"""Run independent evaluation commands concurrently and wait for all of them.

Each line of the job file is ``name | command``; blank lines and lines starting
with ``#`` are ignored. Every job's output goes to ``<log-dir>/<name>.log``. At
most ``--jobs`` run at once. Exits non-zero if any job failed, after all of
them have finished, so one bad rollout does not hide the others' results.

Rollouts are step-driven, not wall-clock-driven, so running them side by side
does not change any success rate; only the per-call latency each report records
is inflated by the shared CPU. Latency figures come from ``bench/benchmark.py``,
which runs alone.

Run:  .venv\\Scripts\\python.exe -m tools.run_parallel scripts\\v2_rollouts.txt --jobs 3 --log-dir out\\parallel
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_jobs(text):
    """``name | command`` lines -> list of (name, command)."""
    jobs = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise ValueError(f"job line needs 'name | command': {line!r}")
        name, command = (part.strip() for part in line.split("|", 1))
        if not name or not command:
            raise ValueError(f"empty name or command: {line!r}")
        jobs.append((name, command))
    names = [name for name, _ in jobs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate job names: {names}")
    return jobs


def run_all(jobs, max_parallel, log_dir, poll_s=1.0):
    """Run ``jobs`` with at most ``max_parallel`` alive; return {name: (exit_code, seconds)}."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pending, running, results = list(jobs), {}, {}
    while pending or running:
        while pending and len(running) < max_parallel:
            name, command = pending.pop(0)
            log = open(log_dir / f"{name}.log", "w", encoding="utf-8")
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, shell=True)
            running[name] = (proc, log, time.perf_counter())
            print(f"started {name}", flush=True)
        for name in list(running):
            proc, log, started = running[name]
            code = proc.poll()
            if code is None:
                continue
            log.close()
            results[name] = (code, time.perf_counter() - started)
            del running[name]
            print(f"finished {name}: exit {code} after {results[name][1] / 60:.1f} min", flush=True)
        if running:
            time.sleep(poll_s)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run independent commands concurrently.")
    parser.add_argument("job_file", type=Path)
    parser.add_argument("--jobs", type=int, default=3, help="maximum number of commands running at once")
    parser.add_argument("--log-dir", type=Path, default=Path("out") / "parallel")
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    jobs = parse_jobs(args.job_file.read_text(encoding="utf-8"))
    started = time.perf_counter()
    results = run_all(jobs, args.jobs, args.log_dir)
    failed = [name for name, (code, _) in results.items() if code != 0]
    print(f"all {len(results)} jobs done in {(time.perf_counter() - started) / 60:.1f} min; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
