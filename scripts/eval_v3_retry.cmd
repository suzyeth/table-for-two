@echo off
REM Re-score the low-scoring stages with per-stage time limits (see scripts\v3_retry_jobs.txt).
cd /d %~dp0..
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set HF_HUB_OFFLINE=1
echo ===== retry evaluations %date% %time% >> out\eval_v3_retry.log
.venv\Scripts\python.exe -m tools.run_parallel scripts\v3_retry_jobs.txt --jobs 3 --log-dir out\parallel_v3_retry >> out\eval_v3_retry.log 2>&1
echo ===== retry evaluations done %date% %time% >> out\eval_v3_retry.log
