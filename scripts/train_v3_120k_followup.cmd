@echo off
REM After the 120k continuation: export and score the low-scoring stages plus the chained run.
REM Log: out\train_v3_120k_followup.log   Per-job logs: out\parallel_v3_120k\
cd /d %~dp0..
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set HF_HUB_OFFLINE=1
set LOG=out\train_v3_120k_followup.log
set STEP=outputs\act_contact_v3\checkpoints\120000
echo ===== waiting for the 120k checkpoint %date% %time% >> %LOG%
:wait
if exist "%STEP%\training_state\training_step.json" goto ready
timeout /t 60 /nobreak >nul
goto wait
:ready
timeout /t 30 /nobreak >nul
echo ===== export v3 at 120k %date% %time% >> %LOG%
.venv\Scripts\python.exe -m policy.export_openvino --checkpoint %STEP%\pretrained_model --dataset-root data\contact_v3\merged --out-dir models\policy_v3_120k >> %LOG% 2>&1
if errorlevel 1 ( echo EXPORT FAILED %date% %time% >> %LOG% & exit /b 1 )
echo ===== evaluations in parallel %date% %time% >> %LOG%
.venv\Scripts\python.exe -m tools.run_parallel scripts\v3_120k_jobs.txt --jobs 4 --log-dir out\parallel_v3_120k >> %LOG% 2>&1
echo ===== 120k follow-up done %date% %time% >> %LOG%
