@echo off
REM After the start-oversampled continuation: export the 100k checkpoint and score it on seeds 0-9.
REM Log: out\train_v3_100k_followup.log   Per-job logs: out\parallel_v3_100k\
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set PY=.venv\Scripts\python.exe
set LOG=out\train_v3_100k_followup.log
set CKPT=outputs\act_contact_v3\checkpoints\100000\pretrained_model
echo ===== waiting for the 100k checkpoint %date% %time% >> %LOG%
:wait
if exist "%CKPT%\train_config.json" goto ready
timeout /t 60 /nobreak >nul
goto wait
:ready
echo ===== [Y1] export v3 at 100k %date% %time% >> %LOG%
%PY% -m policy.export_openvino --checkpoint %CKPT% --dataset-root data\contact_v3\merged --out-dir models\policy_v3_100k >> %LOG% 2>&1
if errorlevel 1 ( echo EXPORT FAILED %date% %time% >> %LOG% & exit /b 1 )
echo ===== [Y2] evaluations in parallel %date% %time% >> %LOG%
%PY% -m tools.run_parallel scripts\v3_100k_jobs.txt --jobs 3 --log-dir out\parallel_v3_100k >> %LOG% 2>&1
echo ===== v3 100k follow-up done %date% %time% >> %LOG%
