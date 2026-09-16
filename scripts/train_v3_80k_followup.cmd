@echo off
REM Everything that follows the 40k -> 80k continuation (out\train_v3_resume.log): wait for the 80k
REM checkpoint, export it, score it on the brief's seeds 0-9 (policy, settle, hybrid, stagewise, INT8,
REM stage head) and on the 30 unseen seeds, then the accuracy-controlled INT8 export and its rollout.
REM The 40k model's own INT8 came out at 0.230 rad worst-case action error (gate: 0.03).
REM Log: out\train_v3_80k_followup.log   Per-job logs: out\parallel_v3_80k\
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set PY=.venv\Scripts\python.exe
set LOG=out\train_v3_80k_followup.log
set DATA=data\contact_v3\merged
set CKPT=outputs\act_contact_v3\checkpoints\080000\pretrained_model
echo ===== waiting for the 80k checkpoint %date% %time% >> %LOG%

:wait
if exist "%CKPT%\train_config.json" goto ready
timeout /t 60 /nobreak >nul
goto wait

:ready
echo ===== [X1] export v3 at 80k %date% %time% >> %LOG%
%PY% -m policy.export_openvino --checkpoint %CKPT% --dataset-root %DATA% --out-dir models\policy_v3_80k >> %LOG% 2>&1
if errorlevel 1 ( echo EXPORT FAILED %date% %time% >> %LOG% & exit /b 1 )

echo ===== [X2] evaluations in parallel %date% %time% >> %LOG%
%PY% -m tools.run_parallel scripts\v3_80k_jobs.txt --jobs 3 --log-dir out\parallel_v3_80k >> %LOG% 2>&1

echo ===== [X3] INT8 with accuracy control %date% %time% >> %LOG%
%PY% -m policy.export_openvino --checkpoint %CKPT% --dataset-root %DATA% --out-dir models\policy_v3_80k_ac --accuracy-control --max-drop 0.02 >> %LOG% 2>&1
if errorlevel 1 ( echo AC EXPORT FAILED %date% %time% >> %LOG% & exit /b 1 )
%PY% -m policy.rollout --mode policy --ensemble 0.01 --seeds 0 1 2 3 4 5 6 7 8 9 --policy models\policy_v3_80k_ac\act_int8.xml --checkpoint %CKPT% --out out\v3_80k_rollout_policy_int8_ac_ens_seeds0-9.json >> %LOG% 2>&1

echo ===== [X4] benchmark on the 80k export, alone %date% %time% >> %LOG%
%PY% -m bench.benchmark --policy-dir models\policy_v3_80k >> %LOG% 2>&1
echo ===== v3 80k follow-up done %date% %time% >> %LOG%
