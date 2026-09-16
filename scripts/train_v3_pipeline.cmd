@echo off
REM Policy v3 on the current scene (prop friction 0.4, no MSAA, 24-D state, still hold at every
REM stage end, demo gate, free-space noise): record -> train -> export -> evaluations -> benchmark.
REM Log: out\train_v3_pipeline.log   Recording logs: data\contact_v3\logs\   Eval logs: out\parallel_v3\
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set PY=.venv\Scripts\python.exe
set LOG=out\train_v3_pipeline.log
set DATA=data\contact_v3\merged
set CKPT=outputs\act_contact_v3\checkpoints\040000\pretrained_model
echo ===== v3 pipeline start %date% %time% >> %LOG%

if exist data\contact_v3 ( echo DATA DIR data\contact_v3 ALREADY EXISTS - remove it first %date% %time% >> %LOG% & exit /b 1 )
if exist outputs\act_contact_v3 ( echo OUTPUT DIR outputs\act_contact_v3 ALREADY EXISTS - remove it first %date% %time% >> %LOG% & exit /b 1 )

echo ===== [A] record 280 demos, 8 shards %date% %time% >> %LOG%
%PY% -m tools.record_parallel --root data\contact_v3 --episodes 280 >> %LOG% 2>&1
if errorlevel 1 ( echo RECORD FAILED %date% %time% >> %LOG% & exit /b 1 )

echo ===== [B] train v3 40k %date% %time% >> %LOG%
%PY% -m policy.train --dataset-root %DATA% --output-dir outputs\act_contact_v3 --steps 40000 --num-workers 12 --amp >> %LOG% 2>&1
if errorlevel 1 ( echo TRAIN FAILED %date% %time% >> %LOG% & exit /b 1 )

echo ===== [C] export v3 %date% %time% >> %LOG%
%PY% -m policy.export_openvino --checkpoint %CKPT% --dataset-root %DATA% --out-dir models\policy_v3 >> %LOG% 2>&1
if errorlevel 1 ( echo EXPORT FAILED %date% %time% >> %LOG% & exit /b 1 )

echo ===== [D] evaluations in parallel %date% %time% >> %LOG%
%PY% -m tools.run_parallel scripts\v3_jobs.txt --jobs 3 --log-dir out\parallel_v3 >> %LOG% 2>&1

echo ===== [E] benchmark v3, alone %date% %time% >> %LOG%
%PY% -m bench.benchmark --policy-dir models\policy_v3 >> %LOG% 2>&1
echo ===== v3 pipeline done %date% %time% >> %LOG%
