@echo off
REM Policy v2, restarted with the speed-ups (9/13): optional AMP training, then all
REM evaluations side by side (tools/run_parallel.py, 3 at a time), benchmark last and alone.
REM Log: out\train_v2b_pipeline.log   Per-job logs: out\parallel_v2\
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set PY=.venv\Scripts\python.exe
set LOG=out\train_v2b_pipeline.log
set CKPT=outputs\act_contact_v2\checkpoints\060000\pretrained_model
REM Set by the AMP comparison: "--amp" if mixed precision won, empty otherwise.
set AMP=--amp
echo ===== v2b pipeline start %date% %time% AMP=[%AMP%] >> %LOG%

if exist outputs\act_contact_v2 ( echo OUTPUT DIR outputs\act_contact_v2 STILL EXISTS - remove it first %date% %time% >> %LOG% & exit /b 1 )
echo ===== [C] train v2 60k %date% %time% >> %LOG%
%PY% -m policy.train --dataset-root data\dinner_table_contact_300 --output-dir outputs\act_contact_v2 --steps 60000 %AMP% >> %LOG% 2>&1
if errorlevel 1 ( echo TRAIN FAILED %date% %time% >> %LOG% & exit /b 1 )

echo ===== [D] export v2 %date% %time% >> %LOG%
%PY% -m policy.export_openvino --checkpoint %CKPT% --dataset-root data\dinner_table_contact_300 --out-dir models\policy_v2 >> %LOG% 2>&1
if errorlevel 1 ( echo EXPORT FAILED %date% %time% >> %LOG% & exit /b 1 )
echo ===== [D2] export v2 INT8 accuracy-control %date% %time% >> %LOG%
%PY% -m policy.export_openvino --checkpoint %CKPT% --dataset-root data\dinner_table_contact_300 --out-dir models\policy_v2_ac --accuracy-control --max-drop 0.02 >> %LOG% 2>&1

echo ===== [E] evaluations in parallel %date% %time% >> %LOG%
%PY% -m tools.run_parallel scripts\v2_jobs.txt --jobs 3 --log-dir out\parallel_v2 >> %LOG% 2>&1

echo ===== [F] rollout with the learned stage switch %date% %time% >> %LOG%
%PY% -m policy.rollout --mode policy --switch head --policy models\policy_v2\act_fp32.xml --checkpoint %CKPT% --out out\v2_rollout_policy_fp32_head.json >> %LOG% 2>&1

echo ===== [G] benchmark v2, alone %date% %time% >> %LOG%
%PY% -m bench.benchmark --policy-dir models\policy_v2 >> %LOG% 2>&1
echo ===== v2b pipeline done %date% %time% >> %LOG%
