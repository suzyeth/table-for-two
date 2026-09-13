@echo off
REM Policy v2 (300 demos, 60k steps) - fully unattended. Run detached via Task Scheduler.
REM   1. wait for run-1 pipeline -> temporal-ensemble rollout of v1 (R4)
REM   2. wait for the second recording -> aggregate 150+150 (R3)
REM   3. train 60k -> export (plain + accuracy-control INT8) -> rollouts -> stage head -> benchmark
REM Log: out\train_v2_pipeline.log
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set PY=.venv\Scripts\python.exe
set LOG=out\train_v2_pipeline.log
set CKPT=outputs\act_contact_v2\checkpoints\060000\pretrained_model
set CKPT1=outputs\act_contact\checkpoints\040000\pretrained_model
echo ===== v2 pipeline start %date% %time% >> %LOG%

:wait_run1
findstr /C:"pipeline done" out\train_pipeline.log >nul 2>&1
if errorlevel 1 ( ping -n 61 127.0.0.1 >nul & goto wait_run1 )
echo ===== [A] run 1 finished; R4 ensemble rollout on v1 %date% %time% >> %LOG%
%PY% -m policy.rollout --mode policy --ensemble 0.01 --policy models\policy\act_fp32.xml --checkpoint %CKPT1% --out out\rollout_policy_fp32_ens.json >> %LOG% 2>&1

:wait_record
findstr /C:"record_b done" out\record_b.log >nul 2>&1
if errorlevel 1 ( ping -n 61 127.0.0.1 >nul & goto wait_record )
if not exist data\dinner_table_contact_b\meta\episodes ( echo RECORD_B NOT FINALIZED %date% %time% >> %LOG% & exit /b 1 )
echo ===== [B] aggregate 150+150 %date% %time% >> %LOG%
%PY% -m tools.aggregate_contact --roots data\dinner_table_contact data\dinner_table_contact_b --out data\dinner_table_contact_300 --overwrite >> %LOG% 2>&1
if errorlevel 1 ( echo AGGREGATE FAILED %date% %time% >> %LOG% & exit /b 1 )

echo ===== [C] train v2 60k %date% %time% >> %LOG%
%PY% -m policy.train --dataset-root data\dinner_table_contact_300 --output-dir outputs\act_contact_v2 --steps 60000 >> %LOG% 2>&1
if errorlevel 1 ( echo TRAIN FAILED %date% %time% >> %LOG% & exit /b 1 )

echo ===== [D] export v2 %date% %time% >> %LOG%
%PY% -m policy.export_openvino --checkpoint %CKPT% --dataset-root data\dinner_table_contact_300 --out-dir models\policy_v2 >> %LOG% 2>&1
if errorlevel 1 ( echo EXPORT FAILED %date% %time% >> %LOG% & exit /b 1 )
echo ===== [D2] export v2 INT8 accuracy-control (R6) %date% %time% >> %LOG%
%PY% -m policy.export_openvino --checkpoint %CKPT% --dataset-root data\dinner_table_contact_300 --out-dir models\policy_v2_ac --accuracy-control --max-drop 0.02 >> %LOG% 2>&1

echo ===== [E] rollouts v2 %date% %time% >> %LOG%
%PY% -m policy.rollout --mode policy   --policy models\policy_v2\act_fp32.xml --checkpoint %CKPT% --out out\v2_rollout_policy_fp32.json >> %LOG% 2>&1
%PY% -m policy.rollout --mode hybrid   --policy models\policy_v2\act_fp32.xml --checkpoint %CKPT% --out out\v2_rollout_hybrid_fp32.json >> %LOG% 2>&1
%PY% -m policy.rollout --stagewise     --policy models\policy_v2\act_fp32.xml --checkpoint %CKPT% --out out\v2_rollout_stagewise_fp32.json >> %LOG% 2>&1
%PY% -m policy.rollout --mode policy   --policy models\policy_v2\act_int8.xml --checkpoint %CKPT% --out out\v2_rollout_policy_int8.json >> %LOG% 2>&1
%PY% -m policy.rollout --mode policy --ensemble 0.01 --policy models\policy_v2\act_fp32.xml --checkpoint %CKPT% --out out\v2_rollout_policy_fp32_ens.json >> %LOG% 2>&1
%PY% -m policy.rollout --stagewise --ensemble 0.01 --policy models\policy_v2\act_fp32.xml --checkpoint %CKPT% --out out\v2_rollout_stagewise_fp32_ens.json >> %LOG% 2>&1
if exist models\policy_v2_ac\act_int8.xml %PY% -m policy.rollout --mode policy --policy models\policy_v2_ac\act_int8.xml --checkpoint %CKPT% --out out\v2_rollout_policy_int8_ac.json >> %LOG% 2>&1

echo ===== [F] stage head v2 %date% %time% >> %LOG%
%PY% -m policy.stage_head train --dataset-root data\dinner_table_contact_300 >> %LOG% 2>&1
%PY% -m policy.stage_head export --dataset-root data\dinner_table_contact_300 >> %LOG% 2>&1
%PY% -m policy.rollout --mode policy --switch head --policy models\policy_v2\act_fp32.xml --checkpoint %CKPT% --out out\v2_rollout_policy_fp32_head.json >> %LOG% 2>&1

echo ===== [G] benchmark v2 %date% %time% >> %LOG%
%PY% -m bench.benchmark --policy-dir models\policy_v2 >> %LOG% 2>&1
echo ===== v2 pipeline done %date% %time% >> %LOG%
