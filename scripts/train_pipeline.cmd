@echo off
REM Contact-physics policy pipeline, run detached (Task Scheduler) so it survives editor/agent restarts.
REM   train ACT (40k steps, ~8 h on RTX 4060) -> OpenVINO export -> rollouts -> stage head -> benchmark
REM Log: out\train_pipeline.log
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set PY=.venv\Scripts\python.exe
set LOG=out\train_pipeline.log
echo ===== pipeline start %date% %time% >> %LOG%

echo ===== [1/9] train >> %LOG%
%PY% -m policy.train --steps 40000 >> %LOG% 2>&1
if errorlevel 1 ( echo TRAIN FAILED %date% %time% >> %LOG% & exit /b 1 )

echo ===== [2/9] export_openvino %date% %time% >> %LOG%
%PY% -m policy.export_openvino >> %LOG% 2>&1
if errorlevel 1 ( echo EXPORT FAILED %date% %time% >> %LOG% & exit /b 1 )

echo ===== [3/9] rollout policy fp32 %date% %time% >> %LOG%
%PY% -m policy.rollout --mode policy --policy models\policy\act_fp32.xml --out out\rollout_policy_fp32.json >> %LOG% 2>&1
echo ===== [4/9] rollout hybrid fp32 %date% %time% >> %LOG%
%PY% -m policy.rollout --mode hybrid --policy models\policy\act_fp32.xml --out out\rollout_hybrid_fp32.json >> %LOG% 2>&1
echo ===== [5/9] rollout stagewise fp32 %date% %time% >> %LOG%
%PY% -m policy.rollout --stagewise --policy models\policy\act_fp32.xml --out out\rollout_stagewise_fp32.json >> %LOG% 2>&1
echo ===== [6/9] rollout policy int8 %date% %time% >> %LOG%
%PY% -m policy.rollout --mode policy --policy models\policy\act_int8.xml --out out\rollout_policy_int8.json >> %LOG% 2>&1

echo ===== [7/9] stage head train + export %date% %time% >> %LOG%
%PY% -m policy.stage_head train >> %LOG% 2>&1
%PY% -m policy.stage_head export >> %LOG% 2>&1
echo ===== [8/9] rollout policy fp32, head switch %date% %time% >> %LOG%
%PY% -m policy.rollout --mode policy --switch head --policy models\policy\act_fp32.xml --out out\rollout_policy_fp32_head.json >> %LOG% 2>&1

echo ===== [9/9] benchmark %date% %time% >> %LOG%
%PY% -m bench.benchmark >> %LOG% 2>&1
echo ===== pipeline done %date% %time% >> %LOG%
