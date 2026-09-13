@echo off
REM Finish run 1 after the 9/13 reboot: stage-head export (if missing), head-switch rollout, benchmark.
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set PY=.venv\Scripts\python.exe
set LOG=out\train_pipeline.log
echo ===== [7b/9] resume after reboot %date% %time% >> %LOG%
if not exist models\stage_head\stage_head_int8.xml %PY% -m policy.stage_head export >> %LOG% 2>&1
echo ===== [8/9] rollout policy fp32, head switch %date% %time% >> %LOG%
%PY% -m policy.rollout --mode policy --switch head --policy models\policy\act_fp32.xml --out out\rollout_policy_fp32_head.json >> %LOG% 2>&1
echo ===== [9/9] benchmark %date% %time% >> %LOG%
%PY% -m bench.benchmark >> %LOG% 2>&1
echo ===== pipeline done %date% %time% >> %LOG%
