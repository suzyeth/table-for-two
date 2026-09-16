@echo off
REM Stage-completion head for v3, queued behind scripts\train_v3_pipeline.cmd: waits until the v3
REM policy is trained and exported, trains + exports the head on the v3 demos (every stage now ends
REM with a 2.5 s still hold, the cue the v2 head lacked), then scores the policy with the head ending
REM each stage on the brief's seeds 0-9. Log: out\stage_head_v3.log
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set PY=.venv\Scripts\python.exe
set LOG=out\stage_head_v3.log
set PIPE=out\train_v3_pipeline.log
set DATA=data\contact_v3\merged
set CKPT=outputs\act_contact_v3\checkpoints\040000\pretrained_model
echo ===== waiting for the v3 policy export %date% %time% >> %LOG%

:wait
findstr /l /c:"RECORD FAILED" /c:"TRAIN FAILED" /c:"EXPORT FAILED" %PIPE% >nul && ( echo v3 PIPELINE FAILED - head not trained %date% %time% >> %LOG% & exit /b 1 )
findstr /l /c:"===== [D] evaluations" %PIPE% >nul && goto ready
timeout /t 60 /nobreak >nul
goto wait

:ready
REM stage_head.py always writes models\stage_head\ - keep the v2 head.
if exist models\stage_head if not exist models\stage_head_v2 xcopy /e /i /q models\stage_head models\stage_head_v2 >> %LOG%
echo ===== [H1] train head on v3 demos %date% %time% >> %LOG%
%PY% -m policy.stage_head train --dataset-root %DATA% >> %LOG% 2>&1
if errorlevel 1 ( echo HEAD TRAIN FAILED %date% %time% >> %LOG% & exit /b 1 )
echo ===== [H2] export head %date% %time% >> %LOG%
%PY% -m policy.stage_head export --dataset-root %DATA% >> %LOG% 2>&1
if errorlevel 1 ( echo HEAD EXPORT FAILED %date% %time% >> %LOG% & exit /b 1 )
echo ===== [H3] policy with the head ending each stage, seeds 0-9 %date% %time% >> %LOG%
%PY% -m policy.rollout --mode policy --ensemble 0.01 --switch head --head models\stage_head\stage_head_int8.xml --seeds 0 1 2 3 4 5 6 7 8 9 --policy models\policy_v3\act_fp32.xml --checkpoint %CKPT% --out out\v3_rollout_policy_ens_head_seeds0-9.json >> %LOG% 2>&1
echo ===== stage head v3 done %date% %time% >> %LOG%
