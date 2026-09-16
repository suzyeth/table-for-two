@echo off
REM Accuracy-controlled INT8 export for v3, queued behind the whole v3 pipeline (it needs the CPU the
REM evaluations are using). The plain INT8 export of v3 has a 0.230 rad (13 deg) worst-case action
REM error - v2's was 0.10 rad and the plan's gate is 0.03 - so NNCF is asked to keep the drop within
REM --max-drop, leaving some layers in floating point. Log: out\int8_ac_v3.log
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set PY=.venv\Scripts\python.exe
set LOG=out\int8_ac_v3.log
set PIPE=out\train_v3_pipeline.log
set DATA=data\contact_v3\merged
set CKPT=outputs\act_contact_v3\checkpoints\040000\pretrained_model
echo ===== waiting for the v3 pipeline to finish %date% %time% >> %LOG%

:wait
findstr /l /c:"TRAIN FAILED" /c:"EXPORT FAILED" %PIPE% >nul && ( echo v3 PIPELINE FAILED - no accuracy-control export %date% %time% >> %LOG% & exit /b 1 )
findstr /l /c:"===== v3 pipeline done" %PIPE% >nul && goto ready
timeout /t 60 /nobreak >nul
goto wait

:ready
echo ===== [I] INT8 with accuracy control %date% %time% >> %LOG%
%PY% -m policy.export_openvino --checkpoint %CKPT% --dataset-root %DATA% --out-dir models\policy_v3_ac --accuracy-control --max-drop 0.02 >> %LOG% 2>&1
if errorlevel 1 ( echo AC EXPORT FAILED %date% %time% >> %LOG% & exit /b 1 )
echo ===== [I2] rollout with the accuracy-controlled INT8, seeds 0-9 %date% %time% >> %LOG%
%PY% -m policy.rollout --mode policy --ensemble 0.01 --seeds 0 1 2 3 4 5 6 7 8 9 --policy models\policy_v3_ac\act_int8.xml --checkpoint %CKPT% --out out\v3_rollout_policy_int8_ac_ens_seeds0-9.json >> %LOG% 2>&1
echo ===== INT8 accuracy control done %date% %time% >> %LOG%
