@echo off
REM Validation loss of every v3 checkpoint on the held-out episodes, queued behind the v3 training:
REM the v3 run was started before policy/train.py passed --eval_steps, so it logged none.
REM Runs on the GPU while the pipeline's evaluations use the CPU. Log: out\val_loss_v3.log
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set PY=.venv\Scripts\python.exe
set LOG=out\val_loss_v3.log
set PIPE=out\train_v3_pipeline.log
echo ===== waiting for the v3 training to finish %date% %time% >> %LOG%

:wait
findstr /l /c:"RECORD FAILED" /c:"TRAIN FAILED" %PIPE% >nul && ( echo v3 TRAINING FAILED - nothing to score %date% %time% >> %LOG% & exit /b 1 )
findstr /l /c:"===== [C] export v3" %PIPE% >nul && goto ready
timeout /t 60 /nobreak >nul
goto wait

:ready
echo ===== [V] validation loss of every checkpoint %date% %time% >> %LOG%
%PY% -m tools.val_loss --output-dir outputs\act_contact_v3 --samples 3000 >> %LOG% 2>&1
echo ===== validation loss done %date% %time% >> %LOG%
