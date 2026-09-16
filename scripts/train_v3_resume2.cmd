@echo off
REM Continue v3 from the 050000 checkpoint to 80k. The first continuation (from 040000) stopped at
REM 03:15 together with the desktop app, at step ~51k; 050000 is the last complete checkpoint.
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
echo ===== resume from 050000 %date% %time% >> out\train_v3_resume.log
.venv\Scripts\python.exe -m policy.train --dataset-root data\contact_v3\merged --output-dir outputs\act_contact_v3 --steps 80000 --num-workers 12 --amp --resume outputs\act_contact_v3\checkpoints\050000\pretrained_model >> out\train_v3_resume.log 2>&1
echo ===== resume finished %date% %time% >> out\train_v3_resume.log
