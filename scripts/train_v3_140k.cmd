@echo off
REM Continue the start-oversampled v3 run 120k to 140k (same settings). From 100k to 120k the hand-over
REM went 0/10 to 5/10 and validation loss still fell (0.0211 to 0.0190).
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
echo ===== continuation 120k to 140k %date% %time% >> out\train_v3_140k.log
.venv\Scripts\python.exe -m policy.train --dataset-root data\contact_v3\merged --output-dir outputs\act_contact_v3 --steps 140000 --num-workers 12 --amp --resume outputs\act_contact_v3\checkpoints\120000\pretrained_model >> out\train_v3_140k.log 2>&1
echo ===== continuation finished %date% %time% >> out\train_v3_140k.log
