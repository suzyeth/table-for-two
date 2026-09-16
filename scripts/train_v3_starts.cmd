@echo off
REM v3 froze at the start of the pour (1/10 even from a scripted start): every demo stage ends with a
REM 2.5 s still hold, and "a static arm starts moving" is only in the first frames of each stage.
REM Continue 80k -> 100k showing the first 10 frames of every stage 5 times per epoch.
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
echo ===== starts-oversampled continuation from 080000 %date% %time% >> out\train_v3_starts.log
.venv\Scripts\python.exe -m policy.train --dataset-root data\contact_v3\merged --output-dir outputs\act_contact_v3 --steps 100000 --num-workers 12 --amp --start-oversample 5 --start-frames 10 --resume outputs\act_contact_v3\checkpoints\080000\pretrained_model >> out\train_v3_starts.log 2>&1
echo ===== continuation finished %date% %time% >> out\train_v3_starts.log
