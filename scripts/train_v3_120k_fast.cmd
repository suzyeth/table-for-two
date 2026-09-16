@echo off
REM Continue the start-oversampled v3 run 100k -> 120k (same settings, more steps). At 100k the pour
REM moves again (20 of 24 beads in the mug on seed 0) but spills 3-4; validation loss still falls.
REM Restarted from 105000 with 12 data workers: with 4 beside the evaluations it ran at ~1.5 steps/s.
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
echo ===== continuation 105k to 120k with 12 workers %date% %time% >> out\train_v3_120k.log
.venv\Scripts\python.exe -m policy.train --dataset-root data\contact_v3\merged --output-dir outputs\act_contact_v3 --steps 120000 --num-workers 12 --amp --resume outputs\act_contact_v3\checkpoints\105000\pretrained_model >> out\train_v3_120k.log 2>&1
echo ===== continuation finished %date% %time% >> out\train_v3_120k.log
