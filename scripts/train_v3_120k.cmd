@echo off
REM Continue the start-oversampled v3 run 100k -> 120k (same settings, more steps). At 100k the pour
REM moves again (20 of 24 beads in the mug on seed 0) but spills 3-4; validation loss still falls.
REM 4 data workers so the evaluations running beside it keep the CPU.
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
echo ===== continuation 100k to 120k %date% %time% >> out\train_v3_120k.log
.venv\Scripts\python.exe -m policy.train --dataset-root data\contact_v3\merged --output-dir outputs\act_contact_v3 --steps 120000 --num-workers 4 --amp --resume outputs\act_contact_v3\checkpoints\100000\pretrained_model >> out\train_v3_120k.log 2>&1
echo ===== continuation finished %date% %time% >> out\train_v3_120k.log
