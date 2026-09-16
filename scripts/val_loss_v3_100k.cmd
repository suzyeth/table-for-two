@echo off
REM Validation loss of the checkpoints from the start-oversampled continuation (85k-100k), on the GPU
REM while the 100k evaluations use the CPU.
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
.venv\Scripts\python.exe -m tools.val_loss --output-dir outputs\act_contact_v3 --samples 3000 --only 085000 090000 095000 100000 --out out\act_contact_v3_val_loss_100k.json >> out\val_loss_v3_100k.log 2>&1
