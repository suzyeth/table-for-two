@echo off
REM Validation loss of the checkpoints written by the 50k -> 80k continuation, on the GPU while the
REM 80k evaluations use the CPU. Tells whether 80k is better than 40k on held-out demos and whether
REM the curve is still falling (i.e. whether yet more training would help).
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
.venv\Scripts\python.exe -m tools.val_loss --output-dir outputs\act_contact_v3 --samples 3000 --only 045000 050000 055000 060000 065000 070000 075000 080000 --out out\act_contact_v3_val_loss_80k.json >> out\val_loss_v3_80k.log 2>&1
