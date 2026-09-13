@echo off
cd /d %~dp0..
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
echo ===== record_b start %date% %time% >> out\record_b.log
.venv\Scripts\python.exe -m data.record --episodes 150 --start-seed 300 --max-tries 220 --root data\dinner_table_contact_b >> out\record_b.log 2>&1
echo ===== record_b done %date% %time% errorlevel %errorlevel% >> out\record_b.log
