# One-command setup on Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1          # CPU PyTorch
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Cuda    # CUDA 12.8 PyTorch for training
param([switch]$Cuda)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$torchIndex = if ($Cuda) { "https://download.pytorch.org/whl/cu128" } else { "https://download.pytorch.org/whl/cpu" }

python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\pip.exe install --force-reinstall --no-deps torch==2.11.0 torchvision==0.26.0 --index-url $torchIndex

# SO-101 robot model (MuJoCo Menagerie, Apache-2.0): only the robotstudio_so101 folder.
if (-not (Test-Path third_party\mujoco_menagerie\robotstudio_so101)) {
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
    git -C third_party/mujoco_menagerie sparse-checkout set robotstudio_so101
}

$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe scene\build_scene.py
.venv\Scripts\python.exe -m pytest tests -q
Write-Output "Setup done. Next: .venv\Scripts\python.exe -m sim.task --seeds 0 1 2 3 4 5 6 7 8 9"
