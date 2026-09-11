#!/usr/bin/env bash
# One-command setup on Ubuntu 24.04 (the Intel challenge's reference OS) or any Linux.
#   bash scripts/install.sh            # CPU PyTorch (enough for everything except fast training)
#   bash scripts/install.sh --cuda     # CUDA 12.8 PyTorch for training on an NVIDIA GPU
# Needs Python 3.12 and git. On Ubuntu:
#   sudo apt install python3.12-venv libegl1 libgl1 fonts-dejavu-core git
# Headless machines (no display) must also `export MUJOCO_GL=egl` before rendering anything.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  echo "Python 3.12 is required (found: $("$PY" --version 2>&1)). Set PYTHON=/path/to/python3.12." >&2
  exit 1
fi

TORCH_INDEX="https://download.pytorch.org/whl/cpu"
if [[ "${1:-}" == "--cuda" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu128"
fi

"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
# Torch first from the chosen index, so the requirements install keeps it instead of pulling CUDA wheels.
.venv/bin/pip install torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX"
.venv/bin/pip install -r requirements.txt

# SO-101 robot model (MuJoCo Menagerie, Apache-2.0): only the robotstudio_so101 folder.
if [[ ! -d third_party/mujoco_menagerie/robotstudio_so101 ]]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
  git -C third_party/mujoco_menagerie sparse-checkout set robotstudio_so101
fi

.venv/bin/python scene/build_scene.py
.venv/bin/python -m pytest -q
echo "Setup done. Next: .venv/bin/python -m sim.task    (scripted 10-seed evaluation, ~5 min)"
echo "No display? export MUJOCO_GL=egl before recording, rollouts or videos."
