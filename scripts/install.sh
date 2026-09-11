#!/usr/bin/env bash
# One-command setup on Ubuntu 24.04 (the Intel challenge's reference OS) or any Linux.
#   bash scripts/install.sh            # CPU PyTorch (enough for everything except fast training)
#   bash scripts/install.sh --cuda     # CUDA 12.8 PyTorch for training on an NVIDIA GPU
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH_INDEX="https://download.pytorch.org/whl/cpu"
if [[ "${1:-}" == "--cuda" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu128"
fi

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --force-reinstall --no-deps torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX"

# SO-101 robot model (MuJoCo Menagerie, Apache-2.0): only the robotstudio_so101 folder.
if [[ ! -d third_party/mujoco_menagerie/robotstudio_so101 ]]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
  git -C third_party/mujoco_menagerie sparse-checkout set robotstudio_so101
fi

.venv/bin/python scene/build_scene.py
.venv/bin/python -m pytest tests -q
echo "Setup done. Next: .venv/bin/python -m sim.task --seeds 0 1 2 3 4 5 6 7 8 9"
