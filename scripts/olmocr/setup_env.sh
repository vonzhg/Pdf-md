#!/bin/bash
# ============================================================
# One-time olmOCR environment setup on NCSA Delta.
#
# Creates /projects/bepc/zfeng2/envs/olmocr_env (Python 3.11),
# installs poppler + fonts via conda-forge, then olmocr[gpu]
# from the PyTorch CUDA wheel index.
#
# Run this once on a GPU node:
#   srun --account=bepc-delta-gpu --partition=gpuA100x4-interactive \
#        --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
#        --time=01:30:00 --pty bash
#   bash /u/zfeng2/Github/PDF-MD/setup_olmocr_env_delta.sh
#
# After install, the env is reused by run_olmocr_delta_interactive.sh.
# ============================================================

set -euo pipefail

ENV_PATH="${ENV_PATH:-/projects/bepc/zfeng2/envs/olmocr_env}"
HF_CACHE="${HF_HOME:-/projects/bepc/zfeng2/hf_cache}"

# CUDA wheel channel — default cu128 (current). Override with CUDA_INDEX=cu121
# if Delta's host driver is < 525.
CUDA_INDEX="${CUDA_INDEX:-cu128}"

echo "=== olmOCR env setup ==="
echo "Env path:    $ENV_PATH"
echo "HF cache:    $HF_CACHE"
echo "CUDA wheels: $CUDA_INDEX"
echo "Node:        $(hostname)"
echo ""

# Activate conda (Delta-style)
if command -v module >/dev/null 2>&1; then
    module purge
    module load miniforge3-python
fi
eval "$(conda shell.bash hook)"

# Driver check (best-effort; only meaningful on a GPU node)
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi ---"
    nvidia-smi | head -3 || true
    echo ""
fi

if [ -d "$ENV_PATH" ]; then
    echo "Env already exists at $ENV_PATH — activating, will install on top."
    conda activate "$ENV_PATH"
else
    echo "Creating conda env at $ENV_PATH (Python 3.11)..."
    conda create -y -p "$ENV_PATH" python=3.11 pip
    conda activate "$ENV_PATH"
fi

echo ""
echo "Installing system deps via conda-forge (poppler, fonts)..."
conda install -y -c conda-forge poppler fonts-conda-ecosystem

echo ""
echo "Installing olmocr[gpu] via pip (PyTorch $CUDA_INDEX wheels)..."
pip install --upgrade pip
pip install "olmocr[gpu]" --extra-index-url "https://download.pytorch.org/whl/$CUDA_INDEX"

echo ""
echo "Creating HF cache dir..."
mkdir -p "$HF_CACHE"

echo ""
echo "=== Verify ==="
which olmocr || true
which pdftoppm || true
pdftoppm -v 2>&1 | head -1 || true
python - <<'PY'
import importlib, sys
print("python:", sys.version.split()[0])
for mod in ("olmocr", "torch", "vllm", "transformers"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod}: {getattr(m, '__version__', '(no __version__)')}")
    except Exception as e:
        print(f"{mod}: IMPORT FAIL: {e}")
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo ""
echo "=== Setup done ==="
echo "Next: bash /u/zfeng2/Github/PDF-MD/run_olmocr_delta_interactive.sh"
