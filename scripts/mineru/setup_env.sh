#!/bin/bash
# ============================================================
# One-time MinerU environment setup on NCSA Delta.
#
# Creates the `mineru` conda env (Python 3.11), installs
# mineru[all] (MinerU 3.x + vLLM + torch CUDA wheels via uv),
# then pre-downloads the MinerU2.5 VLM model weights and writes
# ~/mineru.json pointing the `vlm` backend at the local cache.
#
# Runs idempotently: re-running activates the existing env and
# installs/upgrades on top.
#
# Run once. The install needs no GPU (only internet); a login
# node works but Lustre I/O is slow, so a compute node is nicer:
#   sbatch --account=bepc-delta-gpu --partition=gpuA100x4 \
#          --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
#          --time=03:00:00 \
#          --output=/u/zfeng2/Github/PDF-MD/logs/mineru/sbatch_setup.log \
#          --job-name=mineru_install \
#          /u/zfeng2/Github/PDF-MD/scripts/mineru/setup_env.sh
#
# After install, the env is reused by run.sh / run_single.sh.
# ============================================================

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-mineru}"
# Source to pull model weights from: huggingface (default) or modelscope.
MODEL_SOURCE="${MINERU_MODEL_DL_SOURCE:-huggingface}"

echo "=== MinerU env setup ==="
echo "Conda env:    $CONDA_ENV"
echo "Model source: $MODEL_SOURCE"
echo "Node:         $(hostname)"
echo ""

# Activate conda (Delta-style)
if command -v module >/dev/null 2>&1; then
    module purge
    module load miniforge3-python
fi
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "Env '$CONDA_ENV' already exists — activating, will install on top."
    conda activate "$CONDA_ENV"
else
    echo "Creating conda env '$CONDA_ENV' (Python 3.11)..."
    conda create -y -n "$CONDA_ENV" python=3.11
    conda activate "$CONDA_ENV"
fi

echo ""
echo "Installing mineru[all] (MinerU 3.x + vLLM + torch CUDA wheels)..."
pip install --upgrade pip uv
uv pip install --python "$(which python)" -U "mineru[all]"

echo ""
echo "Downloading MinerU VLM model weights (source: $MODEL_SOURCE)..."
# Writes weights into the HF cache and configures ~/mineru.json (models-dir.vlm).
mineru-models-download -s "$MODEL_SOURCE" -m vlm

echo ""
echo "=== Verify ==="
mineru --version
python - <<'PY'
import importlib
for mod in ("mineru", "vllm", "torch", "transformers"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod}: {getattr(m, '__version__', '(no __version__)')}")
    except Exception as e:
        print(f"{mod}: IMPORT FAIL: {e}")
import torch
print("torch CUDA build:", torch.version.cuda)
PY

echo ""
echo "=== Setup done ==="
echo "Next: bash /u/zfeng2/Github/PDF-MD/scripts/mineru/run_single.sh   (smoke test)"
