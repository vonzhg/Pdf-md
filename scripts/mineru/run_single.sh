#!/bin/bash
# ============================================================
# MinerU single-PDF smoke test (NCSA Delta)
#
# Runs MinerU on one PDF to verify the env + VLM model work
# end-to-end before committing to a full batch. Defaults to
# input/zhang_2025.pdf.
#
# Usage:
#   bash scripts/mineru/run_single.sh                 # default PDF
#   bash scripts/mineru/run_single.sh <pdf> <out_dir>
#
# Submit:
#   sbatch --account=bepc-delta-gpu --partition=gpuA100x4-interactive \
#          --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
#          --time=00:45:00 \
#          --output=/u/zfeng2/Github/PDF-MD/logs/mineru/sbatch_single.log \
#          --job-name=mineru_single \
#          /u/zfeng2/Github/PDF-MD/scripts/mineru/run_single.sh
# ============================================================

set -euo pipefail

# Hardcoded project root — sbatch copies scripts into /var/spool/slurmd/...,
# breaking BASH_SOURCE-based derivation. Override with PROJECT_ROOT=... if needed.
PROJECT_ROOT="${PROJECT_ROOT:-/u/zfeng2/Github/PDF-MD}"
PDF="${1:-$PROJECT_ROOT/input/zhang_2025.pdf}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/output/mineru}"

CONDA_ENV="${CONDA_ENV:-mineru}"
BACKEND="${MINERU_BACKEND:-vlm-auto-engine}"
# Model weights: read from the Hugging Face cache, offline (compute nodes have no network).
# "local" would need ~/mineru.json, which no longer exists; same convention as Econ-RAG.
export MINERU_MODEL_SOURCE="${MINERU_MODEL_SOURCE:-huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/projects/bepc/zfeng2/hf_cache}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

# Activate conda (Delta-style)
if command -v module >/dev/null 2>&1; then
    module purge
    module load miniforge3-python
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1

echo "=== MinerU single-PDF smoke test ==="
echo "Node:    $(hostname)"
echo "Python:  $(which python)"
echo "Backend: $BACKEND"
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "PDF:     $PDF"
echo "Output:  $OUTPUT_DIR"
echo ""

if [ ! -f "$PDF" ]; then
    echo "ERROR: PDF not found at $PDF"
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

name="$(basename "$PDF")"
name="${name%.pdf}"
name="${name%.PDF}"

START=$(date +%s)
mineru -p "$PDF" -o "$OUTPUT_DIR" -b "$BACKEND"
ELAPSED=$(($(date +%s) - START))

echo ""
echo "============================================================"
echo "Done in ${ELAPSED}s"
echo "Output files:"
find "$OUTPUT_DIR/$name" -type f 2>/dev/null | sort || echo "  (no output folder produced)"
echo "============================================================"
