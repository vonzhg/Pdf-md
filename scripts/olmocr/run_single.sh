#!/bin/bash
# ============================================================
# olmOCR single-PDF smoke test (NCSA Delta)
#
# Submits one PDF to olmocr to verify the pipeline works
# end-to-end before committing to the full 14-PDF batch.
#
# Submit:
#   sbatch --account=bepc-delta-gpu --partition=gpuA100x4 \
#          --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
#          --time=00:45:00 \
#          --output=/u/zfeng2/Github/PDF-MD/olmocr_test/sbatch_single.log \
#          --job-name=olmocr_single \
#          /u/zfeng2/Github/PDF-MD/run_olmocr_single_pdf.sh
# ============================================================

set -euo pipefail

# Hardcoded project root — sbatch copies scripts into /var/spool/slurmd/...,
# breaking BASH_SOURCE-based derivation. Override with PROJECT_ROOT=... if needed.
PROJECT_ROOT="${PROJECT_ROOT:-/u/zfeng2/Github/PDF-MD}"

PDF="${1:-$PROJECT_ROOT/workspace/olmocr/single_pdf/LI Bo.pdf}"
WORKSPACE="${2:-$PROJECT_ROOT/workspace/olmocr/single}"

CONDA_ENV="${CONDA_ENV:-/projects/bepc/zfeng2/envs/olmocr_env}"
export HF_HOME="${HF_HOME:-/projects/bepc/zfeng2/hf_cache}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

if command -v module >/dev/null 2>&1; then
    module purge
    module load miniforge3-python
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1

echo "=== olmOCR single-PDF smoke test ==="
echo "Node:      $(hostname)"
echo "Python:    $(which python)"
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "PDF:       $PDF"
echo "Workspace: $WORKSPACE"
echo "HF_HOME:   $HF_HOME"
echo ""

mkdir -p "$WORKSPACE"

if [ ! -f "$PDF" ]; then
    echo "ERROR: PDF not found at $PDF"
    exit 1
fi

START=$(date +%s)
olmocr "$WORKSPACE" --markdown --pdfs "$PDF"
END=$(date +%s)

echo ""
echo "============================================================"
echo "Done in $((END - START))s"
echo "Markdown output:"
ls -la "$WORKSPACE"/markdown/ 2>/dev/null || echo "  (no markdown dir produced)"
echo "============================================================"
