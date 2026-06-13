#!/bin/bash
# ============================================================
# MinerU extraction (NCSA Delta) — full batch over input/
#
# Runs MinerU 3.x (VLM backend, vLLM engine) on every PDF in
# input/ and writes results to output/mineru/. MinerU creates
# one folder per PDF:
#     output/mineru/<name>/vlm/<name>.md
#                              <name>_content_list.json
#                              <name>_layout.pdf
#                              <name>_origin.pdf
#                              images/...
#
# Usage:
#   bash scripts/mineru/run.sh                   # input/ -> output/mineru/
#   bash scripts/mineru/run.sh <in_dir> <out_dir>
#
# Needs a GPU. Submit via sbatch:
#   sbatch --account=bepc-delta-gpu --partition=gpuA100x4-interactive \
#          --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
#          --time=01:00:00 \
#          --output=/u/zfeng2/PDF-MD/logs/mineru/sbatch_run.log \
#          --job-name=mineru_run \
#          /u/zfeng2/PDF-MD/scripts/mineru/run.sh
# ============================================================

set -euo pipefail

# Hardcoded project root — sbatch copies scripts into /var/spool/slurmd/...,
# breaking BASH_SOURCE-based derivation. Override with PROJECT_ROOT=... if needed.
PROJECT_ROOT="${PROJECT_ROOT:-/u/zfeng2/PDF-MD}"
INPUT_DIR="${1:-$PROJECT_ROOT/input}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/output/mineru}"

CONDA_ENV="${CONDA_ENV:-mineru}"
# Backend: vlm-auto-engine picks the vLLM engine for fast local GPU inference.
BACKEND="${MINERU_BACKEND:-vlm-auto-engine}"
# Use the model weights pre-downloaded by setup_env.sh (offline on compute nodes).
export MINERU_MODEL_SOURCE="${MINERU_MODEL_SOURCE:-local}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

# Activate conda (Delta-style)
if command -v module >/dev/null 2>&1; then
    module purge
    module load miniforge3-python
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1

echo "=== MinerU extraction ==="
echo "Node:      $(hostname)"
echo "Python:    $(which python)"
echo "Backend:   $BACKEND"
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "Input:     $INPUT_DIR"
echo "Output:    $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
PDFS=("$INPUT_DIR"/*.pdf "$INPUT_DIR"/*.PDF)
TOTAL=${#PDFS[@]}
echo "Papers:    $TOTAL"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "No PDFs found in $INPUT_DIR"
    exit 0
fi

# Skip already-converted PDFs: MinerU creates output/mineru/<name>/ per PDF.
# Stage the not-yet-done PDFs as symlinks into a temp dir and run MinerU once
# over that dir — a single invocation keeps the vLLM engine resident for the
# whole batch (invoking `mineru` per file would reload the ~2 GB model each
# time and pay the vLLM startup cost repeatedly).
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

PENDING=0
SKIP=0
for pdf in "${PDFS[@]}"; do
    name="$(basename "$pdf")"
    name="${name%.pdf}"
    name="${name%.PDF}"
    if [ -d "$OUTPUT_DIR/$name" ]; then
        echo "SKIP  $name (already extracted)"
        SKIP=$((SKIP + 1))
    else
        ln -s "$(readlink -f "$pdf")" "$STAGE/$(basename "$pdf")"
        PENDING=$((PENDING + 1))
    fi
done
echo ""
echo "To extract: $PENDING   Skipped: $SKIP"

if [ "$PENDING" -eq 0 ]; then
    echo "Nothing to do — all PDFs already converted."
    exit 0
fi

START_TIME=$(date +%s)
mineru -p "$STAGE" -o "$OUTPUT_DIR" -b "$BACKEND"
ELAPSED=$(($(date +%s) - START_TIME))

OUT_COUNT=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo ""
echo "============================================================"
echo "Done.  Extracted this run: $PENDING   Output folders: $OUT_COUNT   Elapsed: ${ELAPSED}s"
echo "Output: $OUTPUT_DIR/"
echo "============================================================"
