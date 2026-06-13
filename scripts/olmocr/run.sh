#!/bin/bash
# ============================================================
# olmOCR extraction (NCSA Delta, interactive)
#
# Mirrors run_marker_delta_interactive.sh but invokes AllenAI
# olmOCR (https://github.com/allenai/olmocr) instead of marker.
# Intended to live at /u/zfeng2/PDF-MD/ and operate on:
#     input/                        -> PDFs
#     olmocr_test/workspace/        -> olmocr workspace + intermediate state
#     olmocr_test/output/           -> flattened markdown (one .md per PDF)
#
# Usage:
#   cd /u/zfeng2/PDF-MD
#   bash run_olmocr_delta_interactive.sh                  # input/ -> olmocr_test/output/
#   bash run_olmocr_delta_interactive.sh <in> <work> <out>
#
# How to use on Delta with GPU (one-time, then run this script):
#
#   srun --account=bepc-delta-gpu --partition=gpuA100x4-interactive \
#        --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
#        --time=01:30:00 --pty bash
#   bash run_olmocr_delta_interactive.sh
#   exit
#
# First run downloads the olmOCR-7B weights (~15 GB) into $HF_HOME.
# Subsequent runs reuse the cache.
# ============================================================

set -euo pipefail

# Hardcoded project root — sbatch copies the submitted script into
# /var/spool/slurmd/job<id>/ before executing, so deriving PROJECT_ROOT from
# BASH_SOURCE[0] breaks under sbatch. Override with PROJECT_ROOT=... if needed.
PROJECT_ROOT="${PROJECT_ROOT:-/u/zfeng2/PDF-MD}"
INPUT_DIR="${1:-$PROJECT_ROOT/input}"
WORKSPACE="${2:-$PROJECT_ROOT/workspace/olmocr/run}"
OUTPUT_DIR="${3:-$PROJECT_ROOT/output/olmocr}"

CONDA_ENV="${CONDA_ENV:-/projects/bepc/zfeng2/envs/olmocr_env}"
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

echo "=== olmOCR extraction (interactive) ==="
echo "Node:      $(hostname)"
echo "Python:    $(which python)"
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "Input:     $INPUT_DIR"
echo "Workspace: $WORKSPACE"
echo "Output:    $OUTPUT_DIR"
echo "HF_HOME:   $HF_HOME"

mkdir -p "$WORKSPACE" "$OUTPUT_DIR" "$HF_HOME"

shopt -s nullglob
PDFS=("$INPUT_DIR"/*.pdf "$INPUT_DIR"/*.PDF)
TOTAL=${#PDFS[@]}
echo "Papers:    $TOTAL"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "No PDFs found in $INPUT_DIR"
    exit 0
fi

START_TIME=$(date +%s)

# olmocr handles its own per-PDF parallelism. Pass each PDF path as a separate
# arg (--pdfs is nargs="*"); olmocr does NOT expand shell globs internally.
# --max_concurrent_requests caps in-flight vLLM HTTP requests; default 1600
# saturates vLLM on a single A100 and causes the engine to deadlock. 64 keeps
# the GPU busy without exhausting the KV cache.
olmocr "$WORKSPACE" --markdown \
    --max_concurrent_requests 64 \
    --pdfs "${PDFS[@]}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Flatten markdown into OUTPUT_DIR. olmocr nests outputs under the input PDF's
# absolute path (e.g. workspace/markdown/u/zfeng2/PDF-MD/input/<name>.md), so
# walk the tree and copy each .md to OUTPUT_DIR/<basename>.
if [ -d "$WORKSPACE/markdown" ]; then
    find "$WORKSPACE/markdown" -type f -name '*.md' -exec cp -f -- {} "$OUTPUT_DIR"/ \;
fi

OUT_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.md' | wc -l)

echo ""
echo "============================================================"
echo "Done.  Markdown files: $OUT_COUNT  Elapsed: ${ELAPSED}s"
echo "Workspace: $WORKSPACE"
echo "Output:    $OUTPUT_DIR"
echo "============================================================"
