#!/bin/bash
#SBATCH --account=bepc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=/u/zfeng2/Github/PDF-MD/logs/mineru/batch_run_%j.log
#SBATCH --job-name=mineru_batch

# ============================================================
# MinerU extraction (NCSA Delta) — Full Batch Processing
#
# This script processes all PDF files in the input/ directory
# using the MinerU VLM backend. It uses a single MinerU
# invocation over a directory of symlinks to keep the vLLM
# engine resident, avoiding repeated model reload costs.
# ============================================================

set -euo pipefail

PROJECT_ROOT="/u/zfeng2/Github/PDF-MD"
INPUT_DIR="$PROJECT_ROOT/input"
OUTPUT_DIR="$PROJECT_ROOT/output/mineru"
CONDA_ENV="mineru"
BACKEND="vlm-auto-engine"

# Offline mode for model weights
export MINERU_MODEL_SOURCE="local"
export HF_HUB_OFFLINE="1"
export LC_ALL="en_US.UTF-8"
export PYTHONUNBUFFERED=1

# Load environment
if command -v module >/dev/null 2>&1; then
    module purge
    module load miniforge3-python
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "=== MinerU Batch Processing ==="
echo "Node:      $(hostname)"
echo "Input:     $INPUT_DIR"
echo "Output:    $OUTPUT_DIR"
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/logs/mineru"

# Prepare staging directory for symlinks
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

shopt -s nullglob
PDFS=("$INPUT_DIR"/*.pdf "$INPUT_DIR"/*.PDF)
PENDING=0
SKIP=0

for pdf in "${PDFS[@]}"; do
    name="$(basename "$pdf")"
    name="${name%.pdf}"
    name="${name%.PDF}"
    
    # Check if output already exists (MinerU creates a folder per PDF)
    if [ -d "$OUTPUT_DIR/$name" ]; then
        # Check if the markdown file exists in the expected subfolder
        # MinerU 3.x structure: output/mineru/<name>/vlm/<name>.md
        if [ -f "$OUTPUT_DIR/$name/vlm/$name.md" ]; then
            SKIP=$((SKIP + 1))
            continue
        fi
    fi
    
    ln -s "$(readlink -f "$pdf")" "$STAGE/$(basename "$pdf")"
    PENDING=$((PENDING + 1))
done

echo "Total PDFs found: ${#PDFS[@]}"
echo "Already skipped:  $SKIP"
echo "To process:       $PENDING"

if [ "$PENDING" -eq 0 ]; then
    echo "Nothing to do."
    exit 0
fi

echo "Starting MinerU extraction..."
START_TIME=$(date +%s)

# Run MinerU on the staged symlinks
mineru -p "$STAGE" -o "$OUTPUT_DIR" -b "$BACKEND"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Batch processing complete."
echo "Processed: $PENDING PDFs"
echo "Elapsed:   ${ELAPSED}s (approx $((ELAPSED / 60))m)"
echo "============================================================"
