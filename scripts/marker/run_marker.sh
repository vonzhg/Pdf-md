#!/bin/bash
# ============================================================
# marker-pdf extraction (NCSA Delta, interactive)
#
# Adapted from run_marker_ncsa.sbatch — no SLURM, runs in the
# foreground on whatever node you're on (login or compute).
# Intended to live at /u/zfeng2/PDF-MD/ and operate on:
#     input/   → PDFs
#     output/  → per-PDF subfolders with Markdown + figures
#
# Usage:
#   cd /u/zfeng2/PDF-MD
#   ./run_marker.sh                   # input/ → output/
#   ./run_marker.sh <in_dir> <out_dir>

#
# how to use on Delta with GPU
#
# srun --account=bepc-delta-gpu --partition=gpuA100x4-interactive --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g --time=00:30:00 --pty bash  
# bash run_marker_delta_interactive.sh
#  exit 
# For real GPU work, request an interactive node first:
#   srun --account=bepc-delta-gpu --partition=gpuA100x4 \
#        --nodes=1 --gpus=1 --mem=64g --time=02:00:00 --pty bash
# Then run this script. On the login node it falls back to CPU.
# ============================================================

set -euo pipefail

# Hardcoded project root — sbatch copies scripts into /var/spool/slurmd/...,
# breaking BASH_SOURCE-based derivation. Override with PROJECT_ROOT=... if needed.
PROJECT_ROOT="${PROJECT_ROOT:-/u/zfeng2/PDF-MD}"
INPUT_DIR="${1:-$PROJECT_ROOT/input}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/output/marker}"

CONDA_ENV="${CONDA_ENV:-marker}"

# Activate conda (Delta-style)
if command -v module >/dev/null 2>&1; then
    module purge
    module load miniforge3-python
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Pick device: CUDA if visible, otherwise CPU. Caller can override.
if [ -z "${TORCH_DEVICE:-}" ]; then
    if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        export TORCH_DEVICE=cuda
    else
        export TORCH_DEVICE=cpu
    fi
fi
export PYTHONUNBUFFERED=1

echo "=== marker-pdf extraction (interactive) ==="
echo "Node:      $(hostname)"
echo "Python:    $(which python)"
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, using: {\"$TORCH_DEVICE\"}')"
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

SUCCESS=0; FAIL=0; SKIP=0; N=0

for pdf in "${PDFS[@]}"; do
    name="$(basename "$pdf")"
    name="${name%.pdf}"
    name="${name%.PDF}"
    N=$((N + 1))

    if [ -f "$OUTPUT_DIR/$name/$name.md" ]; then
        echo "[$N/$TOTAL] SKIP $name (already extracted)"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo "[$N/$TOTAL] $name ..."
    START_TIME=$(date +%s)

    if PDF_PATH="$pdf" OUT_DIR="$OUTPUT_DIR/$name" NAME="$name" python - <<'PY' 2>&1 | tail -5; then
import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

pdf_path = os.environ['PDF_PATH']
out_dir  = os.environ['OUT_DIR']
name     = os.environ['NAME']

converter = PdfConverter(artifact_dict=create_model_dict())
rendered  = converter(pdf_path)
text, _, images = text_from_rendered(rendered)

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, f'{name}.md'), 'w') as f:
    f.write(text)
for img_name, img in images.items():
    img.save(os.path.join(out_dir, img_name))
print(f'{len(text)} chars')
PY
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        if [ -f "$OUTPUT_DIR/$name/$name.md" ]; then
            WORDS=$(wc -w < "$OUTPUT_DIR/$name/$name.md")
            echo "  OK: ${WORDS} words, ${ELAPSED}s"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "  FAIL: no output file"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "  FAIL: marker error"
        FAIL=$((FAIL + 1))
    fi
    echo ""
done

echo "============================================================"
echo "Done.  Success=$SUCCESS  Failed=$FAIL  Skipped=$SKIP  Total=$TOTAL"
echo "Output: $OUTPUT_DIR/"
echo "============================================================"
