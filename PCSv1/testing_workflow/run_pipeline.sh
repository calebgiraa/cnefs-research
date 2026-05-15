#!/usr/bin/env bash
# run_pipeline.sh
#
# Full pipe-detection pipeline with multi-view fusion:
#   1. partial_translation.py  — NBV pose evaluation → top-K perspective images + index maps + manifest
#   2. segment_perspective.py  — GroundingDINO + SAM on each view → vote fusion → labeled LAS
#   3. ransactest.py           — RANSAC cylinder fit → measurements
#
# Usage:
#   ./run_pipeline.sh <input.las> <output_dir> <n_poses> <radius> [extra partial_translation flags]
#
# Examples:
#   ./run_pipeline.sh scan.las out/ 8 3.0
#   ./run_pipeline.sh scan.las out/ 12 4.0 --n_views 4 --elev_range 0 45 --fov 70
#
# Fusion settings (edit below or pass --fusion/--min_votes/--weight_thresh to this script):
#   FUSION        or | voting | weighted   (default: voting)
#   MIN_VOTES     integer                  (default: 2, used with voting)
#   WEIGHT_THRESH float                    (default: 0.5, used with weighted)
#
# RANSAC defaults (edit below if needed):
#   --epsilon 0.5  --ransac-thresh 0.01  --ransac-iters 1000

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <input.las> <output_dir> <n_poses> <radius> [extra partial_translation args]"
    exit 1
fi

INPUT_LAS="$1"
OUTPUT_DIR="$2"
N_POSES="$3"
RADIUS="$4"
shift 4  # remaining args forwarded to partial_translation.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASENAME="$(basename "$INPUT_LAS" .las)"

# --- Fusion config ---
FUSION="weighted"
MIN_VOTES=2
WEIGHT_THRESH=0.5

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  Step 1/3: Multi-View Projection"
echo "========================================"
python "$SCRIPT_DIR/partial_translation.py" \
    "$INPUT_LAS" \
    "$OUTPUT_DIR" \
    "$N_POSES" \
    "$RADIUS" \
    "$@"

MANIFEST="$OUTPUT_DIR/${BASENAME}_views.txt"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Views manifest not found: $MANIFEST"
    exit 1
fi

N_VIEWS=$(wc -l < "$MANIFEST")
echo "  Found $N_VIEWS view(s) in manifest."

echo ""
echo "========================================"
echo "  Step 2/3: Multi-View Segmentation"
echo "========================================"
python "$SCRIPT_DIR/segment_perspective.py" \
    "$MANIFEST" \
    "$INPUT_LAS" \
    "$OUTPUT_DIR" \
    --fusion "$FUSION" \
    --min_votes "$MIN_VOTES" \
    --weight_thresh "$WEIGHT_THRESH"

LABELED_LAS="$OUTPUT_DIR/${BASENAME}_labeled.las"

if [ ! -f "$LABELED_LAS" ]; then
    echo "ERROR: Labeled LAS not found: $LABELED_LAS"
    exit 1
fi

echo ""
echo "========================================"
echo "  Step 3/3: RANSAC Cylinder Fitting"
echo "========================================"
python "$SCRIPT_DIR/ransactest.py" \
    --file "$LABELED_LAS" \
    --voxel-size 0.01 \
    --epsilon 0.08 \
    --ransac-thresh 0.005 \
    --ransac-iters 1000

echo ""
echo "========================================"
echo "  Pipeline complete."
echo "  Outputs in: $OUTPUT_DIR"
echo "========================================"
