#!/usr/bin/env bash
# run_pipeline.sh
#
# Full pipe-detection pipeline:
#   1. partial_translation.py  — NBV perspective projection → image + index_map
#   2. segment_perspective.py  — GroundingDINO + SAM → labeled LAS
#   3. ransactest.py           — RANSAC cylinder fit → measurements
#
# Usage:
#   ./run_pipeline.sh <input.las> <output_dir> <n_poses> <radius> [extra partial_translation flags]
#
# Examples:
#   ./run_pipeline.sh scan.las out/ 5 3.0
#   ./run_pipeline.sh scan.las out/ 8 4.0 --elev_range 0 45 --include_topdown --fov 70
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

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  Step 1/3: Perspective Projection"
echo "========================================"
python "$SCRIPT_DIR/partial_translation.py" \
    "$INPUT_LAS" \
    "$OUTPUT_DIR" \
    "$N_POSES" \
    "$RADIUS" \
    "$@"

IMAGE="$OUTPUT_DIR/${BASENAME}_partial.png"
INDEX_MAP="$OUTPUT_DIR/${BASENAME}_index_map.npy"

if [ ! -f "$IMAGE" ] || [ ! -f "$INDEX_MAP" ]; then
    echo "ERROR: Expected outputs not found: $IMAGE / $INDEX_MAP"
    exit 1
fi

echo ""
echo "========================================"
echo "  Step 2/3: Segmentation"
echo "========================================"
python "$SCRIPT_DIR/segment_perspective.py" \
    "$IMAGE" \
    "$INDEX_MAP" \
    "$INPUT_LAS" \
    "$OUTPUT_DIR"

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
    --epsilon 0.5 \
    --ransac-thresh 0.01 \
    --ransac-iters 1000

echo ""
echo "========================================"
echo "  Pipeline complete."
echo "  Outputs in: $OUTPUT_DIR"
echo "========================================"
