#!/bin/bash

# ===============================
# Run AngioVision Mosaic QA on all models
# ===============================

SCRIPT="02_extract_labels_from_mosaics.py"

# Updated test-data paths
BASE_PATH="/data/Deep_Angiography/Validation_Data/test-data"
VAL_CSV="/data/Deep_Angiography/Validation_Data/test-data/gt.csv"

# Use absolute output dir so both bash + inline python stay consistent
OUTPUT_DIR="/data/Deep_Angiography/AngioVision/frame-processing/model_runs"
mkdir -p "$OUTPUT_DIR"

MODELS=(
    "gemma3:27b"
    "llava:34b"
    "llama3.2-vision:11b"
    "gpt-oss:20b"
    "qwen3-vl:32b"
    "ministral-3:8b"
    "deepseek-ocr:3b"
)

echo "=============================================="
echo "Checking required paths"
echo "=============================================="

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 1
fi

if [ ! -d "$BASE_PATH" ]; then
    echo "ERROR: Base path not found: $BASE_PATH"
    exit 1
fi

if [ ! -f "$VAL_CSV" ]; then
    echo "ERROR: Validation CSV not found: $VAL_CSV"
    exit 1
fi

for MODEL in "${MODELS[@]}"
do
    SAFE_NAME=$(echo "$MODEL" | sed 's/[\/:]/_/g')

    echo "=============================================="
    echo "Running model: $MODEL"
    echo "=============================================="

    python3 "$SCRIPT" \
        --model "$MODEL" \
        --base_path "$BASE_PATH" \
        --validation_csv "$VAL_CSV" \
        --output_csv "$OUTPUT_DIR/${SAFE_NAME}_predictions.csv" \
        --metrics_csv "$OUTPUT_DIR/${SAFE_NAME}_metrics.csv" \
        --errors_csv "$OUTPUT_DIR/${SAFE_NAME}_errors.csv"

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Model failed: $MODEL"
        echo "Exit code: $EXIT_CODE"
    else
        echo "Finished model: $MODEL"
    fi
    echo
done

echo "=============================================="
echo "Merging metrics (inline Python)"
echo "=============================================="

python3 << 'EOF'
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path("/data/Deep_Angiography/AngioVision/frame-processing/model_runs")
OUT_FILE = BASE_DIR / "merged_metrics.csv"

KEEP_COLS = [
    "TP",
    "TN",
    "FP",
    "FN",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
]

metric_files = sorted(BASE_DIR.glob("*metrics.csv"))

if not metric_files:
    print("No metrics.csv files found.")
    sys.exit(0)

dfs = []

for f in metric_files:
    print(f"Reading: {f.name}")
    df = pd.read_csv(f)

    missing_cols = [col for col in KEEP_COLS if col not in df.columns]
    if missing_cols:
        print(f"Skipping {f.name} because columns are missing: {missing_cols}")
        continue

    df = df[KEEP_COLS].copy()

    model_name = f.name.replace("_metrics.csv", "")
    df.insert(0, "Model", model_name)

    dfs.append(df)

if not dfs:
    print("No valid metrics files found after column checks.")
    sys.exit(1)

merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_csv(OUT_FILE, index=False)

print(f"\nMerged {len(dfs)} files.")
print(f"Saved to: {OUT_FILE}")
EOF

MERGE_EXIT_CODE=$?

if [ $MERGE_EXIT_CODE -ne 0 ]; then
    echo "ERROR: merge step failed"
    exit $MERGE_EXIT_CODE
fi

echo "=============================================="
echo "ALL MODELS FINISHED"
echo "MERGED METRICS SUCCESSFULLY"
echo "=============================================="

# chmod +x 02_extract_labels_from_mosaics.sh
# bash 02_extract_labels_from_mosaics.sh