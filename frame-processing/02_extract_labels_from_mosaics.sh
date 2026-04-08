#!/bin/bash

# ===============================
# Run AngioVision Mosaic QA on all models
# ===============================

SCRIPT="02_extract_labels_from_mosaics.py"

BASE_PATH="/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM_Sequence_Processed"
VAL_CSV="/data/Deep_Angiography/Validation_Data/test-data/gt.csv"

OUTPUT_DIR="model_runs"
mkdir -p $OUTPUT_DIR


MODELS=(
"qwen3-vl:8b"
"gemma3:27b"
"llava:34b"
"llama3.2-vision:11b"
"gpt-oss:20b"
"qwen3-vl:32b"
)

for MODEL in "${MODELS[@]}"
do
    SAFE_NAME=$(echo $MODEL | sed 's/[\/:]/_/g')

    echo "=============================================="
    echo "Running model: $MODEL"
    echo "=============================================="

    python3 $SCRIPT \
        --model "$MODEL" \
        --base_path $BASE_PATH \
        --validation_csv $VAL_CSV \
        --output_csv "$OUTPUT_DIR/${SAFE_NAME}_predictions.csv" \
        --metrics_csv "$OUTPUT_DIR/${SAFE_NAME}_metrics.csv" \
        --errors_csv "$OUTPUT_DIR/${SAFE_NAME}_errors.csv"

    echo "Finished model: $MODEL"
    echo
done

echo "=============================================="
echo "ALL MODELS FINISHED"
echo "=============================================="

# chmod +x 02_extract_labels_from_mosaics.sh