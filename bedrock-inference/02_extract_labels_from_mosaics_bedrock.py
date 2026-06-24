#!/usr/bin/env python3
"""
02_extract_labels_from_mosaics_bedrock.py

Validation-only pipeline:
- Reads validation CSV as ground truth
- Finds corresponding mosaic.png files in sequence directories
- Asks an AWS Bedrock VLM only the questions present in the validation CSV
- Uses Amazon Bedrock Converse API with Claude 3 Haiku
- Applies strict post-processing so outputs become only YES/NO
- Treats unclear / not visible / cannot say / can not say / similar answers as NO
- Matching and normalization are case-insensitive
- Appends predictions row-by-row to CSV
- Computes accuracy, precision, recall, F1 score, and TP/TN/FP/FN

Expected validation CSV columns:
  SOPInstanceUID, Question, Answer
Optional columns:
  AccessionNumber

Expected sequence directory layout:
  <base_path>/
    <sequence_dir_1>/
      mosaic.png
      metadata.csv
    <sequence_dir_2>/
      mosaic.png
      metadata.csv
    ...

metadata.csv format:
  Information,Value
  SOPInstanceUID,1.2.3...
  AccessionNumber,12345

Example:
python3 02_extract_labels_from_mosaics_bedrock.py \
  --base_path /data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM_Sequence_Processed \
  --validation_csv /data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/VLM_Test_Data_2026_03_04_v01.csv \
  --region us-west-2 \
  --skip_existing
"""

import os
import sys
import argparse

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.csv_helpers import append_row_csv, ensure_parent_dir
from shared.prompts import build_yesno_prompt
from shared.text_utils import normalize_llm_answer, normalize_text, now_ts
from shared.validation_utils import (
    build_sequence_index,
    compute_binary_metrics,
    load_validation_csv,
    print_metrics,
)


# ----------------------------
# Defaults
# ----------------------------
DEFAULT_BASE_PATH = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM_Sequence_Processed"
DEFAULT_VALIDATION_CSV = "/home/nikhan/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/VLM_Test_Data_2026_03_04_v01.csv"

# MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
# MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"
MODEL_ID = "anthropic.claude-sonnet-4-6"

DEFAULT_REGION = "us-west-2"

DEFAULT_OUTPUT_CSV = "/home/nikhan/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/bedrock_validation_predictions.csv"
DEFAULT_METRICS_CSV = "/home/nikhan/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/bedrock_validation_metrics.csv"
DEFAULT_ERRORS_CSV = "/home/nikhan/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/bedrock_validation_errors.csv"


# ----------------------------
# Bedrock VLM call
# ----------------------------
def call_bedrock_vlm(client, image_path: str, question: str, timeout_unused: int = 180) -> str:
    prompt = build_yesno_prompt(question)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": image_bytes},
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": 16,
            "temperature": 0.0,
            "topP": 1.0,
        },
    )

    content = response.get("output", {}).get("message", {}).get("content", [])
    text_parts = [item["text"] for item in content if "text" in item]
    return " ".join(text_parts).strip() if text_parts else "NO"


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run validation-only Bedrock Claude 3 Haiku mosaic QA and compute metrics."
    )
    parser.add_argument("--base_path", type=str, default=DEFAULT_BASE_PATH)
    parser.add_argument("--validation_csv", type=str, default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--metrics_csv", type=str, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--errors_csv", type=str, default=DEFAULT_ERRORS_CSV)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip rows already present in output_csv based on SOPInstanceUID + Question.",
    )
    args = parser.parse_args()

    print(f"[{now_ts()}] Using model: {MODEL_ID}")
    print(f"[{now_ts()}] AWS region: {args.region}")
    print(f"[{now_ts()}] Temperature: 0")
    print(f"[{now_ts()}] Loading validation CSV: {args.validation_csv}")
    val_df = load_validation_csv(args.validation_csv)
    print(f"[{now_ts()}] Validation rows loaded: {len(val_df)}")

    print(f"[{now_ts()}] Building sequence index from: {args.base_path}")
    seq_index = build_sequence_index(args.base_path)
    print(f"[{now_ts()}] Indexed sequences with metadata + mosaic: {len(seq_index)}")

    try:
        client = boto3.client("bedrock-runtime", region_name=args.region)
    except Exception as e:
        raise RuntimeError(f"Failed to create Bedrock client: {e}")

    existing_keys = set()
    if args.skip_existing and os.path.exists(args.output_csv):
        try:
            existing_df = pd.read_csv(args.output_csv)
            if {"SOPInstanceUID", "Question"}.issubset(existing_df.columns):
                existing_keys = set(
                    zip(
                        existing_df["SOPInstanceUID"].astype(str).map(normalize_text),
                        existing_df["Question"].astype(str).map(normalize_text),
                    )
                )
                print(f"[{now_ts()}] Existing prediction rows detected: {len(existing_keys)}")
        except Exception as e:
            print(f"[WARN] Could not load existing output CSV for skip_existing: {e}")

    pred_fieldnames = [
        "Timestamp", "Model Name", "AccessionNumber", "SOPInstanceUID",
        "Question", "GroundTruth", "Predicted", "Raw_LLM_Output",
        "sequence_dir", "mosaic_path",
    ]
    error_fieldnames = [
        "Timestamp", "Model Name", "AccessionNumber", "SOPInstanceUID",
        "Question", "status", "details",
    ]

    all_gt = []
    all_pred = []
    processed = 0
    skipped = 0
    errors = 0
    unmatched = 0

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Running Bedrock QA"):
        sop_uid = str(row["SOPInstanceUID"]).strip()
        sop_norm = row["SOP_norm"]
        question = str(row["Question"]).strip()
        question_norm = row["Question_norm"]
        gt_answer = row["GT_Answer"]
        accession_from_val = str(row.get("AccessionNumber", "")).strip()

        key = (sop_norm, question_norm)
        if key in existing_keys:
            skipped += 1
            continue

        seq_info = seq_index.get(sop_norm)
        if seq_info is None:
            unmatched += 1
            append_row_csv(
                args.errors_csv,
                {
                    "Timestamp": now_ts(),
                    "Model Name": MODEL_ID,
                    "AccessionNumber": accession_from_val,
                    "SOPInstanceUID": sop_uid,
                    "Question": question,
                    "status": "NO_MATCHING_SEQUENCE",
                    "details": "Could not find sequence dir by SOPInstanceUID (case-insensitive).",
                },
                error_fieldnames,
            )
            continue

        accession = seq_info["accession_number"] or accession_from_val
        mosaic_path = seq_info["mosaic_path"]
        sequence_dir = seq_info["sequence_dir"]

        try:
            raw_llm_output = call_bedrock_vlm(
                client=client,
                image_path=mosaic_path,
                question=question,
                timeout_unused=args.timeout,
            )
            pred_answer = normalize_llm_answer(raw_llm_output)

            append_row_csv(
                args.output_csv,
                {
                    "Timestamp": now_ts(),
                    "Model Name": MODEL_ID,
                    "AccessionNumber": accession,
                    "SOPInstanceUID": sop_uid,
                    "Question": question,
                    "GroundTruth": gt_answer,
                    "Predicted": pred_answer,
                    "Raw_LLM_Output": raw_llm_output,
                    "sequence_dir": sequence_dir,
                    "mosaic_path": mosaic_path,
                },
                pred_fieldnames,
            )

            all_gt.append(gt_answer)
            all_pred.append(pred_answer)
            processed += 1

        except FileNotFoundError as e:
            errors += 1
            append_row_csv(
                args.errors_csv,
                {
                    "Timestamp": now_ts(),
                    "Model Name": MODEL_ID,
                    "AccessionNumber": accession,
                    "SOPInstanceUID": sop_uid,
                    "Question": question,
                    "status": "MOSAIC_NOT_FOUND",
                    "details": str(e),
                },
                error_fieldnames,
            )

        except (NoCredentialsError, ClientError, BotoCoreError, Exception) as e:
            errors += 1
            append_row_csv(
                args.errors_csv,
                {
                    "Timestamp": now_ts(),
                    "Model Name": MODEL_ID,
                    "AccessionNumber": accession,
                    "SOPInstanceUID": sop_uid,
                    "Question": question,
                    "status": "BEDROCK_CALL_FAILED",
                    "details": str(e),
                },
                error_fieldnames,
            )

    print(f"\n[{now_ts()}] Finished.")
    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped}")
    print(f"Unmatched sequences: {unmatched}")
    print(f"Errors: {errors}")

    if len(all_gt) == 0:
        print("[WARN] No predictions were produced, so no metrics were computed.")
        return

    metrics = compute_binary_metrics(all_gt, all_pred)

    metrics_row = {
        "Timestamp": now_ts(),
        "Model Name": MODEL_ID,
        "Region": args.region,
        "Temperature": 0,
        "Validation CSV": args.validation_csv,
        "Base Path": args.base_path,
        "Total Evaluated Rows": len(all_gt),
        **metrics,
        "Processed": processed,
        "Skipped Existing": skipped,
        "Unmatched Sequences": unmatched,
        "Errors": errors,
    }

    metrics_df = pd.DataFrame([metrics_row])
    ensure_parent_dir(args.metrics_csv)

    if os.path.exists(args.metrics_csv):
        old_df = pd.read_csv(args.metrics_csv)
        metrics_df = pd.concat([old_df, metrics_df], ignore_index=True)

    metrics_df.to_csv(args.metrics_csv, index=False)

    print_metrics(metrics)
    print(f"Metrics CSV saved to: {args.metrics_csv}")


if __name__ == "__main__":
    main()
