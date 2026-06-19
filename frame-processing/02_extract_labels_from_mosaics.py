#!/usr/bin/env python3
"""
02_extract_labels_from_mosaics.py

Validation-only pipeline:
- Reads validation CSV as ground truth
- Finds corresponding mosaic.png files in sequence directories
- Asks an Ollama VLM only the questions present in the validation CSV
- Uses Ollama local API with temperature = 0 for deterministic inference
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
python3 02_extract_labels_from_mosaics.py \
  --model llama3.2-vision:11b \
  --base_path /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed \
  --validation_csv /data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/VLM_Test_Data_2026_03_04_v01.csv \
  --skip_existing
"""

import os
import sys
import argparse

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.csv_helpers import append_row_csv, ensure_parent_dir
from shared.image_utils import encode_image_base64
from shared.ollama_client import ollama_generate
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
DEFAULT_VALIDATION_CSV = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/VLM_Test_Data_2026_03_04_v01.csv"
DEFAULT_MODEL = "llama3.2-vision:11b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

DEFAULT_OUTPUT_CSV = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/llm_validation_predictions.csv"
DEFAULT_METRICS_CSV = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/llm_validation_metrics.csv"
DEFAULT_ERRORS_CSV = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/llm_validation_errors.csv"


# ----------------------------
# Ollama VLM call (uses generate endpoint)
# ----------------------------
def call_ollama_vlm(model, image_path, question, ollama_url=DEFAULT_OLLAMA_URL, timeout=180):
    prompt = build_yesno_prompt(question)
    image_b64 = encode_image_base64(image_path)
    return ollama_generate(
        model=model,
        prompt=prompt,
        image_b64=image_b64,
        url=ollama_url,
        timeout=timeout,
    )


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run validation-only LLM mosaic QA and compute metrics."
    )
    parser.add_argument("--base_path", type=str, default=DEFAULT_BASE_PATH)
    parser.add_argument("--validation_csv", type=str, default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--ollama_url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--metrics_csv", type=str, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--errors_csv", type=str, default=DEFAULT_ERRORS_CSV)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip rows already present in output_csv based on SOPInstanceUID + Question."
    )
    args = parser.parse_args()

    print(f"[{now_ts()}] Using model: {args.model}")
    print(f"[{now_ts()}] Ollama URL: {args.ollama_url}")
    print(f"[{now_ts()}] Temperature: 0")
    print(f"[{now_ts()}] Loading validation CSV: {args.validation_csv}")
    val_df = load_validation_csv(args.validation_csv)
    print(f"[{now_ts()}] Validation rows loaded: {len(val_df)}")

    print(f"[{now_ts()}] Building sequence index from: {args.base_path}")
    seq_index = build_sequence_index(args.base_path)
    print(f"[{now_ts()}] Indexed sequences with metadata + mosaic: {len(seq_index)}")

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

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Running LLM QA"):
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
                    "Model Name": args.model,
                    "AccessionNumber": accession_from_val,
                    "SOPInstanceUID": sop_uid,
                    "Question": question,
                    "status": "NO_MATCHING_SEQUENCE",
                    "details": "Could not find sequence dir by SOPInstanceUID (case-insensitive).",
                },
                error_fieldnames
            )
            continue

        accession = seq_info["accession_number"] or accession_from_val
        mosaic_path = seq_info["mosaic_path"]
        sequence_dir = seq_info["sequence_dir"]

        try:
            raw_llm_output = call_ollama_vlm(
                model=args.model,
                image_path=mosaic_path,
                question=question,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
            )
            pred_answer = normalize_llm_answer(raw_llm_output)

            append_row_csv(
                args.output_csv,
                {
                    "Timestamp": now_ts(),
                    "Model Name": args.model,
                    "AccessionNumber": accession,
                    "SOPInstanceUID": sop_uid,
                    "Question": question,
                    "GroundTruth": gt_answer,
                    "Predicted": pred_answer,
                    "Raw_LLM_Output": raw_llm_output,
                    "sequence_dir": sequence_dir,
                    "mosaic_path": mosaic_path,
                },
                pred_fieldnames
            )

            all_gt.append(gt_answer)
            all_pred.append(pred_answer)
            processed += 1

        except Exception as e:
            errors += 1
            append_row_csv(
                args.errors_csv,
                {
                    "Timestamp": now_ts(),
                    "Model Name": args.model,
                    "AccessionNumber": accession,
                    "SOPInstanceUID": sop_uid,
                    "Question": question,
                    "status": "LLM_CALL_FAILED",
                    "details": str(e),
                },
                error_fieldnames
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
        "Model Name": args.model,
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
