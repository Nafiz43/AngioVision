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
import re
import csv
import base64
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tqdm import tqdm


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
# Utility functions
# ----------------------------
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_gt_answer(raw_answer: object) -> str:
    if pd.isna(raw_answer):
        return "NO"

    ans = normalize_text(raw_answer)

    yes_like = {"yes", "y", "true", "1", "present", "positive"}
    no_like = {
        "no", "n", "false", "0", "absent", "negative",
        "unclear", "not visible", "cannot say", "can not say",
        "can't say", "cant say", "unable to determine",
        "cannot determine", "can not determine", "not sure",
        "unknown", "indeterminate", "not identifiable",
        "not seen", "not clear", "not possible to determine",
    }

    if ans in yes_like:
        return "YES"
    if ans in no_like:
        return "NO"

    if "yes" in ans or "present" in ans or "positive" in ans:
        return "YES"

    if any(term in ans for term in [
        "no",
        "unclear",
        "not visible",
        "cannot say",
        "can not say",
        "can't say",
        "cant say",
        "unable to determine",
        "cannot determine",
        "can not determine",
        "not sure",
        "unknown",
        "indeterminate",
        "not identifiable",
        "not seen",
        "not clear",
        "not possible to determine",
        "absent",
        "negative",
    ]):
        return "NO"

    return "NO"


def normalize_llm_answer(raw_answer: object) -> str:
    if raw_answer is None:
        return "NO"

    ans = normalize_text(raw_answer)

    yes_terms = {"yes", "y"}
    no_terms = {
        "no", "n",
        "unclear", "not visible", "cannot say", "can not say",
        "cant say", "can't say",
        "unable to determine", "not sure", "unknown", "indeterminate",
        "not identifiable", "not seen", "not clear",
        "not possible to determine", "cannot determine", "can not determine",
    }

    if ans in yes_terms:
        return "YES"

    if ans in no_terms:
        return "NO"

    if "yes" in ans:
        return "YES"

    if any(term in ans for term in [
        "no",
        "unclear",
        "not visible",
        "cannot say",
        "can not say",
        "can't say",
        "cant say",
        "unable to determine",
        "cannot determine",
        "can not determine",
        "not sure",
        "unknown",
        "indeterminate",
        "not identifiable",
        "not seen",
        "not clear",
        "not possible to determine",
    ]):
        return "NO"

    return "NO"


def load_metadata_csv(metadata_path: str) -> Dict[str, str]:
    out = {}
    try:
        df = pd.read_csv(metadata_path)
        if "Information" not in df.columns or "Value" not in df.columns:
            return out

        for _, row in df.iterrows():
            key = normalize_text(row["Information"])
            val = "" if pd.isna(row["Value"]) else str(row["Value"]).strip()
            out[key] = val
    except Exception:
        return out

    return out


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(question: str) -> str:
    return (
        "You are analyzing a medical angiography mosaic image.\n"
        "Answer the question using only visible image evidence.\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Respond with exactly one word: YES or NO\n"
        "- Do not explain your answer\n"
        "- Do not add punctuation or extra words\n"
        "- If the finding is not clearly visible, answer NO\n"
        "- If the image is ambiguous, uncertain, low-quality, incomplete, or cannot confirm the finding, answer NO\n"
        "- Only answer YES when the finding is clearly supported by the image\n"
        "- Output must be exactly YES or NO\n"
    )


def call_ollama_vlm(
    model: str,
    image_path: str,
    question: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 180,
) -> str:
    prompt = build_prompt(question)
    image_b64 = encode_image_base64(image_path)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    try:
        resp = requests.post(
            ollama_url,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()
    except Exception:
        return "NO"


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_row_csv(csv_path: str, row: Dict, fieldnames: List[str]) -> None:
    ensure_parent_dir(csv_path)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ----------------------------
# Index building
# ----------------------------
def build_sequence_index(base_path: str) -> Dict[str, Dict[str, str]]:
    seq_index = {}

    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")

    for item in os.listdir(base_path):
        seq_dir = os.path.join(base_path, item)
        if not os.path.isdir(seq_dir):
            continue

        metadata_path = os.path.join(seq_dir, "metadata.csv")
        mosaic_path = os.path.join(seq_dir, "mosaic.png")

        if not os.path.isfile(metadata_path):
            continue
        if not os.path.isfile(mosaic_path):
            continue

        meta = load_metadata_csv(metadata_path)
        sop_uid = meta.get("sopinstanceuid", "").strip()
        accession = meta.get("accessionnumber", "").strip()

        if not sop_uid:
            continue

        seq_index[normalize_text(sop_uid)] = {
            "sop_uid": sop_uid,
            "accession_number": accession,
            "sequence_dir": seq_dir,
            "mosaic_path": mosaic_path,
            "metadata_path": metadata_path,
        }

    return seq_index


# ----------------------------
# Validation loading
# ----------------------------
def detect_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    norm_map = {normalize_text(c): c for c in df.columns}
    for cand in candidates:
        if normalize_text(cand) in norm_map:
            return norm_map[normalize_text(cand)]

    if required:
        raise ValueError(
            f"Could not find required column. Expected one of: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def load_validation_csv(validation_csv: str) -> pd.DataFrame:
    df = pd.read_csv(validation_csv)

    sop_col = detect_column(df, ["SOPInstanceUID", "sopinstanceuid", "SOP UID", "SOP_UID"])
    q_col = detect_column(df, ["Question", "question"])
    a_col = detect_column(df, ["Answer", "answer", "GroundTruth", "ground_truth"])

    accession_col = detect_column(
        df,
        ["AccessionNumber", "accessionnumber", "Accession Number"],
        required=False
    )

    out = pd.DataFrame({
        "SOPInstanceUID": df[sop_col].astype(str),
        "Question": df[q_col].astype(str),
        "GT_Answer_Raw": df[a_col].astype(str),
    })

    if accession_col is not None:
        out["AccessionNumber"] = df[accession_col].astype(str)
    else:
        out["AccessionNumber"] = ""

    out["SOP_norm"] = out["SOPInstanceUID"].apply(normalize_text)
    out["Question_norm"] = out["Question"].apply(normalize_text)
    out["GT_Answer"] = out["GT_Answer_Raw"].apply(normalize_gt_answer)

    out = out.dropna(subset=["SOPInstanceUID", "Question"])
    out = out[out["SOP_norm"] != ""].copy()
    out = out[out["Question_norm"] != ""].copy()

    return out


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
        "Timestamp",
        "Model Name",
        "AccessionNumber",
        "SOPInstanceUID",
        "Question",
        "GroundTruth",
        "Predicted",
        "Raw_LLM_Output",
        "sequence_dir",
        "mosaic_path",
    ]

    error_fieldnames = [
        "Timestamp",
        "Model Name",
        "AccessionNumber",
        "SOPInstanceUID",
        "Question",
        "status",
        "details",
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

    y_true = [1 if x == "YES" else 0 for x in all_gt]
    y_pred = [1 if x == "YES" else 0 for x in all_pred]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = round(accuracy_score(y_true, y_pred), 3)
    precision = round(precision_score(y_true, y_pred, zero_division=0), 3)
    recall = round(recall_score(y_true, y_pred, zero_division=0), 3)
    f1 = round(f1_score(y_true, y_pred, zero_division=0), 3)

    metrics_row = {
        "Timestamp": now_ts(),
        "Model Name": args.model,
        "Temperature": 0,
        "Validation CSV": args.validation_csv,
        "Base Path": args.base_path,
        "Total Evaluated Rows": len(all_gt),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
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

    print("\n=== FINAL METRICS ===")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"Metrics CSV saved to: {args.metrics_csv}")


if __name__ == "__main__":
    main()