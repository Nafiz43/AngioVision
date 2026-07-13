"""Shared QA benchmark loop: any answering backend over the validation set.

Both baseline steps (01 Ollama VLMs, 02 zero-shot CLIP) run the exact same
loop — resume via skip_existing, unmatched/error rows, per-model metrics —
differing only in how a (mosaic, question) pair is answered. The backend is
passed in as `answer_fn(mosaic_path, question) -> (predicted_yes_no, raw_output)`.

All backends write the same predictions schema, so the statistics step
consumes them uniformly.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm

from . import common

PRED_FIELDS = [
    "Timestamp", "Model Name", "AccessionNumber", "SOPInstanceUID", "Question",
    "GroundTruth", "Predicted", "Raw_LLM_Output", "sequence_dir", "mosaic_path",
]
ERROR_FIELDS = [
    "Timestamp", "Model Name", "AccessionNumber", "SOPInstanceUID", "Question",
    "status", "details",
]


def _existing_keys(output_csv: str) -> set:
    if not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv)
        if not {"SOPInstanceUID", "Question"}.issubset(df.columns):
            return set()
        return set(zip(
            df["SOPInstanceUID"].astype(str).map(common.normalize_text),
            df["Question"].astype(str).map(common.normalize_text),
        ))
    except Exception as e:
        print(f"[WARN] could not read {output_csv} for skip_existing: {e}")
        return set()


def run_qa_benchmark(model_label: str, answer_fn, val_df: pd.DataFrame,
                     seq_index: dict, cfg, metrics_csv: str) -> dict:
    """Run one model over every validation row; write predictions + metrics."""
    tag = common.sanitize_model_tag(model_label)
    output_csv = os.path.join(cfg.baselines_dir, f"{tag}_predictions.csv")
    errors_csv = os.path.join(cfg.baselines_dir, f"{tag}_errors.csv")

    existing = _existing_keys(output_csv) if cfg.skip_existing else set()
    if existing:
        print(f"[{model_label}] resuming — {len(existing)} rows already predicted")

    all_gt, all_pred = [], []
    processed = skipped = errors = unmatched = 0

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc=model_label):
        key = (row["SOP_norm"], row["Question_norm"])
        if key in existing:
            skipped += 1
            continue

        base = {
            "Timestamp": common.now_ts(),
            "Model Name": model_label,
            "AccessionNumber": str(row.get("AccessionNumber", "")).strip(),
            "SOPInstanceUID": str(row["SOPInstanceUID"]).strip(),
            "Question": str(row["Question"]).strip(),
        }

        seq_info = seq_index.get(row["SOP_norm"])
        if seq_info is None:
            unmatched += 1
            common.append_row_csv(errors_csv, {
                **base, "status": "NO_MATCHING_SEQUENCE",
                "details": "No sequence dir found by SOPInstanceUID.",
            }, ERROR_FIELDS)
            continue

        base["AccessionNumber"] = (
            seq_info["accession_number"] or base["AccessionNumber"]
        )
        try:
            pred, raw = answer_fn(seq_info["mosaic_path"], base["Question"])
            common.append_row_csv(output_csv, {
                **base,
                "GroundTruth": row["GT_Answer"],
                "Predicted": pred,
                "Raw_LLM_Output": raw,
                "sequence_dir": seq_info["sequence_dir"],
                "mosaic_path": seq_info["mosaic_path"],
            }, PRED_FIELDS)
            all_gt.append(row["GT_Answer"])
            all_pred.append(pred)
            processed += 1
        except Exception as e:
            errors += 1
            common.append_row_csv(errors_csv, {
                **base, "status": "MODEL_CALL_FAILED", "details": str(e),
            }, ERROR_FIELDS)

    summary = {
        "model": model_label, "predictions_csv": output_csv,
        "processed": processed, "skipped_existing": skipped,
        "unmatched": unmatched, "errors": errors,
    }

    if all_gt:
        y_true = [1 if x == "YES" else 0 for x in all_gt]
        y_pred = [1 if x == "YES" else 0 for x in all_pred]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics_row = {
            "Timestamp": common.now_ts(),
            "Model Name": model_label,
            "Temperature": 0,
            "Validation CSV": cfg.validation_csv,
            "Mosaics Root": cfg.mosaics_root,
            "Total Evaluated Rows": len(all_gt),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Accuracy": round(accuracy_score(y_true, y_pred), 3),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "Recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "F1 Score": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "Processed": processed, "Skipped Existing": skipped,
            "Unmatched Sequences": unmatched, "Errors": errors,
        }
        common.append_row_csv(metrics_csv, metrics_row, list(metrics_row))
        summary["accuracy"] = metrics_row["Accuracy"]
        summary["f1"] = metrics_row["F1 Score"]
    else:
        print(f"[{model_label}] no new predictions — metrics row not written")

    return summary


def load_inputs(cfg):
    """Validation rows + sequence index, shared by both baseline steps."""
    os.makedirs(cfg.baselines_dir, exist_ok=True)
    val_df = common.load_validation_csv(cfg.validation_csv)
    if getattr(cfg, "qa_limit", 0) > 0:
        val_df = val_df.head(cfg.qa_limit)
    print(f"validation rows: {len(val_df)}")
    seq_index = common.build_sequence_index(cfg.mosaics_root)
    print(f"indexed sequences (mosaic + metadata): {len(seq_index)}")
    return val_df, seq_index
