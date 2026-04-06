#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---- Default paths ----
DEFAULT_PRED_PATH = "/data/Deep_Angiography/AngioVision/fine-tuning/output/clip_binary_qa_predictions.csv"
DEFAULT_GT_PATH = "/data/Deep_Angiography/Validation_Data/test-data/gt.csv"
DEFAULT_UNMATCHED_GT_OUT = "/data/Deep_Angiography/AngioVision/fine-tuning/output/unmatched_gt_rows.csv"
DEFAULT_RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP binary QA predictions against ground truth."
    )

    parser.add_argument(
        "--pred_path",
        type=str,
        default=DEFAULT_PRED_PATH,
        help=f"Path to prediction CSV (default: {DEFAULT_PRED_PATH})",
    )

    parser.add_argument(
        "--gt_path",
        type=str,
        default=DEFAULT_GT_PATH,
        help=f"Path to ground truth CSV (default: {DEFAULT_GT_PATH})",
    )

    parser.add_argument(
        "--unmatched_gt_out",
        type=str,
        default=DEFAULT_UNMATCHED_GT_OUT,
        help=f"Output CSV for GT rows not matched with predictions (default: {DEFAULT_UNMATCHED_GT_OUT})",
    )

    # Kept only for backward compatibility; no rewrite is performed anymore.
    parser.add_argument(
        "--rewrite_pred_if_flipped",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag. Predictions are no longer rewritten.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for coin-flip baseline (default: {DEFAULT_RANDOM_SEED})",
    )

    return parser.parse_args()


def safe_print_header(args):
    print("\nUsing files:")
    print(f"Predictions:        {args.pred_path}")
    print(f"Ground Truth:       {args.gt_path}")
    print(f"Unmatched GT file:  {args.unmatched_gt_out}")
    print(f"Rewrite flipped:    {args.rewrite_pred_if_flipped} (deprecated; ignored)")
    print(f"Random seed:        {args.random_seed}\n")


def normalize_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()


def keep_yes_no(df: pd.DataFrame, col: str, tag: str) -> pd.DataFrame:
    before = len(df)
    if col not in df.columns:
        print(f"[{tag}] WARNING: column '{col}' not found; returning empty dataframe.")
        return df.iloc[0:0].copy()

    df = df.copy()
    df[col] = normalize_str_series(df[col])
    df = df[df[col].isin(["yes", "no"])].copy()
    after = len(df)
    print(f"[{tag}] kept yes/no only in '{col}': {before} -> {after}")
    return df


def require_cols(df: pd.DataFrame, required, tag: str) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: {tag} file missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return False
    return True


def safe_read_csv(path_str: str, tag: str):
    path = Path(path_str)

    if not path.exists():
        print(f"ERROR: {tag} file does not exist: {path}")
        return None

    if not path.is_file():
        print(f"ERROR: {tag} path is not a file: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR: Failed to read {tag} CSV: {path}")
        print(f"Reason: {type(e).__name__}: {e}")
        return None

    print(f"[{tag}] loaded rows: {len(df)}")
    print(f"[{tag}] columns: {list(df.columns)}")

    if df.empty:
        print(f"WARNING: {tag} CSV is empty: {path}")

    return df


def safe_write_csv(df: pd.DataFrame, out_path: str, tag: str):
    try:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[{tag}] Saved to: {out}")
        return True
    except Exception as e:
        print(f"WARNING: Could not write {tag} CSV to {out_path}")
        print(f"Reason: {type(e).__name__}: {e}")
        return False


def empty_metrics_dict(reason: str):
    return {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "reason": reason,
    }


def compute_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return empty_metrics_dict("no_rows_for_scoring")

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except Exception:
        return empty_metrics_dict("confusion_matrix_failed")

    try:
        acc = accuracy_score(y_true, y_pred)
    except Exception:
        acc = 0.0

    try:
        precision = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
    except Exception:
        precision = 0.0

    try:
        recall = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
    except Exception:
        recall = 0.0

    try:
        f1 = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
    except Exception:
        f1 = 0.0

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "reason": "ok",
    }


def print_metrics_block(metrics, prefix: str):
    print(f"\n===== {prefix} RESULT =====")
    print(f"{prefix}_TP: {metrics['tp']}")
    print(f"{prefix}_TN: {metrics['tn']}")
    print(f"{prefix}_FP: {metrics['fp']}")
    print(f"{prefix}_FN: {metrics['fn']}")
    print(f"{prefix}_Accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix}_Precision: {metrics['precision']:.4f}")
    print(f"{prefix}_Recall: {metrics['recall']:.4f}")
    print(f"{prefix}_F1: {metrics['f1']:.4f}")
    if metrics.get("reason", "ok") != "ok":
        print(f"{prefix}_Reason: {metrics['reason']}")


def print_combined_summary(original_metrics, flipped_metrics):
    print("\n===== COMBINED SUMMARY (Original/Flipped) =====")
    print(f"TP: {original_metrics['tp']}/{flipped_metrics['tp']}")
    print(f"TN: {original_metrics['tn']}/{flipped_metrics['tn']}")
    print(f"FP: {original_metrics['fp']}/{flipped_metrics['fp']}")
    print(f"FN: {original_metrics['fn']}/{flipped_metrics['fn']}")
    print(f"Accuracy: {original_metrics['accuracy']:.4f}/{flipped_metrics['accuracy']:.4f}")
    print(f"Precision: {original_metrics['precision']:.4f}/{flipped_metrics['precision']:.4f}")
    print(f"Recall: {original_metrics['recall']:.4f}/{flipped_metrics['recall']:.4f}")
    print(f"F1-score: {original_metrics['f1']:.4f}/{flipped_metrics['f1']:.4f}")


def print_machine_readable_summary(metric_groups):
    """
    Emits rigid machine-readable lines for validate.py to parse reliably.
    metric_groups example:
      {
        "ORIGINAL": {...},
        "FLIPPED": {...},
        "ALL_YES": {...},
        "ALL_NO": {...},
        "RANDOM": {...},
      }
    """
    print("\n===== MACHINE_READABLE_SUMMARY =====")
    for prefix, metrics in metric_groups.items():
        print(f"{prefix}_TP: {metrics['tp']}")
        print(f"{prefix}_TN: {metrics['tn']}")
        print(f"{prefix}_FP: {metrics['fp']}")
        print(f"{prefix}_FN: {metrics['fn']}")
        print(f"{prefix}_ACCURACY: {metrics['accuracy']:.6f}")
        print(f"{prefix}_PRECISION: {metrics['precision']:.6f}")
        print(f"{prefix}_RECALL: {metrics['recall']:.6f}")
        print(f"{prefix}_F1: {metrics['f1']:.6f}")


def build_zero_groups(reason: str):
    return {
        "ORIGINAL": empty_metrics_dict(reason),
        "FLIPPED": empty_metrics_dict(reason),
        "ALL_YES": empty_metrics_dict(reason),
        "ALL_NO": empty_metrics_dict(reason),
        "RANDOM": empty_metrics_dict(reason),
    }


def main():
    args = parse_args()
    safe_print_header(args)

    pred_raw = safe_read_csv(args.pred_path, tag="PRED")
    gt = safe_read_csv(args.gt_path, tag="GT")

    if pred_raw is None or gt is None:
        print("\nERROR: Could not load one or both input files.")
        print("Returning without crashing so upstream validation pipeline can continue.")
        groups = build_zero_groups("input_load_failed")
        for k, v in groups.items():
            print_metrics_block(v, k)
        print_combined_summary(groups["ORIGINAL"], groups["FLIPPED"])
        print_machine_readable_summary(groups)
        sys.exit(0)

    pred_required = ["AccessionNumber", "SOPInstanceUID", "Question", "Answer"]
    gt_required = ["Accession", "SOPInstanceUID", "Question", "Answer"]

    pred_ok = require_cols(pred_raw, pred_required, tag="PRED")
    gt_ok = require_cols(gt, gt_required, tag="GT")

    if not pred_ok or not gt_ok:
        print("\nERROR: Required columns missing.")
        groups = build_zero_groups("missing_required_columns")
        for k, v in groups.items():
            print_metrics_block(v, k)
        print_combined_summary(groups["ORIGINAL"], groups["FLIPPED"])
        print_machine_readable_summary(groups)
        sys.exit(0)

    pred = pred_raw.copy()
    gt = gt.copy()

    pred["AccessionNumber"] = normalize_str_series(pred["AccessionNumber"])
    pred["SOPInstanceUID"] = normalize_str_series(pred["SOPInstanceUID"])
    pred["Question"] = normalize_str_series(pred["Question"])
    pred["Answer"] = normalize_str_series(pred["Answer"])

    gt["Accession"] = normalize_str_series(gt["Accession"])
    gt["SOPInstanceUID"] = normalize_str_series(gt["SOPInstanceUID"])
    gt["Question"] = normalize_str_series(gt["Question"])
    gt["Answer"] = normalize_str_series(gt["Answer"])

    pred = keep_yes_no(pred, "Answer", "PRED")
    gt = keep_yes_no(gt, "Answer", "GT")

    pred_std = pred.rename(
        columns={
            "AccessionNumber": "accession",
            "SOPInstanceUID": "sopinstanceuid",
            "Question": "question",
            "Answer": "answer_pred",
        }
    )[["accession", "sopinstanceuid", "question", "answer_pred"]].copy()

    gt_std = gt.rename(
        columns={
            "Accession": "accession",
            "SOPInstanceUID": "sopinstanceuid",
            "Question": "question",
            "Answer": "answer_gt",
        }
    )[["accession", "sopinstanceuid", "question", "answer_gt"]].copy()

    pred_before_dedup = len(pred_std)
    gt_before_dedup = len(gt_std)

    pred_std = pred_std.drop_duplicates(subset=["sopinstanceuid", "question"], keep="first").copy()
    gt_std = gt_std.drop_duplicates(subset=["sopinstanceuid", "question"], keep="first").copy()

    print("\nDeduplication:")
    print(f"Pred rows: {pred_before_dedup} -> {len(pred_std)}")
    print(f"GT rows:   {gt_before_dedup} -> {len(gt_std)}")

    merged = pd.merge(
        pred_std,
        gt_std,
        how="inner",
        on=["sopinstanceuid", "question"],
    )

    print("\nCounts:")
    print(f"Pred rows (yes/no, deduped): {len(pred_std)}")
    print(f"GT rows   (yes/no, deduped): {len(gt_std)}")
    print(f"Matched rows:                {len(merged)}")

    pred_keys = pred_std[["sopinstanceuid", "question"]].drop_duplicates()

    gt_unmatched = pd.merge(
        gt_std,
        pred_keys,
        how="left",
        on=["sopinstanceuid", "question"],
        indicator=True,
    )

    gt_unmatched = gt_unmatched[gt_unmatched["_merge"] == "left_only"].copy()
    gt_unmatched = gt_unmatched.drop(columns=["_merge"])

    safe_write_csv(gt_unmatched, args.unmatched_gt_out, tag="UNMATCHED_GT")
    print(f"GT rows not in matched rows: {len(gt_unmatched)}")

    if len(pred_std) == 0:
        print("\nWARNING: Prediction dataframe has zero valid YES/NO rows.")
        groups = build_zero_groups("pred_has_zero_valid_yes_no_rows")
        for k, v in groups.items():
            print_metrics_block(v, k)
        print_combined_summary(groups["ORIGINAL"], groups["FLIPPED"])
        print_machine_readable_summary(groups)
        sys.exit(0)

    if len(gt_std) == 0:
        print("\nWARNING: Ground-truth dataframe has zero valid YES/NO rows.")
        groups = build_zero_groups("gt_has_zero_valid_yes_no_rows")
        for k, v in groups.items():
            print_metrics_block(v, k)
        print_combined_summary(groups["ORIGINAL"], groups["FLIPPED"])
        print_machine_readable_summary(groups)
        sys.exit(0)

    if len(merged) == 0:
        print("\nWARNING: No matches after merge on ['sopinstanceuid', 'question'].")
        print("This usually means SOPInstanceUID/question pairs do not align between prediction and GT files.")
        groups = build_zero_groups("no_matches_after_merge")
        for k, v in groups.items():
            print_metrics_block(v, k)
        print_combined_summary(groups["ORIGINAL"], groups["FLIPPED"])
        print_machine_readable_summary(groups)
        sys.exit(0)

    y_true = merged["answer_gt"].map({"yes": 1, "no": 0})
    y_pred = merged["answer_pred"].map({"yes": 1, "no": 0})

    valid_mask = y_true.notna() & y_pred.notna()
    invalid_rows = (~valid_mask).sum()

    if invalid_rows > 0:
        print(f"\nWARNING: Dropping {invalid_rows} matched rows due to invalid label mapping.")

    y_true = y_true[valid_mask].astype(int)
    y_pred = y_pred[valid_mask].astype(int)

    if len(y_true) == 0:
        print("\nWARNING: All matched rows became invalid after label mapping.")
        groups = build_zero_groups("all_rows_invalid_after_label_mapping")
        for k, v in groups.items():
            print_metrics_block(v, k)
        print_combined_summary(groups["ORIGINAL"], groups["FLIPPED"])
        print_machine_readable_summary(groups)
        sys.exit(0)

    original_metrics = compute_metrics(y_true, y_pred)
    flipped_metrics = compute_metrics(y_true, 1 - y_pred)
    all_yes_metrics = compute_metrics(y_true, np.ones(len(y_true), dtype=int))
    all_no_metrics = compute_metrics(y_true, np.zeros(len(y_true), dtype=int))

    rng = np.random.default_rng(args.random_seed)
    random_pred = rng.integers(low=0, high=2, size=len(y_true), dtype=int)
    random_metrics = compute_metrics(y_true, random_pred)

    metric_groups = {
        "ORIGINAL": original_metrics,
        "FLIPPED": flipped_metrics,
        "ALL_YES": all_yes_metrics,
        "ALL_NO": all_no_metrics,
        "RANDOM": random_metrics,
    }

    for prefix, metrics in metric_groups.items():
        print_metrics_block(metrics, prefix)

    print_combined_summary(original_metrics, flipped_metrics)
    print_machine_readable_summary(metric_groups)

    print("\n[INFO] No prediction file rewriting is performed anymore.")
    print("✅ Finished successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()