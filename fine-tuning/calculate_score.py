#!/usr/bin/env python3

import pandas as pd
import sys
import argparse
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---- Default paths ----
DEFAULT_PRED_PATH = "/data/Deep_Angiography/AngioVision/fine-tuning/output/clip_binary_qa_predictions.csv"
DEFAULT_GT_PATH   = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/VLM_Test_Data_2026_03_04_v01.csv"
DEFAULT_UNMATCHED_GT_OUT = "/data/Deep_Angiography/AngioVision/fine-tuning/output/unmatched_gt_rows.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP binary QA predictions against ground truth."
    )

    parser.add_argument(
        "--pred_path",
        type=str,
        default=DEFAULT_PRED_PATH,
        help=f"Path to prediction CSV (default: {DEFAULT_PRED_PATH})"
    )

    parser.add_argument(
        "--gt_path",
        type=str,
        default=DEFAULT_GT_PATH,
        help=f"Path to ground truth CSV (default: {DEFAULT_GT_PATH})"
    )

    parser.add_argument(
        "--unmatched_gt_out",
        type=str,
        default=DEFAULT_UNMATCHED_GT_OUT,
        help=f"Output CSV for GT rows not matched with predictions (default: {DEFAULT_UNMATCHED_GT_OUT})"
    )

    return parser.parse_args()


def normalize_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def keep_yes_no(df: pd.DataFrame, col: str, tag: str) -> pd.DataFrame:
    before = len(df)
    df = df[df[col].isin(["yes", "no"])].copy()
    after = len(df)
    print(f"[{tag}] kept yes/no only in '{col}': {before} -> {after}")
    return df


def require_cols(df: pd.DataFrame, required, tag: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: {tag} file missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(2)


def main():
    args = parse_args()

    print("\nUsing files:")
    print(f"Predictions:        {args.pred_path}")
    print(f"Ground Truth:       {args.gt_path}")
    print(f"Unmatched GT file:  {args.unmatched_gt_out}\n")

    pred = pd.read_csv(args.pred_path)
    gt = pd.read_csv(args.gt_path)

    # --- verify required columns exist
    pred_required = ["AccessionNumber", "SOPInstanceUID", "Question", "Answer"]
    gt_required   = ["Accession", "SOPInstanceUID", "Question", "Answer"]

    require_cols(pred, pred_required, tag="PRED")
    require_cols(gt, gt_required, tag="GT")

    # --- normalize all key fields to make matching case-insensitive
    pred["AccessionNumber"] = normalize_str_series(pred["AccessionNumber"])
    pred["SOPInstanceUID"]  = normalize_str_series(pred["SOPInstanceUID"])
    pred["Question"]        = normalize_str_series(pred["Question"])
    pred["Answer"]          = normalize_str_series(pred["Answer"])

    gt["Accession"]         = normalize_str_series(gt["Accession"])
    gt["SOPInstanceUID"]    = normalize_str_series(gt["SOPInstanceUID"])
    gt["Question"]          = normalize_str_series(gt["Question"])
    gt["Answer"]            = normalize_str_series(gt["Answer"])

    # --- keep only yes/no after normalization
    pred = keep_yes_no(pred, "Answer", "PRED")
    gt   = keep_yes_no(gt, "Answer", "GT")

    # --- standardize names
    pred_std = pred.rename(columns={
        "AccessionNumber": "accession",
        "SOPInstanceUID": "sopinstanceuid",
        "Question": "question",
        "Answer": "answer_pred",
    })[["accession", "sopinstanceuid", "question", "answer_pred"]].copy()

    gt_std = gt.rename(columns={
        "Accession": "accession",
        "SOPInstanceUID": "sopinstanceuid",
        "Question": "question",
        "Answer": "answer_gt",
    })[["accession", "sopinstanceuid", "question", "answer_gt"]].copy()

    # --- merge using normalized lowercase fields
    merged = pd.merge(
        pred_std,
        gt_std,
        how="inner",
        on=["accession", "sopinstanceuid", "question"]
    )

    print("\nCounts:")
    print(f"Pred rows (yes/no): {len(pred_std)}")
    print(f"GT rows   (yes/no): {len(gt_std)}")
    print(f"Matched rows:       {len(merged)}")

    # --- find GT rows not present in predictions
    pred_keys = pred_std[["accession", "sopinstanceuid", "question"]].drop_duplicates()

    gt_unmatched = pd.merge(
        gt_std,
        pred_keys,
        how="left",
        on=["accession", "sopinstanceuid", "question"],
        indicator=True
    )

    gt_unmatched = gt_unmatched[gt_unmatched["_merge"] == "left_only"].copy()
    gt_unmatched = gt_unmatched.drop(columns=["_merge"])

    gt_unmatched.to_csv(args.unmatched_gt_out, index=False)

    print(f"GT rows not in matched rows: {len(gt_unmatched)}")
    print(f"Saved to: {args.unmatched_gt_out}")

    if len(merged) == 0:
        print("\nERROR: No matches after merge.")
        print("Possible causes:")
        print("- Accession formatting differs")
        print("- SOPInstanceUID mismatch")
        print("- Question text differs")
        print("- Extra spaces or punctuation differences remain")
        sys.exit(1)

    # --- map yes/no to 1/0
    y_true = merged["answer_gt"].map({"yes": 1, "no": 0})
    y_pred = merged["answer_pred"].map({"yes": 1, "no": 0})

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n===== CONFUSION MATRIX =====")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")

    print("\n===== METRICS =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    main()