#!/usr/bin/env python3

import pandas as pd
import sys
from sklearn.metrics import confusion_matrix, accuracy_score

PRED_PATH = "/data/Deep_Angiography/AngioVision/fine-tuning/output/clip_binary_qa_predictions.csv"
GT_PATH   = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/Test_Data_2026_02_01_v01.csv"


def normalize_str_series(s: pd.Series) -> pd.Series:
    # robust string normalization (handles NaN safely)
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
    pred = pd.read_csv(PRED_PATH)
    gt   = pd.read_csv(GT_PATH)

    # --- verify required columns exist (exact names)
    pred_required = ["AccessionNumber", "SOPInstanceUID", "Question", "Answer"]
    gt_required   = ["Accession", "SOPInstanceUID", "Question", "Answer"]

    require_cols(pred, pred_required, tag="PRED")
    require_cols(gt, gt_required, tag="GT")

    # --- normalize key fields + answers to lowercase immediately
    pred["AccessionNumber"] = normalize_str_series(pred["AccessionNumber"])
    pred["SOPInstanceUID"]  = normalize_str_series(pred["SOPInstanceUID"])
    pred["Question"]        = normalize_str_series(pred["Question"])
    pred["Answer"]          = normalize_str_series(pred["Answer"])

    gt["Accession"]        = normalize_str_series(gt["Accession"])
    gt["SOPInstanceUID"]   = normalize_str_series(gt["SOPInstanceUID"])
    gt["Question"]         = normalize_str_series(gt["Question"])
    gt["Answer"]           = normalize_str_series(gt["Answer"])

    # --- keep only yes/no answers in both
    pred = keep_yes_no(pred, "Answer", "PRED")
    gt   = keep_yes_no(gt, "Answer", "GT")

    # --- standardize to common names for merging
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

    # --- merge
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

    if len(merged) == 0:
        print("\nERROR: No matches after merge.")
        print("Most common causes:")
        print("- Accession formatting differs (e.g., leading/trailing spaces, different IDs)")
        print("- Question text differs (punctuation/spacing)")
        sys.exit(1)

    # --- map yes/no to 1/0
    y_true = merged["answer_gt"].map({"yes": 1, "no": 0})
    y_pred = merged["answer_pred"].map({"yes": 1, "no": 0})

    # --- confusion matrix (tn fp fn tp)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)

    print("\n===== CONFUSION MATRIX =====")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"\nAccuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
