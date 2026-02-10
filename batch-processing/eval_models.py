#!/usr/bin/env python3
"""
eval_yesno_by_model.py

Evaluate Yes/No performance per model using:
- "Model Name" (model identifier)
- "Answer" (ground truth)
- "llm-answer" (model output)

Rules:
1) Lowercase ALL column names to avoid case ambiguity.
2) Consider ONLY rows where GT ("answer") is yes/no.
3) From those rows, IGNORE any row where prediction ("llm-answer") is NOT yes/no.
4) For remaining rows, compute TP, TN, FP, FN per model.
   Treat "yes" as positive class, "no" as negative class.

Usage:
  python eval_yesno_by_model.py --csv /path/to/file.csv
  python eval_yesno_by_model.py --csv merged.csv --out results.csv
"""

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


YES_TOKENS = {"yes", "y", "true", "1"}
NO_TOKENS = {"no", "n", "false", "0"}


def normalize_yesno(val: object) -> Optional[str]:
    """
    Map a cell value to 'yes' / 'no' if it clearly expresses a binary answer.
    Otherwise return None.

    Handles:
    - exact yes/no
    - common variants (y/n, true/false, 1/0)
    - leading answers like "Yes.", "No - ..."
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None

    s = str(val).strip().lower()
    if not s:
        return None

    # Take first "word-like" token (so "yes." => "yes", "no - ..." => "no")
    m = re.match(r"^\s*([a-z0-9]+)", s)
    token = m.group(1) if m else s

    if token in YES_TOKENS:
        return "yes"
    if token in NO_TOKENS:
        return "no"
    return None


def confusion_counts(gt: pd.Series, pred: pd.Series) -> Tuple[int, int, int, int]:
    """
    Compute TP, TN, FP, FN where:
    positive = 'yes', negative = 'no'
    """
    tp = int(((gt == "yes") & (pred == "yes")).sum())
    tn = int(((gt == "no") & (pred == "no")).sum())
    fp = int(((gt == "no") & (pred == "yes")).sum())
    fn = int(((gt == "yes") & (pred == "no")).sum())
    return tp, tn, fp, fn


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path, help="Input CSV file")
    ap.add_argument("--out", type=Path, default=None, help="Optional output CSV for per-model results")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # 1) lowercase column names + strip whitespace
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Required columns (lowercased)
    required = {"model name", "answer", "llm-answer"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Missing required columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Normalize yes/no from GT + pred
    df["_gt"] = df["answer"].apply(normalize_yesno)
    df["_pred"] = df["llm-answer"].apply(normalize_yesno)

    # 2) keep only rows where GT is yes/no
    gt_yesno = df[df["_gt"].isin(["yes", "no"])].copy()

    # 3) ignore rows where pred isn't yes/no
    eval_df = gt_yesno[gt_yesno["_pred"].isin(["yes", "no"])].copy()

    if eval_df.empty:
        raise SystemExit(
            "No evaluatable rows after filtering.\n"
            "Check that 'answer' contains yes/no and 'llm-answer' contains yes/no for those rows."
        )

    # Per-model confusion counts
    rows = []
    for model, g in eval_df.groupby("model name", dropna=False):
        tp, tn, fp, fn = confusion_counts(g["_gt"], g["_pred"])
        total = tp + tn + fp + fn
        acc = safe_div(tp + tn, total)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)

        rows.append(
            {
                "model name": model,
                "n_evaluated": total,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": acc,
                "precision_yes": precision,
                "recall_yes": recall,
            }
        )

    results = pd.DataFrame(rows).sort_values(by=["n_evaluated", "model name"], ascending=[False, True])

    # Print summary
    print("\n=== Filtering Summary ===")
    print(f"Total rows in CSV:                 {len(df)}")
    print(f"Rows with GT yes/no (answer):      {len(gt_yesno)}")
    print(f"Rows evaluated (GT yes/no AND pred yes/no): {len(eval_df)}")
    print(f"Rows ignored due to non-yes/no pred:        {len(gt_yesno) - len(eval_df)}")

    print("\n=== Per-model Confusion Counts (yes = positive) ===")
    # Pretty print without index
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(results.to_string(index=False))

    # Optional: overall counts
    overall_tp, overall_tn, overall_fp, overall_fn = confusion_counts(eval_df["_gt"], eval_df["_pred"])
    overall_total = overall_tp + overall_tn + overall_fp + overall_fn
    print("\n=== Overall (all models combined) ===")
    print(f"TP={overall_tp}  TN={overall_tn}  FP={overall_fp}  FN={overall_fn}  (N={overall_total})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False)
        print(f"\nSaved per-model results to: {args.out}")


if __name__ == "__main__":
    main()
