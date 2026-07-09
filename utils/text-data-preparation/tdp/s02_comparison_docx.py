"""
Step 02 — Reviewer docx: Original vs Augmented
(from 18_text_report_comparison.py).

Reads the augmented CSV from step 01 and builds a 3-column fixed-width
table (Acc ID | Original | Augmented). One row per (original, augmented)
pair — the original text repeats for each variant so reviewers always see
both side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from tdp.common import DocxColumn, build_comparison_docx


def build_rows(df: pd.DataFrame, acc_col: str, text_col: str) -> List[Dict]:
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in ("Type", text_col, acc_col):
        df[col] = df[col].str.strip()

    originals = df[df["Type"] == "Original"]
    augmented = df[df["Type"] != "Original"].sort_values([acc_col, "Type"])

    rows: List[Dict] = []
    for _, orig in originals.iterrows():
        acc = orig[acc_col]
        aug_rows = augmented[augmented[acc_col] == acc]
        if aug_rows.empty:
            rows.append({"acc_id": acc, "original": orig[text_col],
                         "augmented": "—"})
        else:
            for _, aug in aug_rows.iterrows():
                rows.append({
                    "acc_id": f"{acc}\n({aug['Type']})",
                    "original": orig[text_col],
                    "augmented": aug[text_col],
                })
    return rows


def run(cfg, run_dir: Path, data_dir: Path) -> Dict:
    step_dir = run_dir / "02_comparison_docx"
    augmented_csv = data_dir / "augmented_reports.csv"
    if not augmented_csv.exists():
        raise FileNotFoundError(
            f"{augmented_csv} not found — run step 01 (augmentation) first.")

    df = pd.read_csv(augmented_csv, dtype=str, keep_default_na=False)
    acc_col = cfg.accession_column

    text_col = next((c for c in df.columns if c.startswith("cleaned_")), None)
    if text_col is None:
        text_col = cfg.report_column
    if text_col not in df.columns or "Type" not in df.columns:
        raise KeyError(f"Expected '{text_col}' and 'Type' columns in {augmented_csv}")

    rows = build_rows(df, acc_col, text_col)

    docx_path = build_comparison_docx(
        rows=rows,
        columns=[
            DocxColumn("acc_id", "Acc ID", 1.1, bold=True),
            DocxColumn("original", "Original", 3.2),
            DocxColumn("augmented", "Augmented", 3.2),
        ],
        title="Radiology Report Comparison — Original vs Augmented",
        subtitle=f"{len(rows)} row(s) · source: {augmented_csv.name}",
        out_path=step_dir / "report_comparison.docx",
    )

    summary = {"comparison_pairs": len(rows), "docx": str(docx_path)}
    print(f"[02] {summary}")
    return summary
