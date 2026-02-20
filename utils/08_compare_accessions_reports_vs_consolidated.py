#!/usr/bin/env python3
"""
compare_accessions_reports_vs_consolidated.py

Compares accession numbers between:

1) /data/Deep_Angiography/Reports/Report_List_v01_01.csv
2) /data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_ALL_Sequences.csv

Outputs:
  - Prints counts to console
  - Writes a detailed Markdown analysis summary to:
      /data/Deep_Angiography/DICOM-metadata-stats/accession_overlap_analysis.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Iterable, Optional, Set, List

import pandas as pd


REPORT_CSV = Path("/data/Deep_Angiography/Reports/Report_List_v01_01.csv")
CONSOLIDATED_CSV = Path("/data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_ALL_Sequences.csv")
OUT_DIR = Path("/data/Deep_Angiography/DICOM-metadata-stats")
OUT_MD = OUT_DIR / "accession_overlap_analysis.md"


REPORT_ACCESSION_CANDIDATES = [
    "AccessionNumber",
    "Accession",
    "Anon Acc #",
    "Anon Acc#",
    "AnonAcc",
]

CONSOLIDATED_ACCESSION_CANDIDATES = [
    "AccessionNumber",
    "Accession",
]


def norm_accession(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns.astype(str))
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        c = lower_map.get(cand.lower())
        if c:
            return c
    return None


def split_accessions_cell(val: str) -> Iterable[str]:
    s = norm_accession(val)
    if not s:
        return []
    seps = [",", ";", "|"]
    tokens = [s]
    for sep in seps:
        if any(sep in t for t in tokens):
            new_tokens = []
            for t in tokens:
                new_tokens.extend(t.split(sep))
            tokens = new_tokens
    return [norm_accession(t) for t in tokens if norm_accession(t)]


def extract_accession_set(df: pd.DataFrame, col: str) -> Set[str]:
    accs: Set[str] = set()
    for v in df[col].tolist():
        for tok in split_accessions_cell(v):
            accs.add(tok)
    return accs


def main() -> int:

    if not REPORT_CSV.exists():
        print(f"[ERROR] Missing: {REPORT_CSV}", file=sys.stderr)
        return 2

    if not CONSOLIDATED_CSV.exists():
        print(f"[ERROR] Missing: {CONSOLIDATED_CSV}", file=sys.stderr)
        return 2

    report_df = pd.read_csv(REPORT_CSV)
    cons_df = pd.read_csv(CONSOLIDATED_CSV)

    report_row_count = len(report_df)
    cons_row_count = len(cons_df)

    report_col = find_column(report_df, REPORT_ACCESSION_CANDIDATES)
    cons_col = find_column(cons_df, CONSOLIDATED_ACCESSION_CANDIDATES)

    if not report_col:
        print(f"[ERROR] No accession column found in {REPORT_CSV}", file=sys.stderr)
        return 2

    if not cons_col:
        print(f"[ERROR] No accession column found in {CONSOLIDATED_CSV}", file=sys.stderr)
        return 2

    report_accs = extract_accession_set(report_df, report_col)
    cons_accs = extract_accession_set(cons_df, cons_col)

    only_in_report = sorted(report_accs - cons_accs)
    only_in_cons = sorted(cons_accs - report_accs)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    md_lines = []
    md_lines.append("# Accession Overlap Analysis\n\n")
    md_lines.append(f"**Generated on:** {ts}\n\n")

    md_lines.append("## Dataset Overview\n\n")
    md_lines.append(f"**Dataset A:** `{REPORT_CSV}`\n\n")
    md_lines.append(f"- Count of entries in `{REPORT_CSV}`: {report_row_count}\n")
    md_lines.append(f"- Unique accession numbers in `{REPORT_CSV}`: {len(report_accs)}\n\n")

    md_lines.append(f"**Dataset B:** `{CONSOLIDATED_CSV}`\n\n")
    md_lines.append(f"- Count of entries in `{CONSOLIDATED_CSV}`: {cons_row_count}\n")
    md_lines.append(f"- Unique accession numbers in `{CONSOLIDATED_CSV}`: {len(cons_accs)}\n\n")

    md_lines.append("---------------------------------------------------------\n\n")

    # Section 1
    md_lines.append(
        f"## Accession Numbers found only in `{REPORT_CSV}`, but not in `{CONSOLIDATED_CSV}`\n\n"
    )
    md_lines.append(f"**Count:** {len(only_in_report)}\n\n")
    md_lines.append("**Accession Numbers:**\n\n")

    if not only_in_report:
        md_lines.append("_None_\n\n")
    else:
        for acc in only_in_report:
            md_lines.append(f"- {acc}\n")
        md_lines.append("\n")

    md_lines.append("---------------------------------------------------------\n\n")

    # Section 2
    md_lines.append(
        f"## Accession Numbers found only in `{CONSOLIDATED_CSV}`, but not in `{REPORT_CSV}`\n\n"
    )
    md_lines.append(f"**Count:** {len(only_in_cons)}\n\n")
    md_lines.append("**Accession Numbers:**\n\n")

    if not only_in_cons:
        md_lines.append("_None_\n")
    else:
        for acc in only_in_cons:
            md_lines.append(f"- {acc}\n")

    OUT_MD.write_text("".join(md_lines), encoding="utf-8")

    print(f"[OK] Markdown report written to: {OUT_MD}")
    print(f"[INFO] Entries in {REPORT_CSV}: {report_row_count}")
    print(f"[INFO] Entries in {CONSOLIDATED_CSV}: {cons_row_count}")
    print(f"[INFO] Only in Report: {len(only_in_report)}")
    print(f"[INFO] Only in Consolidated: {len(only_in_cons)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())