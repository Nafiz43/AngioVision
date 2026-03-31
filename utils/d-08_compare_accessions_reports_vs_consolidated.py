#!/usr/bin/env python3
"""
compare_accessions_reports_vs_consolidated_with_duplicates.py

Merges:
- CODE 1 (overlap analysis between report list and consolidated metadata)
- CODE 2 (duplicate accession detection in consolidated metadata + StudyInstanceUID mapping)

Inputs:
  - /data/Deep_Angiography/Reports/Report_List_v01_01.csv
  - /data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_ALL_Sequences.csv

Output:
  - Writes a detailed Markdown report to:
      /data/Deep_Angiography/DICOM-metadata-stats/accession_overlap_analysis.md

Report includes:
  - Entry counts
  - Unique accession token counts
  - Accessions only in report vs only in consolidated (full lists)
  - Duplicate accessions in consolidated + their StudyInstanceUIDs (full lists)
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Iterable, Optional, Set, List, Dict

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


def explode_accessions(df: pd.DataFrame, accession_col: str) -> pd.DataFrame:
    """
    Returns a dataframe with one accession token per row.
    Requires 'StudyInstanceUID' column to exist if caller needs study mapping.
    """
    exploded = (
        df.assign(
            AccessionToken=df[accession_col]
            .fillna("")
            .astype(str)
            .str.split(",")
        )
        .explode("AccessionToken")
    )
    exploded["AccessionToken"] = exploded["AccessionToken"].astype(str).str.strip()
    exploded = exploded[exploded["AccessionToken"] != ""]
    return exploded


def main() -> int:
    # ---- Validate inputs ----
    if not REPORT_CSV.exists():
        print(f"[ERROR] Missing: {REPORT_CSV}", file=sys.stderr)
        return 2
    if not CONSOLIDATED_CSV.exists():
        print(f"[ERROR] Missing: {CONSOLIDATED_CSV}", file=sys.stderr)
        return 2

    # ---- Load ----
    try:
        report_df = pd.read_csv(REPORT_CSV)
    except Exception as e:
        print(f"[ERROR] Failed reading {REPORT_CSV}: {e}", file=sys.stderr)
        return 2

    try:
        cons_df = pd.read_csv(CONSOLIDATED_CSV)
    except Exception as e:
        print(f"[ERROR] Failed reading {CONSOLIDATED_CSV}: {e}", file=sys.stderr)
        return 2

    # ---- Basic counts ----
    report_row_count = len(report_df)
    cons_row_count = len(cons_df)

    # ---- Identify accession columns ----
    report_col = find_column(report_df, REPORT_ACCESSION_CANDIDATES)
    cons_col = find_column(cons_df, CONSOLIDATED_ACCESSION_CANDIDATES)

    if not report_col:
        print(
            f"[ERROR] No accession column found in {REPORT_CSV}. "
            f"Tried: {REPORT_ACCESSION_CANDIDATES}. Found: {list(report_df.columns)}",
            file=sys.stderr,
        )
        return 2

    if not cons_col:
        print(
            f"[ERROR] No accession column found in {CONSOLIDATED_CSV}. "
            f"Tried: {CONSOLIDATED_ACCESSION_CANDIDATES}. Found: {list(cons_df.columns)}",
            file=sys.stderr,
        )
        return 2

    # ---- Extract unique accession tokens (set-based) ----
    report_accs = extract_accession_set(report_df, report_col)
    cons_accs = extract_accession_set(cons_df, cons_col)

    only_in_report = sorted(report_accs - cons_accs)
    only_in_cons = sorted(cons_accs - report_accs)

    # ---- Duplicate accessions in consolidated + StudyInstanceUID mapping ----
    if "StudyInstanceUID" not in cons_df.columns:
        print(
            f"[ERROR] '{CONSOLIDATED_CSV}' does not have 'StudyInstanceUID' column; "
            f"cannot map duplicates to studies. Found columns: {list(cons_df.columns)}",
            file=sys.stderr,
        )
        return 2

    exploded = explode_accessions(cons_df, cons_col)
    dup_counts = exploded["AccessionToken"].value_counts()
    dup_accessions = dup_counts[dup_counts > 1].index.tolist()

    dup_mapping: Dict[str, List[str]] = {}
    if dup_accessions:
        tmp = (
            exploded[exploded["AccessionToken"].isin(dup_accessions)]
            .groupby("AccessionToken")["StudyInstanceUID"]
            .apply(list)
            .to_dict()
        )
        # Normalize/clean StudyInstanceUID strings
        for acc, studies in tmp.items():
            dup_mapping[acc] = [str(s).strip() for s in studies if str(s).strip()]

    # ---- Write Markdown report ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_lines = []
    md_lines.append("# Accession Overlap Analysis (with Duplicate Accession Mapping)\n\n")
    md_lines.append(f"**Generated on:** {ts}\n\n")

    md_lines.append("## Dataset Overview\n\n")
    md_lines.append(f"**Reports dataset:** `{REPORT_CSV}`\n\n")
    md_lines.append(f"- Count of entries in `{REPORT_CSV}`: {report_row_count}\n")
    md_lines.append(f"- Accession column used: `{report_col}`\n")
    md_lines.append(f"- Unique accession numbers in `{REPORT_CSV}` (token-level): {len(report_accs)}\n\n")

    md_lines.append(f"**Consolidated metadata dataset:** `{CONSOLIDATED_CSV}`\n\n")
    md_lines.append(f"- Count of entries in `{CONSOLIDATED_CSV}`: {cons_row_count}\n")
    md_lines.append(f"- Accession column used: `{cons_col}`\n")
    md_lines.append(f"- Unique accession numbers in `{CONSOLIDATED_CSV}` (token-level): {len(cons_accs)}\n\n")

    md_lines.append(
        "### Notes on what “unique accession numbers” means here\n"
        "- The script extracts accession values, **splits comma/semicolon/pipe-separated cells into individual tokens**, "
        "normalizes whitespace, and then counts **distinct tokens** using a Python `set`.\n"
        "- Therefore, if two rows contain the same accession token, it is counted **once** in the “unique” metric.\n\n"
    )

    md_lines.append("---------------------------------------------------------\n\n")

    # Required format block 1
    md_lines.append(
        f"## Accession Numbers found only in `{REPORT_CSV}`, but not in `{CONSOLIDATED_CSV}`\n"
    )
    md_lines.append(f"**Count:** {len(only_in_report)}\n\n")
    md_lines.append("**Accession Numbers:**\n")
    md_lines.append("---------------------------------------------------------\n")
    if not only_in_report:
        md_lines.append("_None_\n\n")
    else:
        for acc in only_in_report:
            md_lines.append(f"- {acc}\n")
        md_lines.append("\n")

    # Required format block 2
    md_lines.append(
        f"## Accession Numbers found only in `{CONSOLIDATED_CSV}`, but not in `{REPORT_CSV}`\n"
    )
    md_lines.append(f"**Count:** {len(only_in_cons)}\n\n")
    md_lines.append("**Accession Numbers:**\n")
    md_lines.append("---------------------------------------------------------\n")
    if not only_in_cons:
        md_lines.append("_None_\n\n")
    else:
        for acc in only_in_cons:
            md_lines.append(f"- {acc}\n")
        md_lines.append("\n")

    md_lines.append("---------------------------------------------------------\n\n")

    # Duplicate section
    md_lines.append(f"## Duplicate Accession Numbers in `{CONSOLIDATED_CSV}` and their StudyInstanceUIDs\n\n")
    md_lines.append(
        "This section identifies accession numbers that appear in more than one row (after exploding comma-separated lists) "
        "within the consolidated metadata file, and lists the corresponding StudyInstanceUIDs.\n\n"
    )
    md_lines.append(f"**Count of duplicated accession numbers:** {len(dup_accessions)}\n\n")

    if not dup_accessions:
        md_lines.append("_No duplicated accession numbers found in the consolidated metadata._\n")
    else:
        # Sort by occurrence count desc, then accession
        dup_accessions_sorted = sorted(dup_accessions, key=lambda a: (-int(dup_counts[a]), a))

        for acc in dup_accessions_sorted:
            studies = dup_mapping.get(acc, [])
            md_lines.append(f"### Accession: `{acc}`\n")
            md_lines.append(f"- Occurrences (token-level): {int(dup_counts[acc])}\n")
            md_lines.append(f"- StudyInstanceUIDs (count={len(studies)}):\n")
            for s in studies:
                md_lines.append(f"  - {s}\n")
            md_lines.append("\n")

    OUT_MD.write_text("".join(md_lines), encoding="utf-8")

    # ---- Print console summary ----
    print(f"[OK] Markdown report written to: {OUT_MD}")
    print(f"[INFO] Entries in {REPORT_CSV}: {report_row_count}")
    print(f"[INFO] Entries in {CONSOLIDATED_CSV}: {cons_row_count}")
    print(f"[INFO] Only in {REPORT_CSV}: {len(only_in_report)}")
    print(f"[INFO] Only in {CONSOLIDATED_CSV}: {len(only_in_cons)}")
    print(f"[INFO] Duplicated accessions in {CONSOLIDATED_CSV}: {len(dup_accessions)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())