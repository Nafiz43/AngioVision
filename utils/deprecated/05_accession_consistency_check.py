"""
accession_diff.py
─────────────────────────────────────────────────────────────────────────────
Identifies accession numbers present in File 1 (filtering CSV) but ABSENT
in File 2 (consolidated DICOM metadata CSV), with descriptive statistics
printed at every processing stage.

File 1 : /data/Deep_Angiography/Deep_Angio_DB_v02/Filtering_IR_2026_04_17_v01.csv
         Column : "Accession Number"

File 2 : /data/Deep_Angiography/DICOM-metadata-stats/n_consolidated_metadata_ALL_Sequences.csv
         Column : "AccessionNumber"
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import textwrap
from pathlib import Path

import pandas as pd

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

FILE1_PATH   = Path("/data/Deep_Angiography/Reports/Report_List_v01_01.csv")
FILE2_PATH   = Path("/data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_ALL_Sequences.csv")

FILE1_COL    = "Anon Acc #"   # column name in File 1
FILE2_COL    = "AccessionNumber"    # column name in File 2

OUTPUT_CSV   = Path("accession_in_file1_not_in_file2.csv")   # written to CWD

# ─── HELPERS ──────────────────────────────────────────────────────────────────

SEP = "─" * 72

def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def stat(label: str, value) -> None:
    print(f"  {label:<45} {value}")

def abort(msg: str) -> None:
    print(f"\n[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)

def normalise_accession(series: pd.Series) -> pd.Series:
    """Strip whitespace and cast to string; treat NaN as empty string."""
    return series.fillna("").astype(str).str.strip()

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── 1. LOAD FILE 1 ────────────────────────────────────────────────────────
    section("STEP 1 — Loading File 1 (Filtering / IR list)")
    stat("Path", FILE1_PATH)

    if not FILE1_PATH.exists():
        abort(f"File 1 not found: {FILE1_PATH}")

    df1 = pd.read_csv(FILE1_PATH, low_memory=False)

    stat("Total rows (raw)", f"{len(df1):,}")
    stat("Total columns", f"{df1.shape[1]:,}")
    stat("Columns", ", ".join(df1.columns.tolist()))

    if FILE1_COL not in df1.columns:
        abort(
            f"Column '{FILE1_COL}' not found in File 1.\n"
            f"Available columns: {df1.columns.tolist()}"
        )

    # Accession column stats
    raw_col1 = df1[FILE1_COL]
    stat(f"'{FILE1_COL}' — total values", f"{len(raw_col1):,}")
    stat(f"'{FILE1_COL}' — missing (NaN)", f"{raw_col1.isna().sum():,}")
    stat(f"'{FILE1_COL}' — unique (raw, including NaN)", f"{raw_col1.nunique(dropna=False):,}")

    # Normalise
    df1["_acc1"] = normalise_accession(raw_col1)
    # Exclude truly empty strings (were NaN or blank)
    df1_valid = df1[df1["_acc1"] != ""].copy()

    stat(f"'{FILE1_COL}' — rows after dropping blank/NaN", f"{len(df1_valid):,}")
    stat(f"'{FILE1_COL}' — unique accession numbers (clean)", f"{df1_valid['_acc1'].nunique():,}")

    # Duplicate check within File 1
    dup1 = df1_valid[df1_valid.duplicated(subset="_acc1", keep=False)]
    stat(f"'{FILE1_COL}' — rows that are duplicates within File 1", f"{len(dup1):,}")
    unique_acc1: set = set(df1_valid["_acc1"].unique())

    # ── 2. LOAD FILE 2 ────────────────────────────────────────────────────────
    section("STEP 2 — Loading File 2 (Consolidated DICOM metadata)")
    stat("Path", FILE2_PATH)

    if not FILE2_PATH.exists():
        abort(f"File 2 not found: {FILE2_PATH}")

    df2 = pd.read_csv(FILE2_PATH, low_memory=False)

    stat("Total rows (raw)", f"{len(df2):,}")
    stat("Total columns", f"{df2.shape[1]:,}")

    if FILE2_COL not in df2.columns:
        abort(
            f"Column '{FILE2_COL}' not found in File 2.\n"
            f"Available columns: {df2.columns.tolist()}"
        )

    raw_col2 = df2[FILE2_COL]
    stat(f"'{FILE2_COL}' — total values", f"{len(raw_col2):,}")
    stat(f"'{FILE2_COL}' — missing (NaN)", f"{raw_col2.isna().sum():,}")
    stat(f"'{FILE2_COL}' — unique (raw, including NaN)", f"{raw_col2.nunique(dropna=False):,}")

    df2["_acc2"] = normalise_accession(raw_col2)
    df2_valid = df2[df2["_acc2"] != ""].copy()

    stat(f"'{FILE2_COL}' — rows after dropping blank/NaN", f"{len(df2_valid):,}")
    stat(f"'{FILE2_COL}' — unique accession numbers (clean)", f"{df2_valid['_acc2'].nunique():,}")

    unique_acc2: set = set(df2_valid["_acc2"].unique())

    # ── 3. SET ANALYSIS ───────────────────────────────────────────────────────
    section("STEP 3 — Set-level analysis")

    in_both   = unique_acc1 & unique_acc2
    only_f1   = unique_acc1 - unique_acc2   # <── the primary result
    only_f2   = unique_acc2 - unique_acc1

    stat("Unique accession numbers in File 1", f"{len(unique_acc1):,}")
    stat("Unique accession numbers in File 2", f"{len(unique_acc2):,}")
    stat("In BOTH files", f"{len(in_both):,}")
    stat("In File 1 ONLY (missing from File 2)", f"{len(only_f1):,}")
    stat("In File 2 ONLY (not in File 1)", f"{len(only_f2):,}")

    coverage_pct = (len(in_both) / len(unique_acc1) * 100) if unique_acc1 else 0.0
    stat("Coverage: File 1 accessions found in File 2", f"{coverage_pct:.1f}%")
    missing_pct  = (len(only_f1) / len(unique_acc1) * 100) if unique_acc1 else 0.0
    stat("Missing: File 1 accessions absent from File 2", f"{missing_pct:.1f}%")

    # ── 4. BUILD RESULT DATAFRAME ─────────────────────────────────────────────
    section("STEP 4 — Building result table")

    # Keep all File 1 rows whose normalised accession is in only_f1
    result_df = df1_valid[df1_valid["_acc1"].isin(only_f1)].copy()
    result_df = result_df.drop(columns=["_acc1"])

    stat("Rows from File 1 that are missing in File 2", f"{len(result_df):,}")
    stat("Unique accession numbers in result", f"{result_df[FILE1_COL].nunique():,}")

    # Show a compact sample
    if not result_df.empty:
        print("\n  Sample (first 10 rows):")
        sample_cols = [FILE1_COL] + [c for c in result_df.columns if c != FILE1_COL][:4]
        print(
            textwrap.indent(
                result_df[sample_cols].head(10).to_string(index=False), "    "
            )
        )

    # ── 5. SAVE OUTPUT ────────────────────────────────────────────────────────
    section("STEP 5 — Saving output")

    result_df.to_csv(OUTPUT_CSV, index=False)
    stat("Output file", str(OUTPUT_CSV.resolve()))
    stat("Rows written", f"{len(result_df):,}")

    # Also write a lean list (accession numbers only)
    acc_only_csv = OUTPUT_CSV.with_name(OUTPUT_CSV.stem + "_accession_list_only.csv")
    pd.DataFrame(sorted(only_f1), columns=["AccessionNumber_missing"]).to_csv(
        acc_only_csv, index=False
    )
    stat("Accession-list-only file", str(acc_only_csv.resolve()))

    # ── 6. FINAL SUMMARY ──────────────────────────────────────────────────────
    section("SUMMARY")
    print(
        f"\n  Out of {len(unique_acc1):,} unique accession numbers in File 1,\n"
        f"  {len(in_both):,} ({coverage_pct:.1f}%) are already present in File 2.\n"
        f"  {len(only_f1):,} ({missing_pct:.1f}%) are MISSING from File 2.\n"
        f"\n  Full result saved to : {OUTPUT_CSV.resolve()}\n"
        f"  Accession list only  : {acc_only_csv.resolve()}\n"
    )
    print(SEP)


if __name__ == "__main__":
    main()