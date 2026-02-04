#!/usr/bin/env python3
"""
merge_csvs.py

Merge all CSV files under a directory into a single CSV.
Assumes all CSVs share the exact same header.
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def merge_csvs(input_dir: Path, output_csv: Path):
    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found under {input_dir}")

    dfs = []
    for csv in tqdm(csv_files, desc="Reading CSVs"):
        try:
            df = pd.read_csv(csv)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {csv}: {e}")

    if not dfs:
        raise RuntimeError("All CSV files were empty or unreadable")

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_csv, index=False)

    print(f"\nMerged {len(dfs)} files → {output_csv}")
    print(f"Total rows: {len(merged)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed_Output-CSVs/"))
    parser.add_argument("--output_csv", type=Path, default=Path("/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed_Output-CSVs/merged_output.csv"))
    args = parser.parse_args()

    merge_csvs(args.input_dir, args.output_csv)
