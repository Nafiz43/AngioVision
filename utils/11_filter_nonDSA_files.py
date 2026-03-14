#!/usr/bin/env python3
"""
filter_sequence_dirs_by_metadata_nested.py

Directory structure assumed:
  BASE_DIR/
    <accession_dir>/
      <sequence_dir>/
        metadata.csv
        frames/
        mosaic.png
        ...

Eligibility criteria (based on metadata.csv inside each sequence_dir):
  RadiationSetting == "GR"
  AND
  SeriesDescription contains "DSA" or "CO 2"
  AND
  PositionerMotion == "STATIC"

Behavior:
- Keeps only eligible sequence directories
- Removes ineligible sequence directories when --apply is used
- Accession directories are not removed automatically

metadata.csv format expected:
  Information,Value
  RadiationSetting,GR
  SeriesDescription,DSA
  PositionerMotion,STATIC
  ...

Usage:
  Dry run:
    python3 filter_sequence_dirs_by_metadata_nested.py \
      --base_dir /data/Deep_Angiography/DICOM_Sequence_Processed

  Actually delete ineligible sequence dirs:
    python3 filter_sequence_dirs_by_metadata_nested.py \
      --base_dir /data/Deep_Angiography/DICOM_Sequence_Processed \
      --apply
"""

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter nested sequence directories by metadata.csv criteria."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/data/Deep_Angiography/DICOM_Sequence_Processed",
        help="Base directory containing accession dirs, each with sequence dirs."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete non-matching sequence directories."
    )
    return parser.parse_args()


def normalize_value(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip()


def load_metadata_kv(metadata_csv: Path) -> Dict[str, str]:
    """
    Reads metadata.csv in key-value format, e.g.:

      Information,Value
      RadiationSetting,GR
      SeriesDescription,DSA
      PositionerMotion,STATIC
    """
    data: Dict[str, str] = {}

    with metadata_csv.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return data

    start_idx = 0
    if len(rows[0]) >= 2:
        c1 = normalize_value(rows[0][0]).lower()
        c2 = normalize_value(rows[0][1]).lower()
        if (c1, c2) in {
            ("information", "value"),
            ("key", "value"),
            ("field", "value"),
            ("attribute", "value"),
        }:
            start_idx = 1

    for row in rows[start_idx:]:
        if len(row) < 2:
            continue
        key = normalize_value(row[0])
        value = normalize_value(row[1])
        if key:
            data[key] = value

    return data


def is_eligible_sequence_dir(sequence_dir: Path) -> Tuple[bool, str]:
    metadata_csv = sequence_dir / "metadata.csv"

    if not metadata_csv.exists():
        return False, "metadata.csv missing"

    try:
        metadata = load_metadata_kv(metadata_csv)
    except Exception as e:
        return False, f"failed to read metadata.csv: {e}"

    radiation_setting = normalize_value(metadata.get("RadiationSetting")).upper()
    series_description = normalize_value(metadata.get("SeriesDescription")).upper()
    positioner_motion = normalize_value(metadata.get("PositionerMotion")).upper()

    if radiation_setting != "GR":
        return False, f"RadiationSetting={radiation_setting!r}"

    if not ("DSA" in series_description or "CO 2" in series_description):
        return False, f"SeriesDescription={series_description!r}"

    if positioner_motion != "STATIC":
        return False, f"PositionerMotion={positioner_motion!r}"

    return True, "eligible"


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)

    if not base_dir.exists() or not base_dir.is_dir():
        raise SystemExit(f"Invalid base directory: {base_dir}")

    accession_dirs = [p for p in sorted(base_dir.iterdir()) if p.is_dir()]

    total_accessions = 0
    total_sequences = 0
    kept_sequences = 0
    removed_sequences = 0

    for accession_dir in accession_dirs:
        total_accessions += 1
        sequence_dirs = [p for p in sorted(accession_dir.iterdir()) if p.is_dir()]

        for sequence_dir in sequence_dirs:
            total_sequences += 1
            ok, reason = is_eligible_sequence_dir(sequence_dir)

            rel_path = sequence_dir.relative_to(base_dir)

            if ok:
                kept_sequences += 1
                print(f"[KEEP]   {rel_path}  -> {reason}")
            else:
                removed_sequences += 1
                print(f"[REMOVE] {rel_path}  -> {reason}")
                if args.apply:
                    shutil.rmtree(sequence_dir)

    print("\n===== SUMMARY =====")
    print(f"Accession dirs scanned : {total_accessions}")
    print(f"Sequence dirs scanned  : {total_sequences}")
    print(f"Kept sequence dirs     : {kept_sequences}")
    print(f"Removed sequence dirs  : {removed_sequences}")

    if args.apply:
        print("\nDeletion applied.")
    else:
        print("\nDry-run only. No sequence directories were deleted.")
        print("Use --apply to actually remove non-eligible sequence directories.")


if __name__ == "__main__":
    main()