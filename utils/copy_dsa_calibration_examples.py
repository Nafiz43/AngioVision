#!/usr/bin/env python3
"""
Copy DSA calibration example sequences to a single directory
=============================================================

Standalone helper — NOT part of any pipeline. Gathers the known-DSA
sequences used to calibrate the frame-based DSA mask detector
(utils/05_dsa_identification_based_on_frame_v2.py and
utils/visual-data-preparation step 06) into one self-contained directory:

    /data/Deep_Angiography/frame_identification_algo_calibration_examples/

Each calibration root is copied recursively. The destination keeps each
root's path relative to /data/Deep_Angiography with its top-level dir
(DICOM_Sequence_Processed / Deep_Angio_DB_v02) stripped, so e.g.

    /data/Deep_Angiography/DICOM_Sequence_Processed/0AwEV1kXtf
        -> <DEST>/0AwEV1kXtf
    /data/Deep_Angiography/DICOM_Sequence_Processed/1MPUcLN3XP/2.16.840...942
        -> <DEST>/1MPUcLN3XP/2.16.840...942          (accession kept)
    /data/Deep_Angiography/Deep_Angio_DB_v02/example_dsa_cases
        -> <DEST>/example_dsa_cases

Roots already present at the destination are skipped, so the script is
safe to re-run (resume after interruption). Missing source roots are
warned about and skipped, never fatal.

How to run (lab server only — paths are hardcoded)
--------------------------------------------------
    python utils/copy_dsa_calibration_examples.py --dry-run   # preview only
    python utils/copy_dsa_calibration_examples.py             # actually copy

No dependencies beyond the standard library.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DATA_ROOT = Path("/data/Deep_Angiography")

# Mirrors CALIBRATION_ROOTS in utils/05_dsa_identification_based_on_frame_v2.py.
# The visual-data-preparation pipeline's dsa_calibration_roots default now
# points at DEST_ROOT (the directory this script populates).
CALIBRATION_ROOTS = [
    DATA_ROOT / "DICOM_Sequence_Processed/0AwEV1kXtf",
    DATA_ROOT / "DICOM_Sequence_Processed/0BH55V6rIB",
    DATA_ROOT / "DICOM_Sequence_Processed/2C9rBTcczL",
    DATA_ROOT / "DICOM_Sequence_Processed/5NUyFXc5Ai",
    DATA_ROOT / "DICOM_Sequence_Processed/5o3Mxk1lx7",
    DATA_ROOT / "DICOM_Sequence_Processed/6kpsDZBHAH",
    DATA_ROOT / "DICOM_Sequence_Processed/1cZA9m5qti",
    DATA_ROOT / "DICOM_Sequence_Processed/P2ykm7rSF8",
    DATA_ROOT / "DICOM_Sequence_Processed/1MPUcLN3XP/2.16.840.1.113883.3.16.245346042915223951797304877264329724942",
    DATA_ROOT / "Deep_Angio_DB_v02/example_dsa_cases",
]

DEST_ROOT = DATA_ROOT / "frame_identification_algo_calibration_examples"


def dest_for(root: Path) -> Path:
    """
    Destination dir for one calibration root: its path relative to
    DATA_ROOT with the first component (DICOM_Sequence_Processed /
    Deep_Angio_DB_v02) stripped. Falls back to the basename for any
    root that doesn't live under DATA_ROOT.
    """
    try:
        rel = root.relative_to(DATA_ROOT)
        parts = rel.parts[1:] if len(rel.parts) > 1 else rel.parts
        return DEST_ROOT.joinpath(*parts)
    except ValueError:
        return DEST_ROOT / root.name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy DSA calibration example sequences into "
                    f"{DEST_ROOT}",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be copied, copy nothing")
    args = parser.parse_args()

    copied = skipped_existing = missing = failed = 0

    print(f"Destination : {DEST_ROOT}")
    print(f"Roots       : {len(CALIBRATION_ROOTS)}"
          f"{'  (DRY RUN — nothing will be copied)' if args.dry_run else ''}\n")

    if not args.dry_run:
        DEST_ROOT.mkdir(parents=True, exist_ok=True)

    for root in CALIBRATION_ROOTS:
        dest = dest_for(root)
        if not root.is_dir():
            print(f"[MISSING] {root}")
            missing += 1
            continue
        if dest.exists():
            print(f"[SKIP]    {dest}  (already present)")
            skipped_existing += 1
            continue
        n_files = sum(1 for p in root.rglob("*") if p.is_file())
        print(f"[COPY]    {root}")
        print(f"      ->  {dest}  ({n_files:,} files)")
        if args.dry_run:
            copied += 1
            continue
        try:
            shutil.copytree(root, dest)
            copied += 1
        except Exception as e:
            # Remove a half-written dest so a re-run doesn't [SKIP] it.
            shutil.rmtree(dest, ignore_errors=True)
            print(f"[ERROR]   {root}: {e}")
            failed += 1

    print(f"\n{'Would copy' if args.dry_run else 'Copied'}: {copied}   "
          f"skipped (already present): {skipped_existing}   "
          f"missing sources: {missing}   failed: {failed}")
    if not args.dry_run and copied + skipped_existing:
        print(f"Calibration examples dir: {DEST_ROOT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
