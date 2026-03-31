#!/usr/bin/env python3
"""
filter_sequence_dirs_by_metadata_and_frames.py
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from tqdm import tqdm


# =========================================================
# EASY CONFIG (EDIT HERE OR VIA ENV VARS)
# =========================================================
DEFAULT_BASE_DIR = Path(
    os.getenv("SEQ_BASE_DIR", "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_23/DICOM_Sequence_Processed")
)

# Remove sequences with frame_count <= this
DEFAULT_MIN_FRAMES = int(os.getenv("SEQ_MIN_FRAMES", "2"))

# Set SEQ_APPLY=1 to delete by default
DEFAULT_APPLY = os.getenv("SEQ_APPLY", "0").lower() in {"1", "true", "yes"}

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

REQUIRED_RADIATION_SETTING = "GR"
REQUIRED_POSITIONER_MOTION = "STATIC"
SERIES_DESCRIPTION_KEYWORDS = ("DSA", "CO 2")

METADATA_FILENAME = "metadata.csv"
FRAMES_DIRNAME = "frames"


# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--apply", action="store_true", default=DEFAULT_APPLY)
    parser.add_argument("--min_frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def normalize_value(x: Optional[str]) -> str:
    return "" if x is None else str(x).strip()


def load_metadata_kv(metadata_csv: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}

    with metadata_csv.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return data

    start_idx = 1 if len(rows[0]) >= 2 else 0

    for row in rows[start_idx:]:
        if len(row) < 2:
            continue
        key = normalize_value(row[0])
        val = normalize_value(row[1])
        if key:
            data[key] = val

    return data


def count_valid_frames(frames_dir: Path) -> int:
    return sum(
        1 for f in frames_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    )


def is_eligible(sequence_dir: Path, min_frames: int) -> Tuple[bool, str]:
    metadata_csv = sequence_dir / METADATA_FILENAME
    frames_dir = sequence_dir / FRAMES_DIRNAME

    if not metadata_csv.exists():
        return False, "no_metadata"

    if not frames_dir.exists():
        return False, "no_frames_dir"

    try:
        metadata = load_metadata_kv(metadata_csv)
    except Exception:
        return False, "metadata_error"

    rs = normalize_value(metadata.get("RadiationSetting")).upper()
    sd = normalize_value(metadata.get("SeriesDescription")).upper()
    pm = normalize_value(metadata.get("PositionerMotion")).upper()

    if rs != REQUIRED_RADIATION_SETTING:
        return False, "bad_radiation"

    if not any(k in sd for k in SERIES_DESCRIPTION_KEYWORDS):
        return False, "bad_series"

    if pm != REQUIRED_POSITIONER_MOTION:
        return False, "bad_motion"

    try:
        n_frames = count_valid_frames(frames_dir)
    except Exception:
        return False, "frame_count_error"

    if n_frames <= min_frames:
        return False, "too_few_frames"

    return True, "ok"


def gather_sequence_dirs(base_dir: Path) -> Tuple[List[Path], int]:
    accession_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    sequence_dirs = []

    for acc in accession_dirs:
        for seq in acc.iterdir():
            if seq.is_dir():
                sequence_dirs.append(seq)

    return sequence_dirs, len(accession_dirs)


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        raise SystemExit(f"Invalid base dir: {base_dir}")

    sequence_dirs, total_accessions = gather_sequence_dirs(base_dir)

    kept = 0
    removed = 0

    stats = {
        "metadata": 0,
        "frames_dir": 0,
        "frame_count": 0,
        "other": 0
    }

    for seq_dir in tqdm(sequence_dirs, desc="Filtering", unit="seq"):
        ok, reason = is_eligible(seq_dir, args.min_frames)

        if ok:
            kept += 1
        else:
            removed += 1

            if reason == "too_few_frames":
                stats["frame_count"] += 1
            elif reason == "no_frames_dir":
                stats["frames_dir"] += 1
            elif reason.startswith("bad") or reason == "no_metadata":
                stats["metadata"] += 1
            else:
                stats["other"] += 1

            if args.verbose:
                tqdm.write(f"[REMOVE] {seq_dir} -> {reason}")

            if args.apply:
                try:
                    shutil.rmtree(seq_dir)
                except Exception:
                    stats["other"] += 1

    print("\n===== SUMMARY =====")
    print(f"Base dir            : {base_dir}")
    print(f"Apply deletion      : {args.apply}")
    print(f"Min frames          : {args.min_frames}")
    print(f"Accessions          : {total_accessions}")
    print(f"Sequences           : {len(sequence_dirs)}")
    print(f"Kept                : {kept}")
    print(f"Removed             : {removed}")
    print(f"  - metadata        : {stats['metadata']}")
    print(f"  - frames missing  : {stats['frames_dir']}")
    print(f"  - <= frames       : {stats['frame_count']}")
    print(f"  - other           : {stats['other']}")

    if not args.apply:
        print("\n(DRY RUN — nothing deleted)")
    else:
        print("\nDeletion applied.")


if __name__ == "__main__":
    main()