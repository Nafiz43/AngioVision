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
INPUT_DIR = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_23/DICOM_Sequence_Processed"
DEFAULT_BASE_DIR = Path(
    os.getenv(
        "SEQ_BASE_DIR",
        INPUT_DIR,
    )
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


def is_eligible(sequence_dir: Path, min_frames: int) -> Tuple[bool, str, Dict[str, str]]:
    metadata_csv = sequence_dir / METADATA_FILENAME
    frames_dir = sequence_dir / FRAMES_DIRNAME

    details: Dict[str, str] = {
        "RadiationSetting": "",
        "SeriesDescription": "",
        "PositionerMotion": "",
        "FrameCount": "",
    }

    if not metadata_csv.exists():
        return False, "no_metadata", details

    if not frames_dir.exists():
        return False, "no_frames_dir", details

    try:
        metadata = load_metadata_kv(metadata_csv)
    except Exception:
        return False, "metadata_error", details

    rs = normalize_value(metadata.get("RadiationSetting")).upper()
    sd = normalize_value(metadata.get("SeriesDescription")).upper()
    pm = normalize_value(metadata.get("PositionerMotion")).upper()

    details["RadiationSetting"] = rs
    details["SeriesDescription"] = sd
    details["PositionerMotion"] = pm

    if rs != REQUIRED_RADIATION_SETTING:
        return False, "bad_radiation", details

    if not any(k in sd for k in SERIES_DESCRIPTION_KEYWORDS):
        return False, "bad_series", details

    if pm != REQUIRED_POSITIONER_MOTION:
        return False, "bad_motion", details

    try:
        n_frames = count_valid_frames(frames_dir)
        details["FrameCount"] = str(n_frames)
    except Exception:
        return False, "frame_count_error", details

    if n_frames <= min_frames:
        return False, "too_few_frames", details

    return True, "ok", details


def gather_sequence_dirs(base_dir: Path) -> Tuple[List[Path], int]:
    accession_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    sequence_dirs: List[Path] = []

    for acc in accession_dirs:
        for seq in acc.iterdir():
            if seq.is_dir():
                sequence_dirs.append(seq)

    return sequence_dirs, len(accession_dirs)


def write_cleaning_summary_md(
    summary_path: Path,
    base_dir: Path,
    args,
    total_accessions: int,
    total_sequences: int,
    kept: int,
    removed: int,
    stats: Dict[str, int],
) -> None:
    metadata_total = (
        stats["no_metadata"]
        + stats["metadata_error"]
        + stats["bad_radiation"]
        + stats["bad_series"]
        + stats["bad_motion"]
    )

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Cleaning Summary\n\n")

        f.write("## Configuration\n")
        f.write(f"- Base dir: `{base_dir}`\n")
        f.write(f"- Apply deletion: `{args.apply}`\n")
        f.write(f"- Min frames threshold: `{args.min_frames}`\n")
        f.write(f"- Required RadiationSetting: `{REQUIRED_RADIATION_SETTING}`\n")
        f.write(f"- Required PositionerMotion: `{REQUIRED_POSITIONER_MOTION}`\n")
        f.write(
            f"- Required SeriesDescription keywords: `{', '.join(SERIES_DESCRIPTION_KEYWORDS)}`\n\n"
        )

        f.write("## Interpretation of removal reasons\n")
        f.write(
            "- The script checks conditions in order: `RadiationSetting` → "
            "`SeriesDescription` → `PositionerMotion` → `FrameCount`.\n"
        )
        f.write(
            "- Each removed sequence is counted using the **first failing condition only**.\n"
        )
        f.write(
            "- Therefore, a sequence counted under `bad_radiation` may also have failed later checks, "
            "but those later checks are not evaluated once the first failure is found.\n\n"
        )

        f.write("## Dataset Stats\n")
        f.write(f"- Accessions: **{total_accessions}**\n")
        f.write(f"- Sequences: **{total_sequences}**\n")
        f.write(f"- Kept: **{kept}**\n")
        f.write(f"- Removed: **{removed}**\n\n")

        f.write("## Removal Breakdown\n")
        f.write(f"- Metadata total: **{metadata_total}**\n")
        f.write(f"  - no metadata: **{stats['no_metadata']}**\n")
        f.write(f"  - metadata error: **{stats['metadata_error']}**\n")
        f.write(f"  - bad radiation: **{stats['bad_radiation']}**\n")
        f.write(f"  - bad series: **{stats['bad_series']}**\n")
        f.write(f"  - bad motion: **{stats['bad_motion']}**\n")
        f.write(f"- Frames missing: **{stats['no_frames_dir']}**\n")
        f.write(f"- Frame count error: **{stats['frame_count_error']}**\n")
        f.write(f"- Too few frames (<= {args.min_frames}): **{stats['too_few_frames']}**\n")
        f.write(f"- Other: **{stats['other']}**\n\n")

        f.write("## Notes\n")
        if not args.apply:
            f.write("- Dry run only. No directories were deleted.\n")
        else:
            f.write("- Deletion was applied.\n")

        f.write(
            "- Use `--verbose` to print one line per removed sequence including the reason and the metadata values.\n"
        )


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        raise SystemExit(f"Invalid base dir: {base_dir}")

    sequence_dirs, total_accessions = gather_sequence_dirs(base_dir)

    kept = 0
    removed = 0

    stats = {
        "no_metadata": 0,
        "metadata_error": 0,
        "bad_radiation": 0,
        "bad_series": 0,
        "bad_motion": 0,
        "no_frames_dir": 0,
        "frame_count_error": 0,
        "too_few_frames": 0,
        "other": 0,
    }

    for seq_dir in tqdm(sequence_dirs, desc="Filtering", unit="seq"):
        ok, reason, details = is_eligible(seq_dir, args.min_frames)

        if ok:
            kept += 1
        else:
            removed += 1

            if reason in stats:
                stats[reason] += 1
            else:
                stats["other"] += 1

            if args.verbose:
                tqdm.write(
                    "[REMOVE] "
                    f"{seq_dir} -> {reason} | "
                    f"RadiationSetting={details.get('RadiationSetting', '')!r}, "
                    f"SeriesDescription={details.get('SeriesDescription', '')!r}, "
                    f"PositionerMotion={details.get('PositionerMotion', '')!r}, "
                    f"FrameCount={details.get('FrameCount', '')!r}"
                )

            if args.apply:
                try:
                    shutil.rmtree(seq_dir)
                except Exception as e:
                    stats["other"] += 1
                    if args.verbose:
                        tqdm.write(f"[DELETE-ERROR] {seq_dir} -> {e}")

    metadata_total = (
        stats["no_metadata"]
        + stats["metadata_error"]
        + stats["bad_radiation"]
        + stats["bad_series"]
        + stats["bad_motion"]
    )

    print("\n===== SUMMARY =====")
    print(f"Base dir                : {base_dir}")
    print(f"Apply deletion          : {args.apply}")
    print(f"Min frames              : {args.min_frames}")
    print(f"Required Radiation      : {REQUIRED_RADIATION_SETTING}")
    print(f"Required Motion         : {REQUIRED_POSITIONER_MOTION}")
    print(f"Required Series contains: {SERIES_DESCRIPTION_KEYWORDS}")
    print(f"Accessions              : {total_accessions}")
    print(f"Sequences               : {len(sequence_dirs)}")
    print(f"Kept                    : {kept}")
    print(f"Removed                 : {removed}")

    print("\nRemoval breakdown:")
    print(f"  - metadata total      : {metadata_total}")
    print(f"      - no metadata     : {stats['no_metadata']}")
    print(f"      - metadata error  : {stats['metadata_error']}")
    print(f"      - bad radiation   : {stats['bad_radiation']}")
    print(f"      - bad series      : {stats['bad_series']}")
    print(f"      - bad motion      : {stats['bad_motion']}")
    print(f"  - frames missing      : {stats['no_frames_dir']}")
    print(f"  - frame count error   : {stats['frame_count_error']}")
    print(f"  - <= frames           : {stats['too_few_frames']}")
    print(f"  - other               : {stats['other']}")

    if not args.apply:
        print("\n(DRY RUN — nothing deleted)")
    else:
        print("\nDeletion applied.")

    summary_path = base_dir / "cleaning_summary.md"
    write_cleaning_summary_md(
        summary_path=summary_path,
        base_dir=base_dir,
        args=args,
        total_accessions=total_accessions,
        total_sequences=len(sequence_dirs),
        kept=kept,
        removed=removed,
        stats=stats,
    )
    print(f"Summary written to      : {summary_path}")


if __name__ == "__main__":
    main()