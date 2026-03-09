#!/usr/bin/env python3
"""
collect_large_sequences_artifacts.py

Reads:
  /data/Deep_Angiography/DICOM-metadata-stats/frame_statistics.csv

Filters rows where:
  number_of_frames > threshold

For each sequence:
  - Goes one directory above frames_dir_path
  - Collects mosaic.png and metadata.csv
  - Copies them into a new directory

Files are renamed to include frame count:
  <number_of_frames>_mosaic.png
  <number_of_frames>_metadata.csv
"""

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_CSV_PATH = "/data/Deep_Angiography/DICOM-metadata-stats/frame_statistics.csv"
DEFAULT_OUTPUT_DIR = "/data/Deep_Angiography/DICOM-metadata-stats/sequences_over_100_frames_artifacts"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect mosaic.png and metadata.csv for sequences with >100 frames."
    )

    parser.add_argument(
        "--csv_path",
        default=DEFAULT_CSV_PATH,
        help="Path to frame_statistics.csv"
    )

    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where sequence folders will be created"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Minimum number_of_frames required"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )

    return parser.parse_args()


def safe_int(v):
    try:
        return int(str(v).strip())
    except Exception:
        return None


def copy_if_needed(src: Path, dst: Path, overwrite: bool):
    if not src.exists():
        return "missing"

    if dst.exists() and not overwrite:
        return "exists"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    selected_rows = 0

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_rows += 1

            num_frames = safe_int(row["number_of_frames"])
            if num_frames is None:
                continue

            if num_frames <= args.threshold:
                continue

            selected_rows += 1

            outer_dir = row["outer_dir_name"].strip()
            inner_dir = row["inner_dir_name"].strip()

            frames_dir = Path(row["frames_dir_path"])
            seq_dir = frames_dir.parent

            metadata_src = seq_dir / "metadata.csv"
            mosaic_src = seq_dir / "mosaic.png"

            seq_output_dir = output_dir / f"{outer_dir}__{inner_dir}"
            seq_output_dir.mkdir(parents=True, exist_ok=True)

            # rename with frame count
            metadata_dst = seq_output_dir / f"{num_frames}_metadata.csv"
            mosaic_dst = seq_output_dir / f"{num_frames}_mosaic.png"

            metadata_status = copy_if_needed(metadata_src, metadata_dst, args.overwrite)
            mosaic_status = copy_if_needed(mosaic_src, mosaic_dst, args.overwrite)

            print(f"[SEQ] {outer_dir} / {inner_dir}")
            print(f"      frames: {num_frames}")
            print(f"      metadata.csv -> {metadata_status}")
            print(f"      mosaic.png  -> {mosaic_status}")

    print("\nFinished.")
    print(f"Total rows scanned: {total_rows}")
    print(f"Sequences with > {args.threshold} frames: {selected_rows}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()