#!/usr/bin/env python3
"""
Copy mosaic.png files for DICOMs where number_of_frames < 10 or > 60.
Mosaic is located one directory up from the frames_dir_path.
Output files are named by their frame count.
"""

import csv
import shutil
import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = Path("/data/Deep_Angiography/DICOM-metadata-stats/frame_statistics.csv")
OUTPUT_DIR  = Path("/data/Deep_Angiography/DICOM_Sequence_Processed_Outliers")
LOW_THRESH  = 10
HIGH_THRESH = 60
# ──────────────────────────────────────────────────────────────────────────────

def is_outlier(n: int) -> bool:
    return n < LOW_THRESH or n > HIGH_THRESH


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copied   = 0
    skipped  = 0
    missing  = 0
    name_collisions = {}   # frame_count -> list of source paths (for logging)

    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n_frames = int(row["number_of_frames"])
            except (KeyError, ValueError) as e:
                print(f"[WARN] Could not parse number_of_frames in row: {row} — {e}")
                skipped += 1
                continue

            if not is_outlier(n_frames):
                skipped += 1
                continue

            frames_dir  = Path(row["frames_dir_path"].strip())
            dicom_dir   = frames_dir.parent          # one level up
            mosaic_src  = dicom_dir / "mosaic.png"

            if not mosaic_src.exists():
                print(f"[MISSING] {mosaic_src}")
                missing += 1
                continue

            # Destination: named by frame count; handle collisions with a suffix
            dest_name = f"{n_frames}.png"
            dest_path = OUTPUT_DIR / dest_name

            if dest_path.exists():
                # Track collision: append outer_dir_name to disambiguate
                outer = row.get("outer_dir_name", "unknown")
                inner = row.get("inner_dir_name", "unknown")
                suffix = f"{outer}_{inner[:8]}"
                dest_name = f"{n_frames}_{suffix}.png"
                dest_path = OUTPUT_DIR / dest_name
                print(f"[COLLISION] frame count {n_frames} already used — saving as {dest_name}")

            shutil.copy2(mosaic_src, dest_path)
            print(f"[COPIED]   {mosaic_src}  →  {dest_path}")

            # Copy metadata.csv from the same dicom_dir, named to match the PNG
            metadata_src  = dicom_dir / "metadata.csv"
            metadata_dest = OUTPUT_DIR / dest_path.with_suffix(".csv").name
            if metadata_src.exists():
                shutil.copy2(metadata_src, metadata_dest)
                print(f"[COPIED]   {metadata_src}  →  {metadata_dest}")
            else:
                print(f"[MISSING]  metadata.csv not found at {metadata_src}")

            copied += 1

    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Copied  : {copied}")
    print(f"  Skipped (in-range): {skipped}")
    print(f"  Missing mosaic.png: {missing}")
    print(f"  Output dir        : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()