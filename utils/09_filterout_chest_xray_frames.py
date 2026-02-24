#!/usr/bin/env python3
"""
remove_small_sequences_parallel.py

Removes SOPInstanceUID directories under DICOM_Sequence_Processed
if the number of frame images inside 'frames/' is less than 2.

Parallelized with ProcessPoolExecutor + tqdm progress bar.

Expected structure:

BASE_DIR/
  <AnonAccession>/
    <SOPInstanceUID>/
      frames/
        *.png / *.jpg / ...
      metadata.csv
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ==========================
# CONFIGURE THIS PATH
# ==========================
BASE_DIR = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class Result:
    sop_dir: str
    checked: bool
    removed: bool
    n_frames: Optional[int]
    reason: str  # "removed_lt2", "kept", "missing_frames_dir", "error"


def _count_frames(frames_dir: Path) -> int:
    # Count only valid image files
    n = 0
    for f in frames_dir.iterdir():
        if f.is_file() and f.suffix.lower() in VALID_EXTS:
            n += 1
    return n


def _process_one(sop_dir_str: str, min_frames: int = 2, dry_run: bool = False) -> Result:
    """
    Worker: checks frames count in sop_dir/frames and deletes sop_dir if < min_frames.
    Returns a small Result summary (pickle-friendly).
    """
    try:
        sop_dir = Path(sop_dir_str)
        frames_dir = sop_dir / "frames"
        if not frames_dir.exists():
            return Result(sop_dir=sop_dir_str, checked=False, removed=False, n_frames=None, reason="missing_frames_dir")

        n_frames = _count_frames(frames_dir)

        if n_frames < min_frames:
            if not dry_run:
                shutil.rmtree(sop_dir)
            return Result(sop_dir=sop_dir_str, checked=True, removed=True, n_frames=n_frames, reason="removed_lt2")

        return Result(sop_dir=sop_dir_str, checked=True, removed=False, n_frames=n_frames, reason="kept")

    except Exception:
        return Result(sop_dir=sop_dir_str, checked=False, removed=False, n_frames=None, reason="error")


def _gather_sop_dirs(base_dir: Path) -> list[str]:
    sop_dirs: list[str] = []
    for accession_dir in base_dir.iterdir():
        if not accession_dir.is_dir():
            continue
        for sop_dir in accession_dir.iterdir():
            if sop_dir.is_dir():
                sop_dirs.append(str(sop_dir))
    return sop_dirs


def main(
    base_dir: Path = BASE_DIR,
    min_frames: int = 2,
    max_workers: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    sop_dirs = _gather_sop_dirs(base_dir)
    total = len(sop_dirs)
    if total == 0:
        print(f"No SOPInstanceUID directories found under: {base_dir}")
        return

    if max_workers is None:
        # Good default: leave 1 core free when possible
        cpu = os.cpu_count() or 4
        max_workers = max(1, cpu - 1)

    removed = 0
    kept = 0
    missing_frames_dir = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one, sop, min_frames, dry_run) for sop in sop_dirs]

        for fut in tqdm(as_completed(futures), total=total, desc="Checking/removing sequences", unit="seq"):
            r: Result = fut.result()

            if r.reason == "removed_lt2":
                removed += 1
            elif r.reason == "kept":
                kept += 1
            elif r.reason == "missing_frames_dir":
                missing_frames_dir += 1
            else:  # "error"
                errors += 1

    # Summary only (no per-entry spam)
    print("\n================== SUMMARY ==================")
    print(f"Base dir                 : {base_dir}")
    print(f"Total sequences found    : {total}")
    print(f"Kept (>= {min_frames})           : {kept}")
    print(f"Removed (< {min_frames})         : {removed}{'  (DRY RUN)' if dry_run else ''}")
    print(f"Missing frames/ dir      : {missing_frames_dir}")
    print(f"Errors                   : {errors}")
    print("============================================\n")


if __name__ == "__main__":
    # Note: ProcessPoolExecutor requires this guard (especially on macOS/Windows).
    main()