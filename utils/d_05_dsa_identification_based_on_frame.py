#!/usr/bin/env python3
"""
DSA Mask Frame Detector  —  Auto-calibrated
============================================

Phase 1 — Calibration
----------------------
Recursively scans CALIBRATION_ROOT (known DSA sequences only) and computes
pixel statistics for every frame:
    - mean intensity
    - std deviation
    - bright pixel fraction  (pixels > 220 / total pixels)

For each sequence the frame with the LOWEST std is the mask-frame candidate.
Thresholds are derived from the distribution of all candidates:

    MASK_MAX_STD         = 95th percentile of candidate std  * margin
    MASK_MAX_MEAN        = 97th percentile of candidate mean * margin
    MASK_MIN_MEAN        = 3rd  percentile of candidate mean / margin
    MASK_MAX_BRIGHT_FRAC = 97th percentile of candidate bright_frac * margin

Phase 2 — Detection + Copy
---------------------------
Uses the calibrated thresholds to scan SOURCE_ROOT and copy every sequence
that contains at least one mask frame to DEST_ROOT.

Reports written to REPORT_DIR:
    calibration_stats.csv           per-frame pixel stats from calibration set
    calibration_summary.txt         derived thresholds with explanation
    mask_frame_detection_report.csv per-sequence verdict + copy status
"""

import csv
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# =========================================================
# Path config
# =========================================================
CALIBRATION_ROOT = Path(
    "/data/Deep_Angiography/DICOM_Sequence_Processed/0AwEV1kXtf"
)
SOURCE_ROOT = Path(
    "/data/Deep_Angiography/DICOM_Sequence_Processed/00_sequence_to_check"
)
DEST_ROOT = Path(
    "/data/Deep_Angiography/DICOM_Sequence_Processed/00_potential_dsas"
)
REPORT_DIR = Path(
    "/data/Deep_Angiography/DICOM-metadata-stats"
)

CALIB_STATS_CSV   = REPORT_DIR / "calibration_stats.csv"
CALIB_SUMMARY_TXT = REPORT_DIR / "calibration_summary.txt"
DETECTION_CSV     = REPORT_DIR / "mask_frame_detection_report.csv"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
WORKERS          = max(1, (os.cpu_count() or 8) - 1)

# Safety margin: 15% headroom on top of raw percentile so borderline
# real mask frames are not missed.
MARGIN = 1.15


# =========================================================
# Pixel stats — single frame
# =========================================================
def frame_stats(img_path_str: str) -> dict | None:
    """
    Compute pixel statistics for one grayscale frame.
    Accepts a string path (picklable for ProcessPoolExecutor).
    Returns None if the file cannot be read.
    """
    try:
        arr = np.array(
            Image.open(img_path_str).convert("L"), dtype=np.float32
        )
        return {
            "path":        img_path_str,
            "mean":        float(np.mean(arr)),
            "std":         float(np.std(arr)),
            "bright_frac": float(np.sum(arr > 220)) / arr.size,
        }
    except (OSError, ValueError) as exc:
        print(f"[WARN] Could not compute frame stats for {img_path_str}: {exc}", file=__import__('sys').stderr)
        return None


# =========================================================
# Phase 1: calibration
# =========================================================
def collect_frame_paths(root: Path) -> list:
    """Recursively find all image frames under root/**/frames/."""
    paths = []
    for frames_dir in root.rglob("frames"):
        if frames_dir.is_dir():
            for f in frames_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(str(f))
    return paths


def calibrate() -> dict:
    """
    Phase 1: scan all frames in CALIBRATION_ROOT, compute per-frame stats,
    identify the mask-frame candidate per sequence (lowest-std frame),
    and derive detection thresholds from their distribution.
    Returns a dict with keys:
        MASK_MAX_STD, MASK_MAX_MEAN, MASK_MIN_MEAN, MASK_MAX_BRIGHT_FRAC
    """
    print(f"\n{'='*60}")
    print("PHASE 1 — Calibration")
    print(f"  Calibration root : {CALIBRATION_ROOT}")
    print(f"{'='*60}\n")

    frame_paths = collect_frame_paths(CALIBRATION_ROOT)
    if not frame_paths:
        raise RuntimeError(
            f"No image frames found under {CALIBRATION_ROOT}. "
            "Check the path and directory structure."
        )
    print(f"Found {len(frame_paths):,} frames in calibration set.\n")

    # ── Compute stats for every frame in parallel ─────────────────────────
    all_stats = []
    with tqdm(total=len(frame_paths), desc="Computing frame stats",
              unit="frame", dynamic_ncols=True) as pbar:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(frame_stats, p): p for p in frame_paths}
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    all_stats.append(res)
                pbar.update(1)

    print(f"Stats computed for {len(all_stats):,} frames.")

    # ── Save full per-frame CSV ───────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with CALIB_STATS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "mean", "std", "bright_frac"])
        w.writeheader()
        w.writerows(all_stats)
    print(f"Per-frame calibration stats  -> {CALIB_STATS_CSV}")

    # ── Group frames by sequence (parent frames/ dir) ─────────────────────
    groups: dict = {}
    for s in all_stats:
        key = str(Path(s["path"]).parent)  # the frames/ dir
        groups.setdefault(key, []).append(s)

    # ── Pick mask candidate per sequence = lowest-std frame ───────────────
    mask_candidates = []
    for fpaths_stats in groups.values():
        candidate = min(fpaths_stats, key=lambda s: s["std"])
        mask_candidates.append(candidate)

    print(f"Mask-frame candidates identified : {len(mask_candidates):,} "
          f"(one per sequence)\n")

    # ── Derive thresholds ─────────────────────────────────────────────────
    stds         = np.array([c["std"]         for c in mask_candidates])
    means        = np.array([c["mean"]        for c in mask_candidates])
    bright_fracs = np.array([c["bright_frac"] for c in mask_candidates])

    raw_max_std         = float(np.percentile(stds,         95))
    raw_max_mean        = float(np.percentile(means,        97))
    raw_min_mean        = float(np.percentile(means,         3))
    raw_max_bright_frac = float(np.percentile(bright_fracs, 97))

    MASK_MAX_STD         = round(raw_max_std         * MARGIN, 2)
    MASK_MAX_MEAN        = round(min(255.0, raw_max_mean        * MARGIN), 2)
    MASK_MIN_MEAN        = round(raw_min_mean        / MARGIN,  2)
    MASK_MAX_BRIGHT_FRAC = round(min(1.0,  raw_max_bright_frac * MARGIN), 4)

    thresholds = {
        "MASK_MAX_STD":         MASK_MAX_STD,
        "MASK_MAX_MEAN":        MASK_MAX_MEAN,
        "MASK_MIN_MEAN":        MASK_MIN_MEAN,
        "MASK_MAX_BRIGHT_FRAC": MASK_MAX_BRIGHT_FRAC,
    }

    # ── Print + save summary ──────────────────────────────────────────────
    lines = [
        "DSA Mask Frame Detector — Calibration Summary",
        "=" * 55,
        f"Calibration root     : {CALIBRATION_ROOT}",
        f"Total frames scanned : {len(all_stats):,}",
        f"Sequences (groups)   : {len(groups):,}",
        f"Mask candidates      : {len(mask_candidates):,}",
        "",
        "Mask-candidate pixel statistics",
        "-" * 40,
        f"  std        min={stds.min():.2f}  median={np.median(stds):.2f}"
        f"  p95={raw_max_std:.2f}  max={stds.max():.2f}",
        f"  mean       min={means.min():.2f}  median={np.median(means):.2f}"
        f"  p97={raw_max_mean:.2f}  max={means.max():.2f}",
        f"  bright_frac  p97={raw_max_bright_frac:.4f}"
        f"  max={bright_fracs.max():.4f}",
        "",
        f"Margin applied : {MARGIN}x  (15% headroom)",
        "",
        "Derived thresholds",
        "-" * 40,
        f"  MASK_MAX_STD         = {MASK_MAX_STD:<10}  (p95 * margin)",
        f"  MASK_MAX_MEAN        = {MASK_MAX_MEAN:<10}  (p97 * margin, cap 255)",
        f"  MASK_MIN_MEAN        = {MASK_MIN_MEAN:<10}  (p3  / margin)",
        f"  MASK_MAX_BRIGHT_FRAC = {MASK_MAX_BRIGHT_FRAC:<10}  (p97 * margin, cap 1.0)",
    ]
    summary = "\n".join(lines)
    print("\n" + summary + "\n")

    with CALIB_SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"Calibration summary          -> {CALIB_SUMMARY_TXT}\n")

    return thresholds


# =========================================================
# Phase 2: detection helpers
# =========================================================
def is_mask_frame(img_path_str: str, thresholds: dict) -> bool:
    """
    Returns True if the frame qualifies as a DSA mask frame
    under the calibrated thresholds.
    """
    try:
        arr = np.array(
            Image.open(img_path_str).convert("L"), dtype=np.float32
        )
        mean_val    = float(np.mean(arr))
        std_val     = float(np.std(arr))
        bright_frac = float(np.sum(arr > 220)) / arr.size

        return (
            thresholds["MASK_MIN_MEAN"] <= mean_val <= thresholds["MASK_MAX_MEAN"]
            and std_val     <= thresholds["MASK_MAX_STD"]
            and bright_frac <= thresholds["MASK_MAX_BRIGHT_FRAC"]
        )
    except (OSError, ValueError):
        return False


def analyse_sequence(args: tuple) -> dict:
    """
    Worker: analyse one SOPInstanceUID dir.
    args = (sop_dir_str, thresholds)
    """
    sop_dir_str, thresholds = args
    sop_dir    = Path(sop_dir_str)
    frames_dir = sop_dir / "frames"

    result = {
        "sequence_dir":     sop_dir_str,
        "total_frames":     0,
        "mask_frame_count": 0,
        "mask_frame_names": "",
        "verdict":          "no_mask_detected",
        "copied":           False,
    }

    if not frames_dir.is_dir():
        result["verdict"] = "skipped_no_frames_dir"
        return result

    frame_files = sorted(
        str(f) for f in frames_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    result["total_frames"] = len(frame_files)

    if not frame_files:
        result["verdict"] = "skipped_too_few"
        return result

    mask_frames = [p for p in frame_files if is_mask_frame(p, thresholds)]
    result["mask_frame_count"] = len(mask_frames)
    result["mask_frame_names"] = ",".join(Path(p).name for p in mask_frames)

    if mask_frames:
        result["verdict"] = "potential_dsa"

    return result


def collect_sop_dirs(source_root: Path) -> list:
    sop_dirs = []
    for acc_dir in source_root.iterdir():
        if not acc_dir.is_dir():
            continue
        for sop_dir in acc_dir.iterdir():
            if sop_dir.is_dir() and (sop_dir / "frames").is_dir():
                sop_dirs.append(sop_dir)
    return sop_dirs


def copy_sequence(sop_dir: Path) -> bool:
    try:
        dest_dir = DEST_ROOT / sop_dir.parent.name / sop_dir.name
        if dest_dir.exists():
            return True
        shutil.copytree(sop_dir, dest_dir)
        return True
    except Exception as e:
        print(f"[COPY ERROR] {sop_dir}: {e}")
        return False


# =========================================================
# Phase 2: detection + copy
# =========================================================
def detect_and_copy(thresholds: dict) -> None:
    print(f"\n{'='*60}")
    print("PHASE 2 — Detection + Copy")
    print(f"  Source  : {SOURCE_ROOT}")
    print(f"  Dest    : {DEST_ROOT}")
    print("  Thresholds:")
    for k, v in thresholds.items():
        print(f"    {k:30s} = {v}")
    print(f"{'='*60}\n")

    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    print("Collecting sequence directories...")
    sop_dirs = collect_sop_dirs(SOURCE_ROOT)
    if not sop_dirs:
        print("No sequence directories found under SOURCE_ROOT. Exiting.")
        return
    print(f"Found {len(sop_dirs):,} sequences to analyse.\n")

    # Pass thresholds with each dir as a tuple (safe for pickling)
    work_items = [(str(d), thresholds) for d in sop_dirs]

    results     = []
    n_potential = 0
    n_no_mask   = 0
    n_skipped   = 0

    with tqdm(total=len(sop_dirs), unit="seq", desc="Detecting",
              dynamic_ncols=True) as pbar:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            future_map = {
                ex.submit(analyse_sequence, item): item[0]
                for item in work_items
            }
            for fut in as_completed(future_map):
                sop_dir_str = future_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {
                        "sequence_dir":     sop_dir_str,
                        "total_frames":     0,
                        "mask_frame_count": 0,
                        "mask_frame_names": "",
                        "verdict":          f"worker_error: {e}",
                        "copied":           False,
                    }

                v = res["verdict"]
                if v == "potential_dsa":
                    n_potential += 1
                elif v == "no_mask_detected":
                    n_no_mask += 1
                else:
                    n_skipped += 1

                results.append(res)
                pbar.set_postfix(
                    potential=n_potential,
                    no_mask=n_no_mask,
                    skipped=n_skipped,
                )
                pbar.update(1)

    # ── Copy flagged sequences ────────────────────────────────────────────
    to_copy = [r for r in results if r["verdict"] == "potential_dsa"]
    print(f"\nCopying {len(to_copy):,} potential DSA sequences → {DEST_ROOT} ...")

    with tqdm(total=len(to_copy), unit="seq", desc="Copying",
              dynamic_ncols=True) as pbar:
        for res in to_copy:
            ok = copy_sequence(Path(res["sequence_dir"]))
            res["copied"] = ok
            pbar.update(1)

    # ── Write detection report ────────────────────────────────────────────
    fieldnames = [
        "sequence_dir", "total_frames", "mask_frame_count",
        "mask_frame_names", "verdict", "copied",
    ]
    with DETECTION_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    copied_ok  = sum(1 for r in results if r["copied"])
    copy_fail  = sum(1 for r in results
                     if r["verdict"] == "potential_dsa" and not r["copied"])

    print(f"\n{'='*60}")
    print(f"  Sequences examined         : {len(sop_dirs):,}")
    print(f"  Potential DSA (mask found) : {n_potential:,}")
    print(f"  No mask detected           : {n_no_mask:,}")
    print(f"  Skipped                    : {n_skipped:,}")
    print(f"  Successfully copied        : {copied_ok:,}")
    if copy_fail:
        print(f"  Copy failures              : {copy_fail:,}")
    print(f"  Detection report           : {DETECTION_CSV}")
    print(f"{'='*60}\n")


# =========================================================
# Entry point
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = calibrate()
    detect_and_copy(thresholds)
    print("All done.")


if __name__ == "__main__":
    main()