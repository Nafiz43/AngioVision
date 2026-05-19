#!/usr/bin/env python3
"""
DSA Mask Frame Detector  —  Auto-calibrated (v4)
=================================================

Pipeline overview
-----------------

  Phase 1  — Calibration
      Learns thresholds from known-positive DSA sequences
      (CALIBRATION_ROOTS).  No test data touches this phase.

  Phase 1b — Validation   (on calibration positives + EVAL_NEGATIVE_DIRS)
      Quick sanity-check: does the detector correctly classify the very
      sequences it was calibrated on, plus a few hand-labelled negatives?
      This is NOT the final performance number — it is optimistically biased
      because the positives were used to set the thresholds.

  Phase 1c — Test Evaluation   ← final, unbiased performance
      Runs the detector on a fully held-out labelled test set
      (TEST_POSITIVE_ROOTS  +  TEST_NEGATIVE_DIRS) that was never seen
      during calibration.  TP / TN / FP / FN reported here are the numbers
      to report in papers / presentations.

  Phase 2  — Detection + Copy   (skipped with --calibration_only)
      Scans SOURCE_ROOT and copies potential-DSA sequences to DEST_ROOT.

Feature engineering (v3 → v4 unchanged)
----------------------------------------
  All statistics computed on NON-BLACK pixels only (> BLACK_THRESHOLD).
  Per frame: mean, std, bright_frac, entropy (Shannon, bits).
  Per sequence: mask candidate = frame with lowest active-pixel std.
  Thresholds derived from p95/p97/p3 of mask-candidate distributions
  with a 15 % safety margin.

Reports
-------
  calibration_stats.csv             per-frame active-pixel stats
  calibration_summary.txt           derived thresholds
  validation_report.csv             Phase 1b confusion matrix (biased)
  test_report.csv                   Phase 1c confusion matrix (unbiased) ★
  mask_frame_detection_report.csv   Phase 2 per-sequence verdict + copy

Usage
-----
  python dsa_mask_detector.py                     # full run
  python dsa_mask_detector.py --calibration_only  # phases 1 + 1b + 1c only
"""

import argparse
import csv
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# =========================================================
# ── Path config ───────────────────────────────────────────
# =========================================================

# Sequences used to LEARN the thresholds.
# Never overlap with test positives.
CALIBRATION_ROOTS = [
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/0AwEV1kXtf"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/0BH55V6rIB"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/2C9rBTcczL"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/5NUyFXc5Ai"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/5o3Mxk1lx7"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/6kpsDZBHAH"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/1cZA9m5qti"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/P2ykm7rSF8"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/1MPUcLN3XP/2.16.840.1.113883.3.16.245346042915223951797304877264329724942")
]

# ── Phase 1b: validation negatives (calibration-set sanity check) ─────────
# These are NOT test data — used only for the optimistic in-sample check.
EVAL_NEGATIVE_DIRS = [
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/00_potential_dsas"
         "/6S8xzxesj2/2.16.840.1.113883.3.16.5618947243828772915608384436445942382"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/00_potential_dsas"
         "/8ahvfqSVx8/2.16.840.1.113883.3.16.219231746599175210986469235923579234273"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/00_potential_dsas"
         "/2614038/1.3.12.2.1107.5.4.5.146694.30000010102213294292100000081.4"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/00_potential_dsas"
         "/2827474/1.3.12.2.1107.5.4.5.146694.30000011031413083546800000231.4"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/00_potential_dsas"
         "/2943631/1.3.12.2.1107.5.4.5.146694.30000011060313021104600000247.4"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/00_potential_dsas"
         "/3593280/1.3.12.2.1107.5.4.5.146694.30000012101703453492100000032.4"),
    Path()
]

# ── Phase 1c: held-out TEST set  ──────────────────────────────────────────
# These sequences were NEVER seen during calibration.
# All sequences inside each root are treated as positives.
TEST_POSITIVE_ROOTS = [
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/6kpsDZBHAH"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/6kWzguQjUG"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/03c64KioU7"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/7VZvAiWXj7"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/1MPUcLN3XP"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/z5CO6YKQrz")
]

# Specific SOPInstanceUID dirs confirmed to be non-DSA.
TEST_NEGATIVE_DIRS = [
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed"
         "/0hCk1LEHWi/2.16.840.1.113883.3.16.150192012923614561440036027316805642868"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed"
         "/0hCk1LEHWi/2.16.840.1.113883.3.16.230081754079257773893093950778753426985"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed"
         "/1chy8vNuWJ/2.16.840.1.113883.3.16.223266400989843036797308954111669120906"),
    Path("/data/Deep_Angiography/DICOM_Sequence_Processed/2z108dmE04/2.16.840.1.113883.3.16.94573398129999594088048109325098324578")
]

# ── Detection / output paths ──────────────────────────────────────────────
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
VAL_CSV           = REPORT_DIR / "validation_report.csv"       # Phase 1b (biased)
TEST_CSV          = REPORT_DIR / "test_report.csv"             # Phase 1c (unbiased)
DETECTION_CSV     = REPORT_DIR / "mask_frame_detection_report.csv"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
WORKERS          = max(1, (os.cpu_count() or 8) - 1)

# Pixels at or below this value are treated as black border and excluded.
BLACK_THRESHOLD = 10

# 15 % headroom on top of raw percentiles so borderline frames are not missed.
MARGIN = 1.15


# =========================================================
# ── Active-pixel helpers ──────────────────────────────────
# =========================================================

def active_pixels(arr: np.ndarray) -> np.ndarray:
    """
    Return a 1-D float32 array with black-border pixels removed.
    arr must be float32, shape (H, W), values 0-255.
    """
    return arr[arr > BLACK_THRESHOLD]


def pixel_entropy(pixels: np.ndarray) -> float:
    """
    Shannon entropy (bits) of an array of pixel values.
    256-bin histogram over [0, 255].  Returns 0.0 for an empty array.

    Why entropy?
    DSA mask frames have uniform gray background → narrow histogram → LOW entropy.
    Non-DSA fluoroscopy has bone/tissue texture → broad histogram → HIGHER entropy.
    This separates cases where std alone is insufficient.
    """
    if pixels.size == 0:
        return 0.0
    counts, _ = np.histogram(pixels, bins=256, range=(0, 255))
    counts = counts[counts > 0].astype(np.float64)
    probs  = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


# =========================================================
# ── Per-frame statistics (black pixels excluded) ──────────
# =========================================================

def frame_stats(img_path_str: str) -> dict | None:
    """
    Compute active-pixel statistics for one grayscale frame.
    Returns None if unreadable or entirely black.
    """
    try:
        arr    = np.array(Image.open(img_path_str).convert("L"), dtype=np.float32)
        active = active_pixels(arr)
        if active.size == 0:
            return None
        return {
            "path":        img_path_str,
            "mean":        float(np.mean(active)),
            "std":         float(np.std(active)),
            "bright_frac": float(np.sum(active > 220)) / active.size,
            "entropy":     pixel_entropy(active),
            "active_frac": float(active.size) / arr.size,
        }
    except Exception:
        return None


# =========================================================
# ── Phase 1: calibration ──────────────────────────────────
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
    Phase 1: learn detection thresholds from CALIBRATION_ROOTS.

    Returns threshold dict:
        MASK_MAX_STD, MASK_MAX_MEAN, MASK_MIN_MEAN,
        MASK_MAX_BRIGHT_FRAC, MASK_MAX_ENTROPY
    """
    print(f"\n{'='*60}")
    print("PHASE 1 — Calibration  (black pixels excluded)")
    for r in CALIBRATION_ROOTS:
        print(f"  Calibration root : {r}")
    print(f"{'='*60}\n")

    frame_paths = []
    for root in CALIBRATION_ROOTS:
        found = collect_frame_paths(root)
        print(f"  {len(found):>8,} frames  ← {root}")
        frame_paths.extend(found)

    if not frame_paths:
        raise RuntimeError(
            "No image frames found in any CALIBRATION_ROOT. "
            "Check paths and directory structure."
        )
    print(f"\nTotal frames across all calibration roots: {len(frame_paths):,}\n")

    # ── Parallel frame-stat computation ──────────────────────────────────
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

    skipped = len(frame_paths) - len(all_stats)
    print(f"Stats computed for {len(all_stats):,} frames "
          f"({skipped} skipped — fully black).")

    # ── Save per-frame CSV ────────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "mean", "std", "bright_frac", "entropy", "active_frac"]
    with CALIB_STATS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_stats)
    print(f"Per-frame calibration stats  -> {CALIB_STATS_CSV}")

    # ── One mask candidate per sequence = frame with lowest active-pixel std
    groups: dict = {}
    for s in all_stats:
        key = str(Path(s["path"]).parent)
        groups.setdefault(key, []).append(s)

    mask_candidates = [
        min(frames, key=lambda s: s["std"])
        for frames in groups.values()
    ]
    print(f"Mask-frame candidates identified : {len(mask_candidates):,} "
          f"(one per sequence — lowest active-pixel std)\n")

    # ── Derive thresholds ─────────────────────────────────────────────────
    stds         = np.array([c["std"]         for c in mask_candidates])
    means        = np.array([c["mean"]        for c in mask_candidates])
    bright_fracs = np.array([c["bright_frac"] for c in mask_candidates])
    entropies    = np.array([c["entropy"]     for c in mask_candidates])

    raw_max_std         = float(np.percentile(stds,         100))
    raw_max_mean        = float(np.percentile(means,        100))
    raw_min_mean        = float(np.percentile(means,          3))
    raw_max_bright_frac = float(np.percentile(bright_fracs, 100))
    raw_max_entropy     = float(np.percentile(entropies,    100))

    MASK_MAX_STD         = round(raw_max_std         * MARGIN,       2)
    MASK_MAX_MEAN        = round(min(255.0, raw_max_mean * MARGIN),  2)
    MASK_MIN_MEAN        = round(raw_min_mean         / MARGIN,      2)
    MASK_MAX_BRIGHT_FRAC = round(min(1.0, raw_max_bright_frac * MARGIN), 4)
    MASK_MAX_ENTROPY     = round(raw_max_entropy      * MARGIN,      4)

    thresholds = {
        "MASK_MAX_STD":         MASK_MAX_STD,
        "MASK_MAX_MEAN":        MASK_MAX_MEAN,
        "MASK_MIN_MEAN":        MASK_MIN_MEAN,
        "MASK_MAX_BRIGHT_FRAC": MASK_MAX_BRIGHT_FRAC,
        "MASK_MAX_ENTROPY":     MASK_MAX_ENTROPY,
    }

    # ── Print + save summary ──────────────────────────────────────────────
    lines = [
        "DSA Mask Frame Detector — Calibration Summary (v4)",
        "=" * 55,
        f"Black-pixel threshold : pixel value <= {BLACK_THRESHOLD}  "
        "(excluded from all stats)",
        "Calibration roots:",
    ]
    for r in CALIBRATION_ROOTS:
        lines.append(f"    {r}")
    lines += [
        f"Total frames scanned : {len(all_stats):,}",
        f"Sequences (groups)   : {len(groups):,}",
        f"Mask candidates      : {len(mask_candidates):,}",
        "",
        "Mask-candidate active-pixel statistics",
        "-" * 40,
        f"  std         min={stds.min():.2f}   median={np.median(stds):.2f}"
        f"   p100={raw_max_std:.2f}   max={stds.max():.2f}",
        f"  mean        min={means.min():.2f}  median={np.median(means):.2f}"
        f"  p100={raw_max_mean:.2f}  max={means.max():.2f}",
        f"  bright_frac   p100={raw_max_bright_frac:.4f}   max={bright_fracs.max():.4f}",
        f"  entropy       min={entropies.min():.4f}  median={np.median(entropies):.4f}"
        f"  p100={raw_max_entropy:.4f}  max={entropies.max():.4f}",
        "",
        f"Margin applied : {MARGIN}x  (15% headroom)",
        "",
        "Derived thresholds",
        "-" * 40,
        f"  MASK_MAX_STD         = {MASK_MAX_STD:<10}  (max * margin)",
        f"  MASK_MAX_MEAN        = {MASK_MAX_MEAN:<10}  (max * margin, cap 255)",
        f"  MASK_MIN_MEAN        = {MASK_MIN_MEAN:<10}  (p3  / margin)",
        f"  MASK_MAX_BRIGHT_FRAC = {MASK_MAX_BRIGHT_FRAC:<10}  (max * margin, cap 1.0)",
        f"  MASK_MAX_ENTROPY     = {MASK_MAX_ENTROPY:<10}  (max * margin)",
    ]
    summary = "\n".join(lines)
    print("\n" + summary + "\n")

    with CALIB_SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"Calibration summary          -> {CALIB_SUMMARY_TXT}\n")

    return thresholds


# =========================================================
# ── Detection helpers (shared by all eval phases + Phase 2)
# =========================================================

def is_mask_frame(img_path_str: str, thresholds: dict) -> bool:
    """
    Returns True if the frame qualifies as a DSA mask frame.
    All checks run on active (non-black) pixels only.
    """
    try:
        arr    = np.array(Image.open(img_path_str).convert("L"), dtype=np.float32)
        active = active_pixels(arr)
        if active.size == 0:
            return False

        mean_val    = float(np.mean(active))
        std_val     = float(np.std(active))
        bright_frac = float(np.sum(active > 220)) / active.size
        entropy_val = pixel_entropy(active)

        return (
            thresholds["MASK_MIN_MEAN"] <= mean_val <= thresholds["MASK_MAX_MEAN"]
            and std_val     <= thresholds["MASK_MAX_STD"]
            and bright_frac <= thresholds["MASK_MAX_BRIGHT_FRAC"]
            and entropy_val <= thresholds["MASK_MAX_ENTROPY"]
        )
    except Exception:
        return False


def analyse_sequence(args: tuple) -> dict:
    """
    Worker: analyse one SOPInstanceUID directory.
    args = (sop_dir_str, thresholds)

    Mirrors calibration exactly:
      1. Compute active-pixel stats for every frame.
      2. Pick lowest-std frame as mask candidate.
      3. Classify only that candidate.
    """
    sop_dir_str, thresholds = args
    sop_dir    = Path(sop_dir_str)
    frames_dir = sop_dir / "frames"

    result = {
        "sequence_dir":        sop_dir_str,
        "total_frames":        0,
        "candidate_frame":     "",
        "candidate_std":       "",
        "candidate_mean":      "",
        "candidate_bright":    "",
        "candidate_entropy":   "",
        "verdict":             "no_mask_detected",
        "copied":              False,
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

    stats_list = [s for p in frame_files if (s := frame_stats(p)) is not None]
    if not stats_list:
        result["verdict"] = "skipped_all_black"
        return result

    # Lowest active-pixel std → mask candidate
    candidate = min(stats_list, key=lambda s: s["std"])
    result["candidate_frame"]   = Path(candidate["path"]).name
    result["candidate_std"]     = round(candidate["std"],         4)
    result["candidate_mean"]    = round(candidate["mean"],        4)
    result["candidate_bright"]  = round(candidate["bright_frac"], 4)
    result["candidate_entropy"] = round(candidate["entropy"],     4)

    if is_mask_frame(candidate["path"], thresholds):
        result["verdict"] = "potential_dsa"

    return result


def run_parallel_analysis(sop_dirs: list, thresholds: dict,
                          desc: str = "Analysing") -> list:
    """Run analyse_sequence in parallel; return list of result dicts."""
    work_items = [(str(d), thresholds) for d in sop_dirs]
    results    = []

    with tqdm(total=len(sop_dirs), unit="seq", desc=desc,
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
                        "sequence_dir":      sop_dir_str,
                        "total_frames":      0,
                        "candidate_frame":   "",
                        "candidate_std":     "",
                        "candidate_mean":    "",
                        "candidate_bright":  "",
                        "candidate_entropy": "",
                        "verdict":           f"worker_error: {e}",
                        "copied":            False,
                    }
                results.append(res)
                pbar.update(1)

    return results


# =========================================================
# ── Shared: collect sequence dirs from a root list ────────
# =========================================================

def collect_sop_dirs_from_roots(roots: list) -> list:
    """
    Collect every sequence directory (parent of a frames/ dir)
    from a list of root paths, at any depth, deduped.
    """
    seen     = set()
    sop_dirs = []
    for root in roots:
        if not root.exists():
            print(f"[WARN] Root not found on disk, skipping: {root}")
            continue
        for frames_dir in root.rglob("frames"):
            if frames_dir.is_dir():
                sop_dir = frames_dir.parent
                if sop_dir not in seen:
                    seen.add(sop_dir)
                    sop_dirs.append(sop_dir)
    return sop_dirs


def collect_valid_dirs(dirs: list) -> list:
    """Filter a list of explicit SOPInstanceUID dirs to those that exist."""
    valid = []
    missing = 0
    for d in dirs:
        if d.is_dir() and (d / "frames").is_dir():
            valid.append(d)
        else:
            missing += 1
            print(f"[WARN] Negative dir not found or has no frames/, skipping: {d}")
    return valid


# =========================================================
# ── Shared: confusion matrix printer / CSV writer ─────────
# =========================================================

def compute_and_print_metrics(
    pos_results: list,
    neg_results: list,
    label: str,
) -> list:
    """
    Compute TP/FN/TN/FP + derived metrics, print them, return eval_rows.
    label — e.g. "VALIDATION" or "TEST"
    """
    TP = sum(1 for r in pos_results if r["verdict"] == "potential_dsa")
    FN = sum(1 for r in pos_results if r["verdict"] != "potential_dsa")
    TN = sum(1 for r in neg_results if r["verdict"] != "potential_dsa")
    FP = sum(1 for r in neg_results if r["verdict"] == "potential_dsa")

    precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    recall    = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else float("nan"))
    accuracy  = ((TP + TN) / (TP + TN + FP + FN)
                 if (TP + TN + FP + FN) > 0 else float("nan"))

    print(f"\n{'='*60}")
    print(f"  {label} — CONFUSION MATRIX")
    print(f"{'='*60}")
    print(f"  TP (correctly detected DSA)       : {TP:>5}")
    print(f"  FN (missed DSA)                   : {FN:>5}")
    print(f"  TN (correctly rejected non-DSA)   : {TN:>5}")
    print(f"  FP (false alarms on non-DSA)      : {FP:>5}")
    print(f"{'-'*60}")
    print(f"  Precision  (TP / TP+FP)           : {precision:>8.4f}")
    print(f"  Recall     (TP / TP+FN)           : {recall:>8.4f}")
    print(f"  F1 Score                          : {f1:>8.4f}")
    print(f"  Accuracy   (TP+TN / all)          : {accuracy:>8.4f}")
    print(f"{'='*60}\n")

    eval_rows = []
    for r in pos_results:
        eval_rows.append({
            **r,
            "true_label": "positive",
            "outcome": "TP" if r["verdict"] == "potential_dsa" else "FN",
        })
    for r in neg_results:
        eval_rows.append({
            **r,
            "true_label": "negative",
            "outcome": "FP" if r["verdict"] == "potential_dsa" else "TN",
        })

    # Surface failures
    fn_rows = [r for r in eval_rows if r["outcome"] == "FN"]
    fp_rows = [r for r in eval_rows if r["outcome"] == "FP"]
    if fn_rows:
        print(f"  False Negatives ({len(fn_rows)}) — DSA sequences missed:")
        for r in fn_rows:
            print(f"    std={r['candidate_std']}  mean={r['candidate_mean']}"
                  f"  bright={r['candidate_bright']}  entropy={r['candidate_entropy']}")
            print(f"    {r['sequence_dir']}")
    if fp_rows:
        print(f"  False Positives ({len(fp_rows)}) — non-DSA wrongly flagged:")
        for r in fp_rows:
            print(f"    std={r['candidate_std']}  mean={r['candidate_mean']}"
                  f"  bright={r['candidate_bright']}  entropy={r['candidate_entropy']}")
            print(f"    {r['sequence_dir']}")

    return eval_rows


def write_eval_csv(eval_rows: list, csv_path: Path) -> None:
    fieldnames = [
        "sequence_dir", "true_label", "outcome",
        "total_frames", "candidate_frame",
        "candidate_std", "candidate_mean", "candidate_bright",
        "candidate_entropy", "verdict",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(eval_rows)
    print(f"  Report written -> {csv_path}\n")


# =========================================================
# ── Phase 1b: validation (in-sample, optimistically biased)
# =========================================================

def validate(thresholds: dict) -> None:
    """
    Phase 1b — sanity check on the calibration positives + known negatives.
    NOTE: because positives drove threshold learning, recall here is
    optimistically biased.  Use Phase 1c test results for reporting.
    """
    print(f"\n{'='*60}")
    print("PHASE 1b — Validation  (in-sample sanity check, biased)")
    print(f"{'='*60}\n")

    pos_dirs = collect_sop_dirs_from_roots(CALIBRATION_ROOTS)
    neg_dirs = collect_valid_dirs(EVAL_NEGATIVE_DIRS)

    print(f"Validation positives (calibration sequences) : {len(pos_dirs):,}")
    print(f"Validation negatives (hand-labelled)         : {len(neg_dirs):,}\n")

    if not pos_dirs:
        print("[WARN] No positive dirs found — skipping validation.")
        return

    pos_results = run_parallel_analysis(pos_dirs, thresholds,
                                        desc="Validating positives")
    neg_results = run_parallel_analysis(neg_dirs, thresholds,
                                        desc="Validating negatives") if neg_dirs else []

    eval_rows = compute_and_print_metrics(pos_results, neg_results, "VALIDATION")
    write_eval_csv(eval_rows, VAL_CSV)


# =========================================================
# ── Phase 1c: test evaluation (held-out, unbiased) ────────
# =========================================================

def test_evaluate(thresholds: dict) -> None:
    """
    Phase 1c — final performance on the fully held-out test set.
    This is the unbiased number to report.
    """
    print(f"\n{'='*60}")
    print("PHASE 1c — Test Evaluation  (held-out, UNBIASED)  ★")
    print(f"{'='*60}\n")

    pos_dirs = collect_sop_dirs_from_roots(TEST_POSITIVE_ROOTS)
    neg_dirs = collect_valid_dirs(TEST_NEGATIVE_DIRS)

    print(f"Test positives : {len(pos_dirs):,}  "
          f"(from {len(TEST_POSITIVE_ROOTS)} root(s))")
    print(f"Test negatives : {len(neg_dirs):,}  "
          f"(explicit SOPInstanceUID dirs)\n")

    if not pos_dirs and not neg_dirs:
        print("[WARN] Test set is empty — skipping test evaluation.")
        return

    pos_results = run_parallel_analysis(pos_dirs, thresholds,
                                        desc="Testing positives") if pos_dirs else []
    neg_results = run_parallel_analysis(neg_dirs, thresholds,
                                        desc="Testing negatives") if neg_dirs else []

    eval_rows = compute_and_print_metrics(pos_results, neg_results, "TEST")
    write_eval_csv(eval_rows, TEST_CSV)


# =========================================================
# ── Phase 2: detection + copy ─────────────────────────────
# =========================================================

def collect_sop_dirs_from_source(source_root: Path) -> list:
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
    sop_dirs = collect_sop_dirs_from_source(SOURCE_ROOT)
    if not sop_dirs:
        print("No sequence directories found under SOURCE_ROOT. Exiting.")
        return
    print(f"Found {len(sop_dirs):,} sequences to analyse.\n")

    results = run_parallel_analysis(sop_dirs, thresholds, desc="Detecting")

    n_potential = sum(1 for r in results if r["verdict"] == "potential_dsa")
    n_no_mask   = sum(1 for r in results if r["verdict"] == "no_mask_detected")
    n_skipped   = sum(1 for r in results
                      if r["verdict"] not in ("potential_dsa", "no_mask_detected"))

    to_copy = [r for r in results if r["verdict"] == "potential_dsa"]
    print(f"\nCopying {len(to_copy):,} potential DSA sequences → {DEST_ROOT} ...")

    with tqdm(total=len(to_copy), unit="seq", desc="Copying",
              dynamic_ncols=True) as pbar:
        for res in to_copy:
            ok = copy_sequence(Path(res["sequence_dir"]))
            res["copied"] = ok
            pbar.update(1)

    fieldnames = [
        "sequence_dir", "total_frames",
        "candidate_frame", "candidate_std", "candidate_mean",
        "candidate_bright", "candidate_entropy",
        "verdict", "copied",
    ]
    with DETECTION_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    copied_ok = sum(1 for r in results if r["copied"])
    copy_fail = sum(1 for r in results
                    if r["verdict"] == "potential_dsa" and not r["copied"])

    print(f"\n{'='*60}")
    print(f"  Sequences examined         : {len(sop_dirs):,}")
    print(f"  Potential DSA (mask found) : {n_potential:,}")
    print(f"  No mask detected           : {n_no_mask:,}")
    print(f"  Skipped                    : {n_skipped:,}")
    print(f"  Successfully copied        : {copied_ok:,}")
    if copy_fail:
        print(f"  Copy failures              : {copy_fail:,}")
    print(f"  Detection report           -> {DETECTION_CSV}")
    print(f"{'='*60}\n")


# =========================================================
# ── Entry point ───────────────────────────────────────────
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DSA Mask Frame Detector — auto-calibrated (v4)"
    )
    parser.add_argument(
        "--calibration_only",
        action="store_true",
        help=(
            "Run Phase 1 (calibration), Phase 1b (validation), and "
            "Phase 1c (test evaluation) only.  Skip Phase 2 (detection + copy)."
        ),
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1 — always runs: learn thresholds
    thresholds = calibrate()

    # Phase 1b — always runs: in-sample sanity check (biased, do not report)
    validate(thresholds)

    # Phase 1c — always runs: held-out test evaluation (unbiased, report this)
    test_evaluate(thresholds)

    if args.calibration_only:
        print("--calibration_only flag set: skipping Phase 2 (detection + copy).")
        print("All done.")
        return

    # Phase 2 — full scan + copy
    detect_and_copy(thresholds)
    print("All done.")


if __name__ == "__main__":
    main()