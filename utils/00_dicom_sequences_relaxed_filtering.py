#!/usr/bin/env python3
"""
Parallel DICOM processing  —  RELAXED FILTER variant
=====================================================

Difference from the original dicom_parallel_processor.py
---------------------------------------------------------
The SeriesDescription filter has been REMOVED.

Original 4-step eligibility order:
    1. RadiationSetting  == "GR"          ← KEPT
    2. SeriesDescription contains "DSA" or "CO 2"  ← REMOVED
    3. PositionerMotion  == "STATIC"      ← KEPT
    4. NumberOfFrames    >  min_frames    ← KEPT

This means sequences that previously failed ONLY because of
bad_series (SeriesDescription) will now pass — those are the
~20 726 new sequences we want to capture.

Output directory
----------------
Newly passing sequences are written to:
    /data/Deep_Angiography/DICOM_Sequence_Processed/00_sequence_to_check/

The original output directory (OUTPUT_ROOT from config.py) is used
ONLY for the already-existing dest-check so we never re-extract
sequences that were processed by the original run.

Concretely, for every file:
    • Check original dest  → if it already exists, SKIP  (already done)
    • Check new dest       → if it already exists, SKIP  (already done this run)
    • Apply relaxed filter → if it passes, extract to NEW dest
    • If it fails the relaxed filter, record as filtered (these were previously
      failing at step 1, 3, or 4 — not step 2)

Design / race-condition safety / log structure
----------------------------------------------
Identical to the original script — see its module docstring for details.
"""

import csv
import datetime
import gc
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from pydicom.multival import MultiValue
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Load paths from config.py ─────────────────────────────────────────────────
try:
    from config import INPUT_ROOT, OUTPUT_ROOT
except ImportError as e:
    raise SystemExit(
        f"Could not import INPUT_ROOT / OUTPUT_ROOT from config.py: {e}\n"
        "Make sure config.py is in the same directory as this script."
    )

# =========================================================
# Configuration
# =========================================================
FRAME_FORMAT    = "png"
NA_VALUE        = "NA"
MAX_VALUE_CHARS = 2000

SKIP_KEYWORDS = {
    "PixelData", "WaveformData", "OverlayData",
    "EncapsulatedDocument", "CurveData", "AudioSampleData",
}
SKIP_TAGS = {(0x7FE0, 0x0010)}  # PixelData

# Eligibility filter — SeriesDescription check intentionally absent
REQUIRED_RADIATION_SETTING  = "GR"
REQUIRED_POSITIONER_MOTION  = "STATIC"
# NOTE: SERIES_DESCRIPTION_KEYWORDS kept for reporting / filtered.csv only
SERIES_DESCRIPTION_KEYWORDS = ("DSA", "CO 2")
DEFAULT_MIN_FRAMES          = 2

# ── Directories ───────────────────────────────────────────────────────────────
# Where newly-passing sequences are written
NEW_OUTPUT_ROOT = Path(
    "/data/Deep_Angiography/DICOM_Sequence_Processed/00_sequence_to_check"
)

# All logs go here
LOG_DIR = Path("/data/Deep_Angiography/DICOM-metadata-stats")


# =========================================================
# Per-worker stats accumulator
# =========================================================
@dataclass
class WorkerStats:
    total_found:    int = 0
    processed:      int = 0
    skipped_done:   int = 0   # dest already existed (original OR new dir)
    skipped_filter: int = 0   # failed relaxed eligibility
    errors:         int = 0

    error_rows:    List[Dict] = field(default_factory=list)
    filtered_rows: List[Dict] = field(default_factory=list)
    skipped_rows:  List[Dict] = field(default_factory=list)

    # filter breakdown (relaxed — no bad_series)
    bad_radiation:  int = 0
    bad_motion:     int = 0
    too_few_frames: int = 0
    filter_error:   int = 0


# =========================================================
# Utilities  (identical to original)
# =========================================================
def is_probably_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            if f.read(4) == b"DICM":
                return True
    except OSError:
        return False
    try:
        pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return True
    except (pydicom.errors.InvalidDicomError, OSError, ValueError):
        return False


def is_nullish(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, (list, tuple, MultiValue)) and len(v) == 0:
        return True
    return False


def safe_str(x: object) -> str:
    try:
        return str(x)
    except (TypeError, ValueError, UnicodeDecodeError):
        return NA_VALUE


def normalize_value(value) -> str:
    if is_nullish(value):
        return NA_VALUE
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary:{len(value)} bytes>"
    if isinstance(value, MultiValue):
        s = ";".join(safe_str(v) for v in value if v is not None).strip()
        return s or NA_VALUE
    s = safe_str(value).strip()
    if not s:
        return NA_VALUE
    if len(s) > MAX_VALUE_CHARS:
        s = s[:MAX_VALUE_CHARS] + "...<truncated>"
    return s


def extract_metadata_pairs(ds) -> List[Dict[str, str]]:
    rows = []
    for elem in ds:
        if elem.VR == "SQ":
            continue
        if elem.keyword in SKIP_KEYWORDS:
            continue
        if (int(elem.tag.group), int(elem.tag.element)) in SKIP_TAGS:
            continue
        if is_nullish(elem.value):
            continue
        key   = elem.keyword if elem.keyword else str(elem.tag)
        value = normalize_value(elem.value)
        rows.append({"Information": key, "Value": value})
    return rows


def sanitize_dirname(name: str, max_len: int = 150) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._-")
    return (name or "unknown")[:max_len]


def get_tag_str(ds, tag: str) -> str:
    val = getattr(ds, tag, None)
    if val is None:
        return ""
    if isinstance(val, MultiValue):
        val = val[0] if val else ""
    return str(val).strip()


# =========================================================
# Output path  (single source of truth)
# =========================================================
def output_dir_for(
    output_root:      Path,
    accession_number: str,
    sop_instance_uid: str,
) -> Path:
    acc = sanitize_dirname(accession_number) if accession_number else "NO_ACCESSION"
    sop = sop_instance_uid                   if sop_instance_uid  else "NO_UID"
    return output_root / acc / sop


def dest_already_exists(
    output_root:      Path,
    accession_number: str,
    sop_instance_uid: str,
) -> bool:
    d = output_dir_for(output_root, accession_number, sop_instance_uid)
    return (d / "metadata.csv").exists() and (d / "frames").exists()


# =========================================================
# RELAXED eligibility filter  — SeriesDescription removed
# =========================================================
def passes_eligibility_filter_relaxed(ds, min_frames: int) -> Tuple[bool, str]:
    """
    3-step filter (SeriesDescription step intentionally omitted):
        1. RadiationSetting  == "GR"
        2. PositionerMotion  == "STATIC"
        3. NumberOfFrames    >  min_frames
    """
    def _get(tag: str) -> str:
        val = getattr(ds, tag, None)
        if val is None:
            return ""
        if isinstance(val, MultiValue):
            val = val[0] if val else ""
        return str(val).strip().upper()

    if _get("RadiationSetting") != REQUIRED_RADIATION_SETTING:
        return False, "bad_radiation"

    # SeriesDescription check deliberately skipped here

    if _get("PositionerMotion") != REQUIRED_POSITIONER_MOTION:
        return False, "bad_motion"

    nof_raw = getattr(ds, "NumberOfFrames", None)
    if nof_raw is not None:
        try:
            if int(str(nof_raw).strip()) <= min_frames:
                return False, "too_few_frames"
        except (ValueError, TypeError):
            pass

    return True, "ok"


# =========================================================
# Image conversion  (identical to original)
# =========================================================
def to_uint8_windowed(arr: np.ndarray, ds) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)

    slope     = float(getattr(ds, "RescaleSlope",     1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    center = getattr(ds, "WindowCenter", None)
    width  = getattr(ds, "WindowWidth",  None)
    if isinstance(center, MultiValue): center = center[0]
    if isinstance(width,  MultiValue): width  = width[0]

    if center is not None and width is not None and width > 0:
        lo  = center - width / 2
        hi  = center + width / 2
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    else:
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

    if "MONOCHROME1" in str(getattr(ds, "PhotometricInterpretation", "")).upper():
        arr = 1.0 - arr

    return (arr * 255).astype(np.uint8)


def save_frames(ds, frames_dir: Path, base_name: str) -> int:
    # Access pixel_array BEFORE creating any directory.
    # If this raises, no empty dir is left on disk.
    px = ds.pixel_array

    # Only create the directory once we know pixel data is readable.
    frames_dir.mkdir(parents=True, exist_ok=True)

    def _save(img, idx):
        Image.fromarray(img, mode="L").save(
            frames_dir / f"{base_name}_frame_{idx:04d}.{FRAME_FORMAT}"
        )

    if px.ndim == 2:
        _save(to_uint8_windowed(px, ds), 1)
        return 1
    if px.ndim == 3:
        for i in range(px.shape[0]):
            _save(to_uint8_windowed(px[i], ds), i + 1)
        return px.shape[0]

    raise ValueError(f"Unexpected pixel_array shape: {px.shape}")


# =========================================================
# Directory discovery  (identical to original)
# =========================================================
def collect_leaf_dirs(root_dir: Path) -> List[Path]:
    leaf_dirs = []
    for dirpath, _, filenames in os.walk(root_dir):
        if filenames:
            leaf_dirs.append(Path(dirpath))
    return leaf_dirs


# =========================================================
# Worker
# =========================================================
def _process_leaf_dirs(
    leaf_dir_strs:    List[str],
    original_out_str: str,   # used ONLY for already-exists check
    new_out_str:      str,   # where newly-passing sequences are written
    min_frames:       int,
    skip_existing:    bool,
) -> WorkerStats:
    """
    For each DICOM file:
        1. Read header
        2. Extract identifiers
        3. Skip if already in ORIGINAL output  (processed by prior run)
        4. Skip if already in NEW output       (processed earlier this run)
        5. Apply RELAXED eligibility filter    (no SeriesDescription check)
        6. Extract frames + write metadata to NEW output dir
    """
    GC_NUDGE_INTERVAL = 10_000
    original_out = Path(original_out_str)
    new_out      = Path(new_out_str)
    stats        = WorkerStats()
    file_count   = 0

    for leaf_str in leaf_dir_strs:
        leaf = Path(leaf_str)
        for f in leaf.iterdir():
            if not f.is_file() or not is_probably_dicom(f):
                continue

            stats.total_found += 1
            file_count        += 1
            if file_count % GC_NUDGE_INTERVAL == 0:
                gc.collect()

            # ── 1. Read header ─────────────────────────────────────────────
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            except Exception as e:
                stats.errors += 1
                stats.error_rows.append({
                    "file":   str(f),
                    "stage":  "header_read",
                    "reason": f"{type(e).__name__}: {e}",
                })
                continue

            # ── 2. Extract identifiers ─────────────────────────────────────
            uid = get_tag_str(ds, "SOPInstanceUID") or None
            acc = get_tag_str(ds, "AccessionNumber")

            # ── 3. Skip if already in ORIGINAL output ──────────────────────
            #    These were processed by the original stricter run — we don't
            #    want to duplicate them in 00_sequence_to_check.
            if skip_existing and uid and \
                    dest_already_exists(original_out, acc, uid):
                stats.skipped_done += 1
                stats.skipped_rows.append({
                    "file":             str(f),
                    "accession_number": acc,
                    "sop_instance_uid": uid,
                    "dest":             str(output_dir_for(original_out, acc, uid)),
                    "skip_reason":      "already_in_original_output",
                })
                del ds
                continue

            # ── 4. Skip if already in NEW output ──────────────────────────
            if skip_existing and uid and \
                    dest_already_exists(new_out, acc, uid):
                stats.skipped_done += 1
                stats.skipped_rows.append({
                    "file":             str(f),
                    "accession_number": acc,
                    "sop_instance_uid": uid,
                    "dest":             str(output_dir_for(new_out, acc, uid)),
                    "skip_reason":      "already_in_new_output",
                })
                del ds
                continue

            # ── 5. Relaxed eligibility filter (no SeriesDescription) ───────
            try:
                ok, reason = passes_eligibility_filter_relaxed(ds, min_frames)
            except Exception as fe:
                ok     = False
                reason = "filter_error"
                stats.filter_error += 1
                stats.error_rows.append({
                    "file":   str(f),
                    "stage":  "filter_eval",
                    "reason": f"{type(fe).__name__}: {fe}",
                })

            if not ok:
                stats.skipped_filter += 1
                setattr(stats, reason, getattr(stats, reason, 0) + 1)
                stats.filtered_rows.append({
                    "file":             str(f),
                    "reason":           reason,
                    "accession_number": acc,
                    "sop_instance_uid": uid or "",
                    "radiation":        get_tag_str(ds, "RadiationSetting"),
                    "series_desc":      get_tag_str(ds, "SeriesDescription"),
                    "positioner":       get_tag_str(ds, "PositionerMotion"),
                    "num_frames":       get_tag_str(ds, "NumberOfFrames"),
                })
                del ds
                continue

            # ── 6. Extract frames + write metadata to NEW dir ──────────────
            # FIX: no mkdir here — mkdir is now inside save_frames, deferred
            # until after pixel_array is confirmed readable.
            per_dicom_dir = output_dir_for(new_out, acc, uid or "NO_UID")
            frames_dir    = per_dicom_dir / "frames"
            metadata_csv  = per_dicom_dir / "metadata.csv"
            metadata_tmp  = per_dicom_dir / "metadata.csv.tmp"

            try:
                # FIX 2: full pixel read + save_frames are ONE atomic step.
                # save_frames raises on any failure — no silent NA_VALUE path.
                ds_full   = pydicom.dcmread(f, force=True)
                base_name = uid if uid else sanitize_dirname(f.stem)
                frame_count = save_frames(ds_full, frames_dir, base_name)
                del ds_full

                metadata_rows = extract_metadata_pairs(ds)
                metadata_rows.extend([
                    {"Information": "source_file",      "Value": safe_str(f.name)},
                    {"Information": "source_path",      "Value": safe_str(str(f))},
                    {"Information": "frame_count",      "Value": safe_str(frame_count)},
                    {"Information": "accession_number", "Value": safe_str(acc)},
                    {"Information": "sop_instance_uid", "Value": safe_str(uid)},
                    {"Information": "filter_variant",   "Value": "relaxed_no_series_desc"},
                ])

                df = pd.DataFrame(metadata_rows)
                df["Information"] = df["Information"].fillna(NA_VALUE).map(safe_str)
                df["Value"]       = df["Value"].fillna(NA_VALUE).map(safe_str)

                df.to_csv(metadata_tmp, index=False, encoding="utf-8")
                os.replace(metadata_tmp, metadata_csv)

                stats.processed += 1

            except Exception as e:
                stats.errors += 1
                stats.error_rows.append({
                    "file":   str(f),
                    "stage":  "processing",
                    "reason": f"{type(e).__name__}: {e}",
                })
                # FIX 3: remove the entire per_dicom_dir on any failure so
                # no empty skeleton dirs are left on disk, and re-runs do not
                # get stuck skipping half-created destinations forever.
                import shutil
                try:
                    shutil.rmtree(per_dicom_dir, ignore_errors=True)
                except OSError:
                    pass
                try:
                    metadata_tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            finally:
                del ds

    return stats


# =========================================================
# Stats merging
# =========================================================
def _merge_stats(all_stats: List[WorkerStats]) -> WorkerStats:
    merged = WorkerStats()
    for s in all_stats:
        merged.total_found      += s.total_found
        merged.processed        += s.processed
        merged.skipped_done     += s.skipped_done
        merged.skipped_filter   += s.skipped_filter
        merged.errors           += s.errors
        merged.bad_radiation    += s.bad_radiation
        merged.bad_motion       += s.bad_motion
        merged.too_few_frames   += s.too_few_frames
        merged.filter_error     += s.filter_error
        merged.error_rows       += s.error_rows
        merged.filtered_rows    += s.filtered_rows
        merged.skipped_rows     += s.skipped_rows
    return merged


# =========================================================
# Report writing
# =========================================================
def _write_reports(
    run_dir:        Path,
    stats:          WorkerStats,
    run_ts:         str,
    input_root:     Path,
    original_out:   Path,
    new_out:        Path,
    n_workers:      int,
    min_frames:     int,
    skip_existing:  bool,
    leaf_dir_count: int,
    elapsed_sec:    float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    dt = (f"{run_ts[:4]}-{run_ts[4:6]}-{run_ts[6:8]} "
          f"{run_ts[9:11]}:{run_ts[11:13]}:{run_ts[13:15]}")

    # errors.csv
    if stats.error_rows:
        with (run_dir / "errors.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "stage", "reason"])
            w.writeheader(); w.writerows(stats.error_rows)

    # filtered.csv
    if stats.filtered_rows:
        with (run_dir / "filtered.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "file", "reason", "accession_number", "sop_instance_uid",
                "radiation", "series_desc", "positioner", "num_frames",
            ])
            w.writeheader(); w.writerows(stats.filtered_rows)

    # skipped.csv  (extended with skip_reason column)
    if stats.skipped_rows:
        with (run_dir / "skipped.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "file", "accession_number", "sop_instance_uid",
                "dest", "skip_reason",
            ])
            w.writeheader(); w.writerows(stats.skipped_rows)

    # summary.md
    with (run_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# DICOM Processing Run Summary  —  Relaxed Filter Variant\n\n")
        f.write(f"**Run timestamp:** {dt}  \n")
        f.write(f"**Elapsed:** {elapsed_sec:.1f} s  \n")
        f.write(f"**Input root:** `{input_root}`  \n")
        f.write(f"**Original output (skip-check only):** `{original_out}`  \n")
        f.write(f"**New output (newly-passing sequences):** `{new_out}`  \n")
        f.write(f"**Workers:** {n_workers}  \n")
        f.write(f"**Min frames threshold:** {min_frames}  \n")
        f.write(f"**Skip existing:** {skip_existing}  \n")
        f.write(f"**Leaf directories scanned:** {leaf_dir_count}  \n\n")
        f.write("> **Filter change:** `SeriesDescription` check has been "
                "**removed**. Only `RadiationSetting == GR`, "
                "`PositionerMotion == STATIC`, and `NumberOfFrames > "
                f"{min_frames}` are applied.\n\n")
        f.write("---\n\n")

        f.write("## Overall Counts\n\n")
        f.write("| Metric | Count |\n|--------|-------|\n")
        f.write(f"| Total DICOM files found                    | **{stats.total_found}** |\n")
        f.write(f"| Successfully processed → new output        | **{stats.processed}** |\n")
        f.write(f"| Skipped (already in original or new dir)   | **{stats.skipped_done}** |\n")
        f.write(f"| Rejected (relaxed eligibility filter)      | **{stats.skipped_filter}** |\n")
        f.write(f"| Errors (exceptions)                        | **{stats.errors}** |\n\n")

        f.write("## Relaxed Eligibility Filter Breakdown\n\n")
        f.write("> SeriesDescription check has been removed from this run.\n\n")
        f.write("| Reason | Count | Meaning |\n|--------|-------|---------|\n")
        f.write(f"| `bad_radiation`  | **{stats.bad_radiation}** "
                f"| RadiationSetting ≠ `{REQUIRED_RADIATION_SETTING}` |\n")
        f.write(f"| `bad_motion`     | **{stats.bad_motion}** "
                f"| PositionerMotion ≠ `{REQUIRED_POSITIONER_MOTION}` |\n")
        f.write(f"| `too_few_frames` | **{stats.too_few_frames}** "
                f"| NumberOfFrames ≤ {min_frames} |\n")
        f.write(f"| `filter_error`   | **{stats.filter_error}** "
                f"| Exception during filter evaluation |\n\n")

        f.write("## Skipped Files\n\n")
        f.write(f"{stats.skipped_done} file(s) skipped — already present in "
                "original or new output directory.  \n")
        if stats.skipped_rows:
            f.write(f"See `skipped.csv` ({len(stats.skipped_rows)} rows) — "
                    "column `skip_reason` distinguishes origin.\n\n")

        f.write("## Errors\n\n")
        if stats.errors == 0:
            f.write("No errors encountered.\n\n")
        else:
            stage_counts: Dict[str, int] = defaultdict(int)
            for row in stats.error_rows:
                stage_counts[row["stage"]] += 1
            f.write("| Stage | Count |\n|-------|-------|\n")
            for stage, cnt in sorted(stage_counts.items()):
                f.write(f"| `{stage}` | {cnt} |\n")
            f.write(f"\nSee `errors.csv` ({len(stats.error_rows)} rows)\n\n")

        f.write("---\n\n")
        f.write("*Report generated by dicom_parallel_processor_relaxed.py*\n")

    print(f"\n{'='*60}")
    print(f"  Run summary written to: {run_dir}")
    print(f"{'='*60}")


# =========================================================
# Main orchestrator
# =========================================================
def process_root_directory(
    root_dir:      Path,
    original_out:  Path,
    new_out:       Path,
    workers:       int,
    min_frames:    int,
    skip_existing: bool,
):
    run_ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_DIR / f"run_relaxed_{run_ts}"
    t_start = datetime.datetime.now()

    print(f"Input            : {root_dir}")
    print(f"Original output  : {original_out}  (skip-check only)")
    print(f"New output       : {new_out}")
    print(f"Logs             : {run_dir}")
    print(f"Workers          : {workers}  |  min_frames: {min_frames}  "
          f"|  skip_existing: {skip_existing}")
    print(f"Filter change    : SeriesDescription check REMOVED\n")

    print("Discovering leaf directories...")
    leaf_dirs = collect_leaf_dirs(root_dir)
    if not leaf_dirs:
        print("No directories with files found. Exiting.")
        return

    n_workers       = max(1, min(workers, len(leaf_dirs)))
    original_out_str = str(original_out)
    new_out_str      = str(new_out)

    print(f"Found {len(leaf_dirs)} leaf directories — "
          f"dividing across {n_workers} worker(s).\n")

    all_stats:    List[WorkerStats] = []
    agg_found     = 0
    agg_processed = 0
    agg_skipped   = 0
    agg_filtered  = 0
    agg_errors    = 0

    with tqdm(
        total        = len(leaf_dirs),
        unit         = "dir",
        desc         = "Processing dirs",
        dynamic_ncols= True,
    ) as pbar:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_dir = {
                ex.submit(
                    _process_leaf_dirs,
                    [str(d)],
                    original_out_str,
                    new_out_str,
                    min_frames,
                    skip_existing,
                ): str(d)
                for d in leaf_dirs
            }

            for fut in as_completed(future_to_dir):
                dir_str = future_to_dir[fut]
                try:
                    s = fut.result()
                    all_stats.append(s)
                    agg_found     += s.total_found
                    agg_processed += s.processed
                    agg_skipped   += s.skipped_done
                    agg_filtered  += s.skipped_filter
                    agg_errors    += s.errors
                except Exception as e:
                    tqdm.write(
                        f"[WORKER ERROR] {Path(dir_str).name}: "
                        f"{type(e).__name__}: {e}"
                    )

                pbar.set_postfix(
                    found     = agg_found,
                    new       = agg_processed,
                    filtered  = agg_filtered,
                    skipped   = agg_skipped,
                    errors    = agg_errors,
                )
                pbar.set_description(f"Processing  [{Path(dir_str).name[:30]}]")
                pbar.update(1)

    merged      = _merge_stats(all_stats)
    elapsed_sec = (datetime.datetime.now() - t_start).total_seconds()

    print(f"\n{'='*60}")
    print(f"  Total found        : {merged.total_found}")
    print(f"  Newly processed    : {merged.processed}  → {new_out}")
    print(f"  Skipped (existing) : {merged.skipped_done}")
    print(f"  Filtered out       : {merged.skipped_filter}")
    print(f"    bad_radiation    : {merged.bad_radiation}")
    print(f"    bad_motion       : {merged.bad_motion}")
    print(f"    too_few_frames   : {merged.too_few_frames}")
    print(f"    filter_error     : {merged.filter_error}")
    print(f"  Errors             : {merged.errors}")
    print(f"  Elapsed            : {elapsed_sec:.1f} s")
    print(f"{'='*60}")

    _write_reports(
        run_dir        = run_dir,
        stats          = merged,
        run_ts         = run_ts,
        input_root     = root_dir,
        original_out   = original_out,
        new_out        = new_out,
        n_workers      = n_workers,
        min_frames     = min_frames,
        skip_existing  = skip_existing,
        leaf_dir_count = len(leaf_dirs),
        elapsed_sec    = elapsed_sec,
    )


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Relaxed-filter DICOM processor — SeriesDescription check removed.\n"
            "Newly-passing sequences (the ~20 726 previously blocked by bad_series)\n"
            "are written to 00_sequence_to_check/. Already-processed sequences\n"
            "(present in the original OUTPUT_ROOT) are skipped automatically."
        )
    )
    parser.add_argument(
        "--input_root", type=Path, default=Path(INPUT_ROOT),
        help="Root directory to scan for DICOM files",
    )
    parser.add_argument(
        "--original_output", type=Path, default=Path(OUTPUT_ROOT),
        help="Original output root — used ONLY for already-exists check",
    )
    parser.add_argument(
        "--new_output", type=Path, default=NEW_OUTPUT_ROOT,
        help="Where newly-passing sequences are written (default: 00_sequence_to_check)",
    )
    parser.add_argument(
        "--workers", type=int,
        default=max(1, (os.cpu_count() or 8) - 1),
        help="Number of worker processes (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--min_frames", type=int, default=DEFAULT_MIN_FRAMES,
        help=f"Reject DICOMs with NumberOfFrames <= this (default: {DEFAULT_MIN_FRAMES})",
    )
    parser.add_argument(
        "--skip_existing",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
        help="Skip DICOMs already present in original or new output (default: True)",
    )

    args = parser.parse_args()
    args.new_output.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    process_root_directory(
        root_dir     = args.input_root,
        original_out = args.original_output,
        new_out      = args.new_output,
        workers      = args.workers,
        min_frames   = args.min_frames,
        skip_existing= args.skip_existing,
    )