#!/usr/bin/env python3
"""
Parallel DICOM processing  —  worker-centric architecture
=========================================================

Design
------
No pre-scan phase. The input directory tree is walked once in the main
process to collect leaf directories (directories that contain files).
Those leaf dirs are divided evenly among N worker processes.

Each worker is fully self-sufficient:
    1. Walk its assigned leaf dirs.
    2. For every DICOM file found:
        a. Read the header (stop_before_pixels=True).
        b. Apply eligibility filter  (RadiationSetting / SeriesDescription /
           PositionerMotion / NumberOfFrames).
        c. Construct the destination path  output_root/AccessionNumber/SOPInstanceUID/
           and stat-check whether it already exists  (O(1) filesystem lookup).
        d. If it passes the filter AND the destination is empty → extract frames
           and write metadata.csv  (atomic: .tmp → os.replace).
    3. Accumulate per-worker stats for every decision made.

After all workers finish, the main process merges stats and writes a
detailed final report covering every file seen at every stage.

Race-condition safety
---------------------
- Every file lands in  AccessionNumber/SOPInstanceUID/  — a path derived
  entirely from DICOM tags that are globally unique per instance.
- No two workers will ever produce the same SOPInstanceUID (DICOM standard
  guarantees uniqueness), so no two workers ever write to the same dir.
- Even if the same file somehow appears in two workers' queues, the O(1)
  dest check in step (c) means the second worker detects the first worker's
  completed output and skips it.
- metadata.csv is written atomically (tmp → os.replace) so a partially
  written file is never mistaken for a completed one.

Output structure
----------------
    output_root/
        <AccessionNumber>/
            <SOPInstanceUID>/
                frames/
                    <SOPInstanceUID>_frame_0001.png
                    ...
                metadata.csv

Logs  (always written to LOG_DIR)
------
    LOG_DIR/
        run_<YYYYMMDD_HHMMSS>/
            summary.md          ← full human-readable report
            errors.csv          ← one row per file that raised an exception
            filtered.csv        ← one row per file rejected by eligibility filter
            skipped.csv         ← one row per file skipped (dest already exists)

Tips
----
- --workers      number of processing workers  (default: cpu_count - 1)
- --min_frames   reject DICOMs with NumberOfFrames <= this  (default: 2)
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

# Eligibility filter
REQUIRED_RADIATION_SETTING  = "GR"
REQUIRED_POSITIONER_MOTION  = "STATIC"
SERIES_DESCRIPTION_KEYWORDS = ("DSA", "CO 2")
DEFAULT_MIN_FRAMES          = 2

# All logs go here, never mixed with DICOM output
LOG_DIR = Path("/data/Deep_Angiography/DICOM-metadata-stats")


# =========================================================
# Per-worker stats accumulator
# =========================================================
@dataclass
class WorkerStats:
    """
    Accumulated by each worker and returned to the main process.
    All lists store dicts for later CSV / MD rendering.
    """
    total_found:   int = 0
    processed:     int = 0
    skipped_done:  int = 0   # dest already existed
    skipped_filter: int = 0  # failed eligibility
    errors:        int = 0   # exception during processing

    # detailed rows for the per-category CSVs
    error_rows:    List[Dict] = field(default_factory=list)
    filtered_rows: List[Dict] = field(default_factory=list)
    skipped_rows:  List[Dict] = field(default_factory=list)

    # filter breakdown
    bad_radiation:  int = 0
    bad_series:     int = 0
    bad_motion:     int = 0
    too_few_frames: int = 0
    filter_error:   int = 0   # exception inside filter evaluation


# =========================================================
# Utilities
# =========================================================
def is_probably_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            if f.read(4) == b"DICM":
                return True
    except Exception:
        return False
    try:
        pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return True
    except Exception:
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
    except Exception:
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
    """
    output_root / <AccessionNumber> / <SOPInstanceUID>

    SOPInstanceUID contains only digits and dots — already filesystem-safe.
    AccessionNumber may contain arbitrary chars — sanitized.
    """
    acc = sanitize_dirname(accession_number) if accession_number else "NO_ACCESSION"
    sop = sop_instance_uid                  if sop_instance_uid  else "NO_UID"
    return output_root / acc / sop


def dest_already_exists(
    output_root:      Path,
    accession_number: str,
    sop_instance_uid: str,
) -> bool:
    """
    O(1) check: two stat() calls on a deterministically constructed path.
    Returns True only if BOTH frames/ and metadata.csv exist — i.e. a
    previous run completed this file successfully.
    """
    d = output_dir_for(output_root, accession_number, sop_instance_uid)
    return (d / "metadata.csv").exists() and (d / "frames").exists()


# =========================================================
# Eligibility filter
# =========================================================
def passes_eligibility_filter(ds, min_frames: int) -> Tuple[bool, str]:
    """
    Returns (True, "ok") or (False, reason_string).
    Evaluation order matches filter_sequence_dirs_by_metadata_and_frames.py:
        1. RadiationSetting  == "GR"
        2. SeriesDescription contains "DSA" or "CO 2"
        3. PositionerMotion  == "STATIC"
        4. NumberOfFrames    >  min_frames
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

    sd = _get("SeriesDescription")
    if not any(kw.upper() in sd for kw in SERIES_DESCRIPTION_KEYWORDS):
        return False, "bad_series"

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
# Image conversion
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
    px = ds.pixel_array

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
# Directory discovery
# =========================================================
def collect_leaf_dirs(root_dir: Path) -> List[Path]:
    """
    Return directories that contain at least one file directly.
    These are the units of work handed to workers — no rglob overlap
    because leaf dirs never contain each other.
    """
    leaf_dirs = []
    for dirpath, _, filenames in os.walk(root_dir):
        if filenames:
            leaf_dirs.append(Path(dirpath))
    return leaf_dirs


# =========================================================
# Worker
# =========================================================
def _process_leaf_dirs(
    leaf_dir_strs:   List[str],
    output_root_str: str,
    min_frames:      int,
    skip_existing:   bool,
) -> WorkerStats:
    """
    Process all DICOM files found directly inside the given leaf directories.

    Pipeline per file:
        1. Read header  (stop_before_pixels=True)
        2. Extract SOPInstanceUID + AccessionNumber
        3. Eligibility filter
        4. O(1) dest-exists check
        5. Extract frames + write metadata  (atomic)

    Returns a WorkerStats with full accounting of every decision.

    GC hygiene: explicit del ds + nudge every GC_NUDGE_INTERVAL files.
    """
    GC_NUDGE_INTERVAL = 10_000
    output_root = Path(output_root_str)
    stats       = WorkerStats()
    file_count  = 0

    for leaf_str in leaf_dir_strs:
        leaf = Path(leaf_str)
        for f in leaf.iterdir():          # flat — no rglob; leaf has only files
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

            # ── 3. Eligibility filter ──────────────────────────────────────
            try:
                ok, reason = passes_eligibility_filter(ds, min_frames)
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
                # increment breakdown counter
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

            # ── 4. O(1) dest-exists check ──────────────────────────────────
            if skip_existing and uid and \
                    dest_already_exists(output_root, acc, uid):
                stats.skipped_done += 1
                stats.skipped_rows.append({
                    "file":             str(f),
                    "accession_number": acc,
                    "sop_instance_uid": uid,
                    "dest":             str(output_dir_for(output_root, acc, uid)),
                })
                del ds
                continue

            # ── 5. Extract frames + write metadata ─────────────────────────
            per_dicom_dir = output_dir_for(output_root, acc, uid or "NO_UID")
            frames_dir    = per_dicom_dir / "frames"
            metadata_csv  = per_dicom_dir / "metadata.csv"
            metadata_tmp  = per_dicom_dir / "metadata.csv.tmp"

            try:
                frames_dir.mkdir(parents=True, exist_ok=True)

                # Full read for pixel data
                ds_full = pydicom.dcmread(f, force=True)

                base_name = uid if uid else sanitize_dirname(f.stem)
                try:
                    frame_count = save_frames(ds_full, frames_dir, base_name)
                except Exception as fe:
                    frame_count = NA_VALUE
                    stats.error_rows.append({
                        "file":   str(f),
                        "stage":  "frame_extraction",
                        "reason": f"{type(fe).__name__}: {fe}",
                    })
                del ds_full

                # Metadata
                metadata_rows = extract_metadata_pairs(ds)
                metadata_rows.extend([
                    {"Information": "source_file",      "Value": safe_str(f.name)},
                    {"Information": "source_path",      "Value": safe_str(str(f))},
                    {"Information": "frame_count",      "Value": safe_str(frame_count)},
                    {"Information": "accession_number", "Value": safe_str(acc)},
                    {"Information": "sop_instance_uid", "Value": safe_str(uid)},
                ])

                df = pd.DataFrame(metadata_rows)
                df["Information"] = df["Information"].fillna(NA_VALUE).map(safe_str)
                df["Value"]       = df["Value"].fillna(NA_VALUE).map(safe_str)

                # Atomic write
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
                try:
                    metadata_tmp.unlink(missing_ok=True)
                except Exception:
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
        merged.bad_series       += s.bad_series
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
    run_dir:       Path,
    stats:         WorkerStats,
    run_ts:        str,
    input_root:    Path,
    output_root:   Path,
    n_workers:     int,
    min_frames:    int,
    skip_existing: bool,
    leaf_dir_count: int,
    elapsed_sec:   float,
) -> None:
    """
    Write four files into run_dir:
        summary.md      human-readable full report
        errors.csv      files that raised exceptions
        filtered.csv    files rejected by eligibility filter
        skipped.csv     files skipped (dest already existed)
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    dt = (f"{run_ts[:4]}-{run_ts[4:6]}-{run_ts[6:8]} "
          f"{run_ts[9:11]}:{run_ts[11:13]}:{run_ts[13:15]}")

    # ── errors.csv ───────────────────────────────────────────────────────────
    if stats.error_rows:
        ep = run_dir / "errors.csv"
        with ep.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "stage", "reason"])
            w.writeheader(); w.writerows(stats.error_rows)

    # ── filtered.csv ─────────────────────────────────────────────────────────
    if stats.filtered_rows:
        fp = run_dir / "filtered.csv"
        with fp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "file", "reason", "accession_number", "sop_instance_uid",
                "radiation", "series_desc", "positioner", "num_frames",
            ])
            w.writeheader(); w.writerows(stats.filtered_rows)

    # ── skipped.csv ──────────────────────────────────────────────────────────
    if stats.skipped_rows:
        sp = run_dir / "skipped.csv"
        with sp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "file", "accession_number", "sop_instance_uid", "dest",
            ])
            w.writeheader(); w.writerows(stats.skipped_rows)

    # ── summary.md ───────────────────────────────────────────────────────────
    md = run_dir / "summary.md"
    with md.open("w", encoding="utf-8") as f:

        f.write("# DICOM Processing Run Summary\n\n")
        f.write(f"**Run timestamp:** {dt}  \n")
        f.write(f"**Elapsed:** {elapsed_sec:.1f} s  \n")
        f.write(f"**Input root:** `{input_root}`  \n")
        f.write(f"**Output root:** `{output_root}`  \n")
        f.write(f"**Workers:** {n_workers}  \n")
        f.write(f"**Min frames threshold:** {min_frames}  \n")
        f.write(f"**Skip existing:** {skip_existing}  \n")
        f.write(f"**Leaf directories scanned:** {leaf_dir_count}  \n\n")
        f.write("---\n\n")

        # Overall counts
        f.write("## Overall Counts\n\n")
        f.write("| Metric | Count |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total DICOM files found       | **{stats.total_found}** |\n")
        f.write(f"| Successfully processed        | **{stats.processed}** |\n")
        f.write(f"| Skipped (already exists)      | **{stats.skipped_done}** |\n")
        f.write(f"| Rejected (eligibility filter) | **{stats.skipped_filter}** |\n")
        f.write(f"| Errors (exceptions)           | **{stats.errors}** |\n\n")

        # Filter breakdown
        f.write("## Eligibility Filter Breakdown\n\n")
        f.write("> Files are evaluated in order. Each file is counted under "
                "the **first** failing criterion only.\n\n")
        f.write("| Reason | Count | Meaning |\n")
        f.write("|--------|-------|---------|\n")
        f.write(f"| `bad_radiation`  | **{stats.bad_radiation}** "
                f"| RadiationSetting ≠ `{REQUIRED_RADIATION_SETTING}` |\n")
        f.write(f"| `bad_series`     | **{stats.bad_series}** "
                f"| SeriesDescription does not contain "
                f"`{'` or `'.join(SERIES_DESCRIPTION_KEYWORDS)}` |\n")
        f.write(f"| `bad_motion`     | **{stats.bad_motion}** "
                f"| PositionerMotion ≠ `{REQUIRED_POSITIONER_MOTION}` |\n")
        f.write(f"| `too_few_frames` | **{stats.too_few_frames}** "
                f"| NumberOfFrames ≤ {min_frames} |\n")
        f.write(f"| `filter_error`   | **{stats.filter_error}** "
                f"| Exception during filter evaluation |\n\n")
        if stats.filtered_rows:
            f.write(f"Full details: `filtered.csv` ({len(stats.filtered_rows)} rows)\n\n")

        # Skipped
        f.write("## Already-Existing Files\n\n")
        if stats.skipped_done == 0:
            f.write("No files were skipped — destination was empty or "
                    "`--skip_existing false` was set.\n\n")
        else:
            f.write(f"{stats.skipped_done} file(s) were skipped because their "
                    f"destination (`AccessionNumber/SOPInstanceUID/`) already "
                    f"contained `metadata.csv` and `frames/`.\n\n")
            f.write(f"Full details: `skipped.csv` ({len(stats.skipped_rows)} rows)\n\n")

        # Errors
        f.write("## Errors\n\n")
        if stats.errors == 0:
            f.write("No errors encountered.\n\n")
        else:
            f.write(f"{stats.errors} error(s) occurred across stages:\n\n")
            # Stage breakdown
            stage_counts: Dict[str, int] = defaultdict(int)
            for row in stats.error_rows:
                stage_counts[row["stage"]] += 1
            f.write("| Stage | Count |\n")
            f.write("|-------|-------|\n")
            for stage, cnt in sorted(stage_counts.items()):
                f.write(f"| `{stage}` | {cnt} |\n")
            f.write(f"\nFull details: `errors.csv` ({len(stats.error_rows)} rows)\n\n")

        f.write("---\n\n")
        f.write("*Report generated by dicom_parallel_processor.py*\n")

    print(f"\n{'='*60}")
    print(f"  Run summary written to: {run_dir}")
    print(f"{'='*60}")


# =========================================================
# Main orchestrator
# =========================================================
def process_root_directory(
    root_dir:      Path,
    output_root:   Path,
    workers:       int,
    min_frames:    int,
    skip_existing: bool,
):
    run_ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_DIR / f"run_{run_ts}"
    t_start = datetime.datetime.now()

    print(f"Input  : {root_dir}")
    print(f"Output : {output_root}")
    print(f"Logs   : {run_dir}")
    print(f"Workers: {workers}  |  min_frames: {min_frames}  "
          f"|  skip_existing: {skip_existing}\n")

    # ── Discover leaf directories ─────────────────────────────────────────────
    print("Discovering leaf directories...")
    leaf_dirs = collect_leaf_dirs(root_dir)
    if not leaf_dirs:
        print("No directories with files found. Exiting.")
        return

    n_workers = max(1, min(workers, len(leaf_dirs)))
    print(f"Found {len(leaf_dirs)} leaf directories — "
          f"dividing across {n_workers} worker(s).\n")

    output_root_str = str(output_root)

    # ── Worker pool ───────────────────────────────────────────────────────────
    # One future per leaf directory so the bar ticks continuously as each
    # directory finishes — not once per large chunk.
    # The worker signature accepts List[str], so we pass a single-element list.
    all_stats:   List[WorkerStats] = []
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
                    [str(d)],          # single-element list — no worker change needed
                    output_root_str,
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
                    processed = agg_processed,
                    filtered  = agg_filtered,
                    skipped   = agg_skipped,
                    errors    = agg_errors,
                )
                pbar.set_description(f"Processing  [{Path(dir_str).name[:30]}]")
                pbar.update(1)

    # ── Merge + report ────────────────────────────────────────────────────────
    merged       = _merge_stats(all_stats)
    elapsed_sec  = (datetime.datetime.now() - t_start).total_seconds()

    # Print summary to terminal
    print(f"\n{'='*60}")
    print(f"  Total found     : {merged.total_found}")
    print(f"  Processed       : {merged.processed}")
    print(f"  Already existed : {merged.skipped_done}")
    print(f"  Filtered out    : {merged.skipped_filter}")
    print(f"    bad_radiation : {merged.bad_radiation}")
    print(f"    bad_series    : {merged.bad_series}")
    print(f"    bad_motion    : {merged.bad_motion}")
    print(f"    too_few_frames: {merged.too_few_frames}")
    print(f"    filter_error  : {merged.filter_error}")
    print(f"  Errors          : {merged.errors}")
    print(f"  Elapsed         : {elapsed_sec:.1f} s")
    print(f"{'='*60}")

    _write_reports(
        run_dir        = run_dir,
        stats          = merged,
        run_ts         = run_ts,
        input_root     = root_dir,
        output_root    = output_root,
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
            "Process DSA DICOM directories — filter, extract frames + metadata. "
            "Output: output_root / AccessionNumber / SOPInstanceUID /"
        )
    )
    parser.add_argument("--input_root",  type=Path, default=Path(INPUT_ROOT))
    parser.add_argument("--output_root", type=Path, default=Path(OUTPUT_ROOT))
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
        help="Skip DICOMs whose dest already has frames/ + metadata.csv (default: True)",
    )

    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    process_root_directory(
        root_dir      = args.input_root,
        output_root   = args.output_root,
        workers       = args.workers,
        min_frames    = args.min_frames,
        skip_existing = args.skip_existing,
    )