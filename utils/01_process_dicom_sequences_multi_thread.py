#!/usr/bin/env python3
"""
Parallel DICOM processing (multi-process):

- Pre-scan for DICOM files under keyword-matched directories
- Process each file in a ProcessPoolExecutor:
    * Extract frames -> PNG
    * Extract metadata -> metadata.csv
- Single tqdm bar in main process

Tips:
- Use --workers to tune parallelism (default: CPU count)
- Use --chunksize to reduce IPC overhead when you have tons of files
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from pydicom.multival import MultiValue
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


# ----------------------------
# Configuration
# ----------------------------
FRAME_FORMAT = "png"
KEYWORDS: List[str] = []   # if empty, matches all dirs
NA_VALUE = "NA"
MAX_VALUE_CHARS = 2000

SKIP_KEYWORDS = {
    "PixelData", "WaveformData", "OverlayData",
    "EncapsulatedDocument", "CurveData", "AudioSampleData"
}
SKIP_TAGS = {(0x7FE0, 0x0010)}  # PixelData


# ----------------------------
# Utilities
# ----------------------------
def contains_required_keywords(dir_name: str) -> bool:
    if not KEYWORDS:
        return True
    name = dir_name.upper()
    return all(k.upper() in name for k in KEYWORDS)


def is_probably_dicom(path: Path) -> bool:
    """
    Fast-ish DICOM check:
    - Many DICOMs have 'DICM' at byte offset 128
    - If not, we can still have valid DICOM without preamble, so fallback to pydicom.
    """
    try:
        with open(path, "rb") as f:
            f.seek(128)
            magic = f.read(4)
        if magic == b"DICM":
            return True
    except Exception:
        return False

    # Fallback: attempt reading minimal header
    try:
        pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def is_nullish(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
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
        if s == "":
            return NA_VALUE
    else:
        s = safe_str(value).strip()
        if s == "":
            return NA_VALUE

    if len(s) > MAX_VALUE_CHARS:
        s = s[:MAX_VALUE_CHARS] + "...<truncated>"
    return s


def extract_metadata_pairs(ds):
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

        key = elem.keyword if elem.keyword else str(elem.tag)
        value = normalize_value(elem.value)
        rows.append({"Information": key, "Value": value})
    return rows


def sanitize_dirname(name: str, max_len: int = 150) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._-")
    if not name:
        name = "dicom"
    return name[:max_len]


# ----------------------------
# Image conversion
# ----------------------------
def to_uint8_windowed(arr: np.ndarray, ds) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)

    if isinstance(center, MultiValue):
        center = center[0]
    if isinstance(width, MultiValue):
        width = width[0]

    if center is not None and width is not None and width > 0:
        lo = center - width / 2
        hi = center + width / 2
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    else:
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

    if "MONOCHROME1" in str(getattr(ds, "PhotometricInterpretation", "")).upper():
        arr = 1.0 - arr

    return (arr * 255).astype(np.uint8)


def save_frames(ds, frames_dir: Path, base_name: str) -> int:
    px = ds.pixel_array

    def save_gray(img, idx):
        Image.fromarray(img, mode="L").save(
            frames_dir / f"{base_name}_frame_{idx:04d}.{FRAME_FORMAT}"
        )

    if px.ndim == 2:
        save_gray(to_uint8_windowed(px, ds), 1)
        return 1

    if px.ndim == 3:
        for i in range(px.shape[0]):
            save_gray(to_uint8_windowed(px[i], ds), i + 1)
        return px.shape[0]

    raise ValueError(f"Unexpected pixel_array shape: {px.shape}")


# ----------------------------
# Planning pass
# ----------------------------
def iter_matched_dirs(root_dir: Path):
    for dirpath, _, _ in os.walk(root_dir):
        current = Path(dirpath)
        if contains_required_keywords(current.name):
            yield current


def collect_dicom_files(root_dir: Path) -> List[Tuple[str, str]]:
    """
    Returns list of (matched_dir_str, dicom_file_str)
    """
    pairs: List[Tuple[str, str]] = []
    for matched_dir in iter_matched_dirs(root_dir):
        for f in matched_dir.rglob("*"):
            if f.is_file() and is_probably_dicom(f):
                pairs.append((str(matched_dir), str(f)))
    return pairs


# ----------------------------
# Worker: single file
# ----------------------------
def process_one_dicom_worker(matched_dir_str: str, file_str: str, output_root_str: str, skip_existing: bool):
    matched_dir = Path(matched_dir_str)
    file = Path(file_str)
    output_root = Path(output_root_str)

    # Output paths
    study_output_dir = output_root / matched_dir.name
    dicom_folder_name = sanitize_dirname(file.stem)
    per_dicom_dir = study_output_dir / dicom_folder_name
    frames_dir = per_dicom_dir / "frames"
    metadata_csv = per_dicom_dir / "metadata.csv"

    if skip_existing and metadata_csv.exists() and frames_dir.exists():
        # Fast skip: assume already processed
        return ("skipped", matched_dir.name, file.name, "")

    try:
        study_output_dir.mkdir(parents=True, exist_ok=True)
        per_dicom_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        ds_meta = pydicom.dcmread(file, stop_before_pixels=True, force=True)
        ds_full = pydicom.dcmread(file, force=True)

        base_name = sanitize_dirname(file.stem)

        try:
            frame_count = save_frames(ds_full, frames_dir, base_name)
        except Exception:
            frame_count = NA_VALUE

        metadata_rows = extract_metadata_pairs(ds_meta)
        metadata_rows.extend([
            {"Information": "source_file", "Value": safe_str(file.name)},
            {"Information": "source_path", "Value": safe_str(str(file))},
            {"Information": "frame_count", "Value": safe_str(frame_count)},
        ])

        df = pd.DataFrame(metadata_rows)
        df["Information"] = df["Information"].fillna(NA_VALUE).map(safe_str)
        df["Value"] = df["Value"].fillna(NA_VALUE).map(safe_str)
        df.to_csv(metadata_csv, index=False, encoding="utf-8")

        return ("ok", matched_dir.name, file.name, "")

    except Exception as e:
        return ("error", matched_dir.name, file.name, f"{type(e).__name__}: {e}")


# ----------------------------
# Root Processing (parallel)
# ----------------------------
def process_root_directory(root_dir: Path, output_root: Path, workers: int, chunksize: int, skip_existing: bool):
    print("Scanning for DICOM files (to initialize progress bar)...")
    targets = collect_dicom_files(root_dir)

    if not targets:
        print("No DICOM files found under keyword-matched directories.")
        return

    # Make sure spawn is used safely in some environments
    # (Linux defaults to fork, which is usually fine; spawn is safer but slower to start.)
    # We'll keep the default; just ensure this function is under __main__.

    total = len(targets)
    output_root_str = str(output_root)

    with tqdm(total=total, unit="dicom", desc="Processing DICOMs") as pbar:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            # Submit in batches to reduce overhead
            futures = []
            for matched_dir_str, file_str in targets:
                futures.append(ex.submit(process_one_dicom_worker, matched_dir_str, file_str, output_root_str, skip_existing))

                # Optionally throttle submissions a bit to keep memory reasonable
                if len(futures) >= workers * max(4, chunksize):
                    for fut in as_completed(futures):
                        status, mdir, fname, err = fut.result()
                        if status == "error":
                            tqdm.write(f"[ERROR] {mdir}/{fname}: {err}")
                        pbar.set_postfix_str(f"{mdir}/{fname}"[:80])
                        pbar.update(1)
                    futures = []

            # Drain remaining
            for fut in as_completed(futures):
                status, mdir, fname, err = fut.result()
                if status == "error":
                    tqdm.write(f"[ERROR] {mdir}/{fname}: {err}")
                pbar.set_postfix_str(f"{mdir}/{fname}"[:80])
                pbar.update(1)


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process DSA DICOM directories and extract frames + metadata (parallel)"
    )
    parser.add_argument("--input_root", type=Path, default=Path("/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM"))
    parser.add_argument("--output_root", type=Path, default=Path("/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM_Sequence_Processed"))

    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) - 1),
        help="Number of worker processes (default: cpu_count-1)"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=32,
        help="Submission batch size multiplier knob (bigger = less overhead; default 32)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip DICOMs that already have frames/ + metadata.csv"
    )

    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    process_root_directory(
        root_dir=args.input_root,
        output_root=args.output_root,
        workers=args.workers,
        chunksize=args.chunksize,
        skip_existing=args.skip_existing,
    )
