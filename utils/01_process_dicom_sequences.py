#!/usr/bin/env python3
"""
Purpose
-------
Batch-process DICOM studies under an input root, locate directories whose *folder name*
contains required keyword(s) (default: "DSA"), and for each DICOM file found inside:

1) Extract image frames from pixel data and save them as PNGs (8-bit grayscale) to:
     <output_root>/<matched_dir_name>/<dicom_filename_sanitized>/frames/

2) Export per-file metadata (excluding sequences and large/binary fields like PixelData)
   as a simple key/value CSV:
     <output_root>/<matched_dir_name>/<dicom_filename_sanitized>/metadata.csv

Adds:
- A tqdm progress bar showing percent complete + ETA over all DICOM files found.
"""

import os
from pathlib import Path
import re
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from pydicom.multival import MultiValue
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------
FRAME_FORMAT = "png"
KEYWORDS = ["DSA"]

NA_VALUE = "NA"
MAX_VALUE_CHARS = 2000

# Never export binary / huge fields
SKIP_KEYWORDS = {
    "PixelData", "WaveformData", "OverlayData",
    "EncapsulatedDocument", "CurveData", "AudioSampleData"
}
SKIP_TAGS = {(0x7FE0, 0x0010)}  # PixelData

# ----------------------------
# Utilities
# ----------------------------
def is_dicom_file(path: Path) -> bool:
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
    """Force safe UTF-8 string for CSV."""
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
    """
    Returns a list of dicts:
    {Information, Value}
    Only non-null metadata included.
    """
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

        rows.append({
            "Information": key,
            "Value": value
        })

    return rows

def sanitize_dirname(name: str, max_len: int = 150) -> str:
    """
    Make a filesystem-safe directory name (keeps letters, numbers, dot, dash, underscore).
    """
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._-")
    if not name:
        name = "dicom"
    return name[:max_len]

def contains_required_keywords(dir_name: str) -> bool:
    name = dir_name.upper()
    return all(k.upper() in name for k in KEYWORDS)

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
# Planning pass (for progress bar)
# ----------------------------
def iter_matched_dirs(root_dir: Path):
    """Yield directories whose *name* contains required KEYWORDS."""
    for dirpath, _, _ in os.walk(root_dir):
        current = Path(dirpath)
        if contains_required_keywords(current.name):
            yield current

def collect_dicom_files(root_dir: Path):
    """
    Pre-scan to collect all DICOM files we will process.
    Returns a list of (matched_dir, dicom_file_path).
    """
    pairs = []
    for matched_dir in iter_matched_dirs(root_dir):
        for f in matched_dir.rglob("*"):
            if f.is_file() and is_dicom_file(f):
                pairs.append((matched_dir, f))
    return pairs

# ----------------------------
# Core Processing (single file)
# ----------------------------
def process_one_dicom(matched_dir: Path, file: Path, output_root: Path):
    """
    Process ONE DICOM file that lives somewhere under matched_dir.
    Outputs under: output_root/<matched_dir.name>/<safe(file.stem)>/{frames,metadata.csv}
    """
    study_output_dir = output_root / matched_dir.name
    study_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds_meta = pydicom.dcmread(file, stop_before_pixels=True, force=True)
        ds_full = pydicom.dcmread(file, force=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read {file}: {e}") from e

    dicom_folder_name = sanitize_dirname(file.stem)
    per_dicom_dir = study_output_dir / dicom_folder_name
    frames_dir = per_dicom_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    base_name = sanitize_dirname(file.stem)

    try:
        frame_count = save_frames(ds_full, frames_dir, base_name)
    except Exception as e:
        # Keep going, but record NA
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

    per_dicom_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(per_dicom_dir / "metadata.csv", index=False, encoding="utf-8")

# ----------------------------
# Root Processing (with loader)
# ----------------------------
def process_root_directory(root_dir: Path, output_root: Path):
    # 1) Pre-scan so we can show percent + ETA.
    print("Scanning for DICOM files (to initialize progress bar)...")
    targets = collect_dicom_files(root_dir)

    if not targets:
        print("No DICOM files found under keyword-matched directories.")
        return

    # 2) Process with progress bar.
    with tqdm(total=len(targets), unit="dicom", desc="Processing DICOMs") as pbar:
        for matched_dir, file in targets:
            pbar.set_postfix_str(f"{matched_dir.name}/{file.name}"[:80])

            try:
                process_one_dicom(matched_dir, file, output_root)
            except Exception as e:
                # Don't crash the run; show error and continue.
                tqdm.write(str(e))

            pbar.update(1)

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process DSA DICOM directories and extract frames + metadata (with progress bar)"
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("/data/Deep_Angiography/DICOM")
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
    )

    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    process_root_directory(args.input_root, args.output_root)
