#!/usr/bin/env python3
import os
from pathlib import Path
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from pydicom.multival import MultiValue

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
# Core Processing
# ----------------------------
def process_dicom_directory(dicom_dir: Path, output_root: Path):
    output_dir = output_root / dicom_dir.name
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    for file in dicom_dir.rglob("*"):
        if not file.is_file() or not is_dicom_file(file):
            continue

        try:
            ds_meta = pydicom.dcmread(file, stop_before_pixels=True, force=True)
            ds_full = pydicom.dcmread(file, force=True)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        base_name = file.stem

        try:
            frame_count = save_frames(ds_full, frames_dir, base_name)
        except Exception as e:
            print(f"Frame extraction failed {file}: {e}")
            frame_count = NA_VALUE

        rows = extract_metadata_pairs(ds_meta)

        rows.extend([
            {"Information": "source_file", "Value": safe_str(file.name)},
            {"Information": "frame_count", "Value": safe_str(frame_count)},
        ])

        metadata_rows.extend(rows)

    if not metadata_rows:
        return

    df = pd.DataFrame(metadata_rows)

    # absolutely no nulls
    df["Information"] = df["Information"].fillna(NA_VALUE).map(safe_str)
    df["Value"] = df["Value"].fillna(NA_VALUE).map(safe_str)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8")

def contains_required_keywords(dir_name: str) -> bool:
    name = dir_name.upper()
    return all(k.upper() in name for k in KEYWORDS)

def process_root_directory(root_dir: Path, output_root: Path):
    for dirpath, _, _ in os.walk(root_dir):
        current = Path(dirpath)
        if contains_required_keywords(current.name):
            print(f"Processing: {current}")
            process_dicom_directory(current, output_root)

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process DSA DICOM directories and extract frames + metadata"
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
