#!/usr/bin/env python3
"""
Batch-process ALL DICOM files under input_root (no keyword matching), and for each DICOM:

1) Extract image frames from pixel data and save as PNG (8-bit grayscale) to:
   <output_root>/<relative_parent>/<dicom_name>/frames/

2) Export per-file metadata (excluding sequences and large/binary fields like PixelData)
   to:
   <output_root>/<relative_parent>/<dicom_name>/metadata.csv

Notes
-----
- Even if your dataset is currently flat (no nested folders), this still works.
  In that case, <relative_parent> is just the one folder containing the DICOMs.
- dicom_name is based ONLY on the original DICOM filename (stem), sanitized.
- Includes a tqdm progress bar with ETA.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.misc import is_dicom as fast_is_dicom
from pydicom.multival import MultiValue
from PIL import Image
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------
FRAME_FORMAT = "png"

NA_VALUE = "NA"
MAX_VALUE_CHARS = 2000

# Never export binary / huge fields
SKIP_KEYWORDS = {
    "PixelData", "WaveformData", "OverlayData",
    "EncapsulatedDocument", "CurveData", "AudioSampleData",
}
SKIP_TAGS = {(0x7FE0, 0x0010)}  # PixelData


# ----------------------------
# Small helpers
# ----------------------------
def sanitize_dirname(name: str, max_len: int = 150) -> str:
    """Filesystem-safe directory name."""
    name = (name or "").strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._-")
    return (name or "dicom")[:max_len]


def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


def is_dicom_file(path: Path) -> bool:
    """
    DICOM detection:
    - Try pydicom.misc.is_dicom() (preamble check).
    - If that fails, try a lightweight read with force=True (covers no-preamble DICOMs).
    """
    try:
        if fast_is_dicom(str(path)):
            return True
    except Exception:
        pass

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
        if not s:
            return NA_VALUE
    else:
        s = safe_str(value).strip()
        if not s:
            return NA_VALUE

    if len(s) > MAX_VALUE_CHARS:
        s = s[:MAX_VALUE_CHARS] + "...<truncated>"

    return s


def extract_metadata_pairs(ds) -> list[dict]:
    """Return [{Information, Value}, ...] excluding sequences and big/binary fields."""
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
        rows.append({"Information": key, "Value": normalize_value(elem.value)})

    return rows


def relative_parent_dir(input_root: Path, file_path: Path) -> Path:
    """
    Returns the relative parent folder of file_path under input_root.
    If your DICOMs are flat under one folder, this is that folder name.
    """
    try:
        rel_parent = file_path.parent.relative_to(input_root)
        if not rel_parent.parts:
            return Path(".")
        return Path(*[sanitize_dirname(p) for p in rel_parent.parts])
    except Exception:
        return Path("_unknown_path")


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

    def save_gray(img: np.ndarray, idx: int):
        Image.fromarray(img, mode="L").save(
            frames_dir / f"{base_name}_frame_{idx:04d}.{FRAME_FORMAT}"
        )

    if px.ndim == 2:
        save_gray(to_uint8_windowed(px, ds), 1)
        return 1

    if px.ndim == 3:
        for i in range(px.shape[0]):
            save_gray(to_uint8_windowed(px[i], ds), i + 1)
        return int(px.shape[0])

    raise ValueError(f"Unexpected pixel_array shape: {px.shape}")


# ----------------------------
# Scanning + processing
# ----------------------------
def collect_dicom_files(input_root: Path) -> list[Path]:
    targets: list[Path] = []
    for f in input_root.rglob("*"):
        if f.is_file() and is_dicom_file(f):
            targets.append(f)
    return targets


def make_dicom_name(file_path: Path) -> str:
    """
    Output folder name based ONLY on original DICOM file name (stem).
    No UID suffixes.
    """
    return sanitize_dirname(file_path.stem)


def process_one_dicom(input_root: Path, file_path: Path, output_root: Path, skip_existing: bool) -> None:
    # Read metadata first
    ds_meta = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)

    rel_parent = relative_parent_dir(input_root, file_path)   # "." or folder name
    out_parent = output_root / rel_parent
    out_parent.mkdir(parents=True, exist_ok=True)

    dicom_name = make_dicom_name(file_path)
    per_dicom_dir = out_parent / dicom_name
    metadata_csv = per_dicom_dir / "metadata.csv"

    if skip_existing and metadata_csv.exists():
        return

    frames_dir = per_dicom_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames (best-effort)
    frame_count = NA_VALUE
    try:
        ds_full = pydicom.dcmread(file_path, force=True)
        frame_count = save_frames(ds_full, frames_dir, sanitize_dirname(file_path.stem))
    except Exception:
        frame_count = NA_VALUE

    # Metadata CSV
    rows = extract_metadata_pairs(ds_meta)
    rows.extend([
        {"Information": "source_file", "Value": safe_str(file_path.name)},
        {"Information": "source_path", "Value": safe_str(str(file_path))},
        {"Information": "relative_parent", "Value": safe_str(str(rel_parent))},
        {"Information": "output_dicom_name", "Value": safe_str(dicom_name)},
        {"Information": "frame_count", "Value": safe_str(frame_count)},
    ])

    df = pd.DataFrame(rows)
    df["Information"] = df["Information"].fillna(NA_VALUE).map(safe_str)
    df["Value"] = df["Value"].fillna(NA_VALUE).map(safe_str)

    per_dicom_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_csv, index=False, encoding="utf-8")


def process_root_directory(input_root: Path, output_root: Path, skip_existing: bool = False) -> None:
    print(f"Scanning for DICOM files under: {input_root}")
    targets = collect_dicom_files(input_root)

    if not targets:
        print("No DICOM files found.")
        return

    with tqdm(total=len(targets), unit="dicom", desc="Processing DICOMs") as pbar:
        for f in targets:
            try:
                rel = str(f.relative_to(input_root))
            except Exception:
                rel = str(f)
            pbar.set_postfix_str(rel[:80])

            try:
                process_one_dicom(input_root, f, output_root, skip_existing)
            except Exception as e:
                tqdm.write(f"[ERROR] {f}: {e}")

            pbar.update(1)


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process ALL DICOM files under input_root and extract frames + metadata (with progress bar)."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM"),
        help="Root directory to recursively scan for DICOM files.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed"),
        help="Root directory where outputs will be written.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a DICOM if its metadata.csv already exists in the output.",
    )

    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    process_root_directory(args.input_root, args.output_root, skip_existing=args.skip_existing)
