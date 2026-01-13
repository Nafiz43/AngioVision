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
FRAME_FORMAT = "png"   # png or jpg
KEYWORDS = ["DSA"]

# ----------------------------
# Utilities
# ----------------------------
def is_dicom_file(path: Path) -> bool:
    try:
        pydicom.dcmread(path, stop_before_pixels=True)
        return True
    except Exception:
        return False

def normalize_value(value):
    """Make DICOM values CSV-safe."""
    if isinstance(value, MultiValue):
        return ";".join(map(str, value))
    return str(value)

def extract_metadata(ds):
    metadata = {}
    for elem in ds:
        if elem.VR != "SQ":
            metadata[elem.keyword or elem.tag.__str__()] = normalize_value(elem.value)
    return metadata

# ----------------------------
# Windowing / conversion
# ----------------------------
def to_uint8_windowed(arr: np.ndarray, ds, wl_override=None, ww_override=None) -> np.ndarray:
    """
    Convert pixel data to 8-bit grayscale with:
      - rescale slope/intercept
      - window center/width (supports MultiValue)
      - fallback min-max normalization
      - MONOCHROME1 inversion
    """
    arr = arr.astype(np.float32, copy=False)

    # Rescale slope/intercept (if present)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or inter != 0.0:
        arr = arr * slope + inter

    # Choose window params
    if wl_override is not None and ww_override is not None and ww_override > 0:
        center, width = float(wl_override), float(ww_override)
    else:
        center = getattr(ds, "WindowCenter", None)
        width = getattr(ds, "WindowWidth", None)

        # MultiValue handling
        if isinstance(center, (MultiValue, list, tuple)):
            center = center[0] if len(center) > 0 else None
        if isinstance(width, (MultiValue, list, tuple)):
            width = width[0] if len(width) > 0 else None

        center = float(center) if center is not None else None
        width = float(width) if width is not None else None

    # Apply windowing if available; else min-max normalize
    if center is not None and width is not None and width > 0:
        lo, hi = center - width / 2.0, center + width / 2.0
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    else:
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            arr = np.zeros_like(arr, dtype=np.float32)
        else:
            arr = (arr - mn) / (mx - mn)

    # MONOCHROME1 inversion
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if "MONOCHROME1" in photometric:
        arr = 1.0 - arr

    return (arr * 255.0).clip(0, 255).astype(np.uint8)

def save_frames(ds, frames_dir: Path, base_name: str, wl_override=None, ww_override=None):
    px = ds.pixel_array
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()

    def save_gray(frame2d_u8, idx):
        img = Image.fromarray(frame2d_u8, mode="L")
        img.save(frames_dir / f"{base_name}_frame_{idx:04d}.{FRAME_FORMAT}")

    # RGB handling (rare for DSA, but safe)
    if "RGB" in photometric:
        # px could be (H,W,3) or (F,H,W,3)
        if px.ndim == 3 and px.shape[-1] == 3:
            # Convert RGB->L via PIL
            img = Image.fromarray(px.astype(np.uint8), mode="RGB").convert("L")
            img.save(frames_dir / f"{base_name}_frame_0001.{FRAME_FORMAT}")
            return 1
        elif px.ndim == 4 and px.shape[-1] == 3:
            for i in range(px.shape[0]):
                img = Image.fromarray(px[i].astype(np.uint8), mode="RGB").convert("L")
                img.save(frames_dir / f"{base_name}_frame_{i+1:04d}.{FRAME_FORMAT}")
            return px.shape[0]
        # fallback to monochrome path below if unexpected

    # Monochrome single-frame
    if px.ndim == 2:
        out = to_uint8_windowed(px, ds, wl_override, ww_override)
        save_gray(out, 1)
        return 1

    # Monochrome multi-frame: (F,H,W)
    if px.ndim == 3:
        frame_count = px.shape[0]
        for i in range(frame_count):
            out = to_uint8_windowed(px[i], ds, wl_override, ww_override)
            save_gray(out, i + 1)
        return frame_count

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
        if not file.is_file():
            continue
        if not is_dicom_file(file):
            continue

        try:
            ds = pydicom.dcmread(file)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        base_name = file.stem
        try:
            frame_count = save_frames(ds, frames_dir, base_name)
        except Exception as e:
            print(f"Failed to extract frames from {file}: {e}")
            continue

        meta = extract_metadata(ds)
        meta["source_file"] = file.name
        meta["frame_count"] = frame_count
        metadata_rows.append(meta)

    if metadata_rows:
        df = pd.DataFrame(metadata_rows)
        df.to_csv(output_dir / "metadata.csv", index=False)

def contains_required_keywords(dir_name: str) -> bool:
    name_upper = dir_name.upper()
    return all(k.upper() in name_upper for k in KEYWORDS)

def process_root_directory(root_dir: Path, output_root: Path):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        current_dir = Path(dirpath)
        if contains_required_keywords(current_dir.name):
            print(f"Processing: {current_dir}")
            process_dicom_directory(current_dir, output_root)

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process DSA Single DICOM directories")
    parser.add_argument("--input_root", type=Path,
                        default=Path("/data/Deep_Angiography/DICOM"),
                        help="Root directory containing input DICOM files")
    parser.add_argument("--output_root", type=Path,
                        default=Path("/data/Deep_Angiography/DICOM_Sequence_Processed"),
                        help="Root directory for processed outputs")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    process_root_directory(args.input_root, args.output_root)
