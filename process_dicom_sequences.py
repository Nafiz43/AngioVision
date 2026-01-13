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
KEYWORDS = ["DSA", "Single"]


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


def save_frames(ds, frames_dir: Path, base_name: str):
    pixel_array = ds.pixel_array

    # Single-frame
    if pixel_array.ndim == 2:
        img = Image.fromarray(pixel_array)
        img.save(frames_dir / f"{base_name}_frame_0001.{FRAME_FORMAT}")
        return 1

    # Multi-frame
    frame_count = pixel_array.shape[0]
    for i in range(frame_count):
        img = Image.fromarray(pixel_array[i])
        img.save(
            frames_dir / f"{base_name}_frame_{i+1:04d}.{FRAME_FORMAT}"
        )
    return frame_count


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
        frame_count = save_frames(ds, frames_dir, base_name)

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
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)

    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    process_root_directory(args.input_root, args.output_root)
