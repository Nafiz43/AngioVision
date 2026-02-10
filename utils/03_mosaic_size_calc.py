#!/usr/bin/env python3
"""
mosaic_size_calculator.py

Recursively scans a root directory, finds all `mosaic.png` files inside
inner directories, and records their image dimensions (width, height).

Outputs a CSV with:
- outer_dir
- inner_dir
- mosaic_path
- width
- height
"""

import argparse
import csv
from pathlib import Path
from PIL import Image
DEFAULT_FALLBACK_ROOT = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")


def find_mosaic_sizes(root_dir: Path, output_csv: Path) -> None:
    rows = []

    for mosaic_path in root_dir.rglob("mosaic.png"):
        try:
            with Image.open(mosaic_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"[WARN] Failed to read {mosaic_path}: {e}")
            continue

        # Infer outer + inner directory names
        # Example:
        # DICOM_Sequence_Processed/01_5sDSA/<INNER_DIR>/mosaic.png
        parts = mosaic_path.parts
        outer_dir = parts[-3] if len(parts) >= 3 else "UNKNOWN"
        inner_dir = parts[-2]

        rows.append([
            outer_dir,
            inner_dir,
            str(mosaic_path),
            width,
            height,
        ])

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "outer_dir",
            "inner_dir",
            "mosaic_path",
            "width",
            "height",
        ])
        writer.writerows(rows)

    print(f"[DONE] Found {len(rows)} mosaic files")
    print(f"[DONE] Output written to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate mosaic.png image dimensions recursively"
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        help="Root directory to scan (falls back if missing)",
    )
    parser.add_argument(
        "--output-csv",
        default=Path("mosaic_sizes.csv"),
        type=Path,
        help="Output CSV file path",
    )

    args = parser.parse_args()

    # Resolve root directory with fallback
    if args.root_dir and args.root_dir.exists():
        root_dir = args.root_dir
    else:
        print(
            f"[WARN] Provided root dir missing. "
            f"Falling back to {DEFAULT_FALLBACK_ROOT}"
        )
        root_dir = DEFAULT_FALLBACK_ROOT

    if not root_dir.exists():
        raise FileNotFoundError(
            f"Fallback root directory does not exist: {root_dir}"
        )

    find_mosaic_sizes(root_dir, args.output_csv)


if __name__ == "__main__":
    main()
