#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd

BASE_DIR = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
OUTPUT_CSV = Path("/data/Deep_Angiography/DICOM-metadata-stats/radiation_GR_without_DSA.csv")


def read_metadata_csv(csv_path: Path) -> dict | None:
    """
    Supports both:
      1) Wide format (columns include RadiationSetting/SeriesDescription)
      2) Key/value format (Information, Value)
    Returns a dict with extracted metadata (at least those keys if present).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Wide format
    if ("RadiationSetting" in df.columns) or ("SeriesDescription" in df.columns):
        if len(df) == 0:
            return {}
        return df.iloc[0].to_dict()

    # Key/value format
    if ("Information" in df.columns) and ("Value" in df.columns):
        try:
            # Ensure strings (avoid NaN issues)
            info = df["Information"].astype(str)
            val = df["Value"].astype(str)
            return dict(zip(info, val))
        except Exception:
            return None

    return None


def check_one_metadata(csv_path_str: str) -> dict | None:
    """
    Worker function (must be top-level for ProcessPoolExecutor).
    Returns a dict row if violating, else None.
    """
    csv_path = Path(csv_path_str)
    meta = read_metadata_csv(csv_path)
    if meta is None:
        return None

    radiation = str(meta.get("RadiationSetting", "")).strip()
    series_desc = str(meta.get("SeriesDescription", "")).strip()

    # Condition: RadiationSetting is GR AND SeriesDescription does NOT contain "DSA" (case-insensitive)
    if radiation.upper() == "GR" and "DSA" not in series_desc.upper():
        return {
            "metadata_path": str(csv_path),
            "RadiationSetting": radiation,
            "SeriesDescription": series_desc,
        }

    return None


def main():
    metadata_files = list(BASE_DIR.rglob("metadata.csv"))
    print(f"Found {len(metadata_files)} metadata.csv files")

    # Tune this if your filesystem is slow / shared:
    # - For heavy I/O, too many workers can slow things down.
    max_workers = min(32, (os.cpu_count() or 4))
    print(f"Using max_workers={max_workers}")

    violating_rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(check_one_metadata, str(p)) for p in metadata_files]

        for fut in as_completed(futures):
            try:
                row = fut.result()
                if row is not None:
                    violating_rows.append(row)
            except Exception:
                # If one file is malformed, keep going
                continue

    if violating_rows:
        out_df = pd.DataFrame(violating_rows)
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nFound {len(violating_rows)} violating cases.")
        print(f"Saved to: {OUTPUT_CSV}")
    else:
        print("\nNo violations found.")


if __name__ == "__main__":
    main()