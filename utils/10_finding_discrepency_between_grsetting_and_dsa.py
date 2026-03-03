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
            info = df["Information"].astype(str)
            val = df["Value"].astype(str)
            return dict(zip(info, val))
        except Exception:
            return None

    return None


def check_one_metadata(csv_path_str: str) -> dict | None:
    """
    Worker function.
    Returns dict with flags for counting + optional violation row.
    """
    csv_path = Path(csv_path_str)
    meta = read_metadata_csv(csv_path)
    if meta is None:
        return None

    radiation = str(meta.get("RadiationSetting", "")).strip()
    series_desc = str(meta.get("SeriesDescription", "")).strip()

    is_gr = radiation.upper() == "GR"
    has_dsa = "DSA" in series_desc.upper()
    is_violation = is_gr and not has_dsa

    return {
        "metadata_path": str(csv_path),
        "RadiationSetting": radiation,
        "SeriesDescription": series_desc,
        "is_gr": is_gr,
        "has_dsa": has_dsa,
        "is_violation": is_violation,
    }


def main():
    metadata_files = list(BASE_DIR.rglob("metadata.csv"))
    print(f"Found {len(metadata_files)} metadata.csv files")

    max_workers = min(32, (os.cpu_count() or 4))
    print(f"Using max_workers={max_workers}")

    violating_rows = []
    total_gr = 0
    total_dsa = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(check_one_metadata, str(p)) for p in metadata_files]

        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result is None:
                    continue

                # Global counters
                if result["is_gr"]:
                    total_gr += 1
                if result["has_dsa"]:
                    total_dsa += 1

                # Violations
                if result["is_violation"]:
                    violating_rows.append({
                        "metadata_path": result["metadata_path"],
                        "RadiationSetting": result["RadiationSetting"],
                        "SeriesDescription": result["SeriesDescription"],
                    })

            except Exception:
                continue

    print("\n========== SUMMARY ==========")
    print(f"Total files with RadiationSetting == 'GR': {total_gr}")
    print(f"Total files with SeriesDescription containing 'DSA': {total_dsa}")
    print(f"Total violating cases (GR but no DSA): {len(violating_rows)}")

    if violating_rows:
        out_df = pd.DataFrame(violating_rows)
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved violating cases to: {OUTPUT_CSV}")
    else:
        print("\nNo violations found.")


if __name__ == "__main__":
    main()