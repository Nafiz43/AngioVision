#!/usr/bin/env python3

from pathlib import Path

BASE_DIR = Path("/data/Deep_Angiography/DICOM_Sequence_Processed-Only-DSA-Dir")

def count_metadata_files(base_dir: Path):
    """
    Recursively count all files named 'metadata.csv'
    """
    count = 0

    for file in base_dir.rglob("metadata.csv"):
        if file.is_file():
            count += 1

    return count


if __name__ == "__main__":
    total_metadata = count_metadata_files(BASE_DIR)

    print(f"Base directory: {BASE_DIR}")
    print(f"Total metadata.csv files found: {total_metadata}")