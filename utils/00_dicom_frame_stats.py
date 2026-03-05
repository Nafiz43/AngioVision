#!/usr/bin/env python3
from pathlib import Path
import csv
from tqdm import tqdm
from urllib.parse import quote

def path_to_file_url(p: Path) -> str:
    """
    Create a file:// URL that many terminals/IDEs can open.
    Note: Some remote HPC terminals won't "click-open" these, but the URL is still correct.
    """
    # Quote spaces and special chars safely
    return "file://" + quote(str(p))

def count_files_in_dir(d: Path) -> int:
    """Count only files directly inside d (not recursive)."""
    try:
        return sum(1 for x in d.iterdir() if x.is_file())
    except Exception:
        return 0

def generate_frame_stats(root_dir, output_csv):
    root_dir = Path(root_dir)
    output_csv = Path(output_csv)

    # 1) Find ALL frames dirs under root_dir
    print(f"Scanning for 'frames' directories under: {root_dir}")
    frames_dirs = [p for p in root_dir.rglob("frames") if p.is_dir()]

    if not frames_dirs:
        print("No 'frames' directories found.")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "outer_dir_name",
                "inner_dir_name",
                "number_of_frames",
                "frames_dir_path",
                "frames_dir_url"
            ])
        return

    rows = []
    for frames_dir in tqdm(frames_dirs, desc="Counting frames", unit="frames_dir"):
        # frames_dir is .../<outer>/<inner>/frames (in your processed layout)
        inner_dir = frames_dir.parent
        outer_dir = inner_dir.parent

        outer_dir_name = outer_dir.name if outer_dir else "NA"
        inner_dir_name = inner_dir.name if inner_dir else "NA"

        frame_count = count_files_in_dir(frames_dir)

        rows.append([
            outer_dir_name,
            inner_dir_name,
            frame_count,
            str(frames_dir),
            path_to_file_url(frames_dir),
        ])

    # 2) Sort by frame count descending
    rows.sort(key=lambda x: x[2], reverse=True)

    # 3) Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "outer_dir_name",
            "inner_dir_name",
            "number_of_frames",
            "frames_dir_path",
            "frames_dir_url"
        ])
        writer.writerows(rows)

    print(f"Done. Wrote: {output_csv}")

if __name__ == "__main__":
    generate_frame_stats(
        root_dir="/data/Deep_Angiography/DICOM_Sequence_Processed",
        output_csv="/data/Deep_Angiography/DICOM-metadata-stats/frame_statistics.csv"
    )