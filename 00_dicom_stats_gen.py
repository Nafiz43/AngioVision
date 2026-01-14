from pathlib import Path
import csv

def generate_frame_stats(root_dir, output_csv):
    root_dir = Path(root_dir)

    rows = []

    for dicom_dir in root_dir.iterdir():
        if not dicom_dir.is_dir():
            continue

        frames_dir = dicom_dir / "frames"

        if frames_dir.exists() and frames_dir.is_dir():
            frame_count = sum(
                1 for f in frames_dir.iterdir() if f.is_file()
            )
        else:
            frame_count = 0  # frames folder missing

        rows.append([dicom_dir.name, frame_count])

    # write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dir_name", "number_of_frames"])
        writer.writerows(rows)


# -------- example usage --------
generate_frame_stats(
    root_dir="/data/Deep_Angiography/DICOM_Sequence_Processed",
    output_csv="frame_statistics.csv"
)
