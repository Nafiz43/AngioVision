from pathlib import Path
import csv

def generate_frame_stats(root_dir, output_csv):
    root_dir = Path(root_dir)
    rows = []

    # Iterate over OUTER directories
    for outer_dir in sorted(root_dir.iterdir()):
        if not outer_dir.is_dir():
            continue

        # Iterate over INNER directories
        for inner_dir in sorted(outer_dir.iterdir()):
            if not inner_dir.is_dir():
                continue

            # Recursively search for a 'frames' directory under inner_dir
            frames_dirs = [
                p for p in inner_dir.rglob("frames") if p.is_dir()
            ]

            if not frames_dirs:
                rows.append([
                    outer_dir.name,
                    inner_dir.name,
                    0
                ])
                continue

            # Usually there should be exactly ONE frames dir per inner dir
            for frames_dir in frames_dirs:
                frame_count = sum(
                    1 for f in frames_dir.iterdir() if f.is_file()
                )

                rows.append([
                    outer_dir.name,
                    inner_dir.name,
                    frame_count
                ])

    # 🔥 Sort by number_of_frames (index 2) in descending order
    rows.sort(key=lambda x: x[2], reverse=True)

    # Write CSV
    output_csv = Path(output_csv)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "outer_dir_name",
            "inner_dir_name",
            "number_of_frames"
        ])
        writer.writerows(rows)


# -----------------------------
# Example usage
# -----------------------------
generate_frame_stats(
    root_dir="/data/Deep_Angiography/DICOM_Sequence_Processed",
    output_csv="/data/Deep_Angiography/DICOM-metadata-stats/frame_statistics.csv"
)

