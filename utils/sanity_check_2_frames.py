#!/usr/bin/env python3

import os
import shutil
import argparse

BASE_DIR = "/data/Deep_Angiography/DICOM_Sequence_Processed"
OUTPUT_DIR = "/data/Deep_Angiography/DICOM-metadata-stats/sequences_with_2_frames"
OUTPUT_TXT = "/data/Deep_Angiography/DICOM-metadata-stats/sequences_with_2_frames.txt"


def get_frame_count(frames_dir):
    return len([
        f for f in os.listdir(frames_dir)
        if os.path.isfile(os.path.join(frames_dir, f)) and not f.startswith(".")
    ])


def get_mosaic_dest_filename(outer, inner):
    safe_outer = outer.replace("/", "_")
    safe_inner = inner.replace("/", "_")
    return f"{safe_outer}__{safe_inner}_mosaic.png"


def main():
    parser = argparse.ArgumentParser(
        description="Find sequences with exactly 2 frames and optionally delete them."
    )
    parser.add_argument(
        "-apply",
        action="store_true",
        help="Actually delete matching sequence directories and copied mosaics. "
             "Without this flag, the script only performs a dry run."
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

    total_sequences = 0
    count_exact_2 = 0

    matching_dirs = []
    deleted_dirs = 0
    deleted_mosaics = 0
    missing_output_mosaics = 0
    delete_errors = 0

    for outer in os.listdir(BASE_DIR):
        outer_path = os.path.join(BASE_DIR, outer)
        if not os.path.isdir(outer_path):
            continue

        for inner in os.listdir(outer_path):
            inner_path = os.path.join(outer_path, inner)
            if not os.path.isdir(inner_path):
                continue

            frames_dir = os.path.join(inner_path, "frames")
            if not os.path.isdir(frames_dir):
                continue

            num_frames = get_frame_count(frames_dir)
            total_sequences += 1

            if num_frames == 2:
                count_exact_2 += 1
                matching_dirs.append(inner_path)

                if args.apply:
                    # Delete the copied mosaic from OUTPUT_DIR if present
                    dest_filename = get_mosaic_dest_filename(outer, inner)
                    dest_path = os.path.join(OUTPUT_DIR, dest_filename)

                    if os.path.isfile(dest_path):
                        try:
                            os.remove(dest_path)
                            deleted_mosaics += 1
                        except Exception as e:
                            delete_errors += 1
                            print(f"[ERROR] Failed to delete copied mosaic: {dest_path}")
                            print(f"        Reason: {e}")
                    else:
                        missing_output_mosaics += 1

                    # Delete the actual sequence directory
                    try:
                        shutil.rmtree(inner_path)
                        deleted_dirs += 1
                        print(f"[DELETED] {inner_path}")
                    except Exception as e:
                        delete_errors += 1
                        print(f"[ERROR] Failed to delete directory: {inner_path}")
                        print(f"        Reason: {e}")
                else:
                    print(f"[DRY RUN] Would delete: {inner_path}")

    # Save current matching directories list
    with open(OUTPUT_TXT, "w") as f:
        for path in matching_dirs:
            f.write(path + "\n")

    print("\n===== SUMMARY =====")
    print(f"Total sequences checked: {total_sequences}")
    print(f"Sequences with exactly 2 frames: {count_exact_2}")
    print(f"Matching directories saved to: {OUTPUT_TXT}")

    if args.apply:
        print(f"Deleted sequence directories: {deleted_dirs}")
        print(f"Deleted copied mosaics: {deleted_mosaics}")
        print(f"Copied mosaics not found in output dir: {missing_output_mosaics}")
        print(f"Deletion errors: {delete_errors}")
    else:
        print("Dry run only. No files/directories were deleted.")
        print("Use -apply to perform deletion.")


if __name__ == "__main__":
    main()