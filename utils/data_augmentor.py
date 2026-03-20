#!/usr/bin/env python3

import os
import random
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, ImageOps

# =========================================================
# CONFIG
# =========================================================
INPUT_BASE_DIR = "/data/Deep_Angiography/DICOM_Sequence_Processed"
OUTPUT_BASE_DIR = "/data/Deep_Angiography/DICOM_Sequence_Processed_Augmented"

# Number of augmented versions to create PER sequence
# If you want original + 4 augmented = 5x total, set this to 4
NUM_AUG_PER_SEQUENCE = 4

# Process pool workers
MAX_WORKERS = max(1, os.cpu_count() - 1)

# Random seed for reproducibility
GLOBAL_SEED = 42

# Supported image extensions
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Rotation magnitude
ROTATE_DEGREES = 5

# Zoom factors to sample from
ZOOM_FACTORS = [1.10, 1.15, 1.20]

# Flip mode: "horizontal" or "vertical"
FLIP_MODE = "horizontal"

# Fill color used for rotation borders
FILL_COLOR = 0  # black for grayscale; works fine for RGB too as 0 -> black

# Resampling quality
try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_LANCZOS = Image.LANCZOS


# =========================================================
# IMAGE AUGMENTATION HELPERS
# =========================================================
def rotate_image(img: Image.Image, degrees: float) -> Image.Image:
    """
    Rotate while keeping original canvas size.
    """
    return img.rotate(
        degrees,
        resample=RESAMPLE_BICUBIC,
        expand=False,
        fillcolor=FILL_COLOR
    )


def flip_image(img: Image.Image, mode: str = "horizontal") -> Image.Image:
    """
    Flip image horizontally or vertically.
    """
    if mode == "horizontal":
        return ImageOps.mirror(img)
    elif mode == "vertical":
        return ImageOps.flip(img)
    else:
        raise ValueError(f"Unsupported flip mode: {mode}")


def zoom_in_image(img: Image.Image, zoom_factor: float) -> Image.Image:
    """
    Zoom in by center-cropping and resizing back to original size.
    zoom_factor > 1 means zooming in.
    """
    if zoom_factor <= 1.0:
        return img.copy()

    w, h = img.size
    new_w = int(round(w / zoom_factor))
    new_h = int(round(h / zoom_factor))

    left = max(0, (w - new_w) // 2)
    top = max(0, (h - new_h) // 2)
    right = min(w, left + new_w)
    bottom = min(h, top + new_h)

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), RESAMPLE_LANCZOS)


def apply_policy(img: Image.Image, policy: dict) -> Image.Image:
    """
    Apply the same augmentation policy to an image.
    """
    out = img.copy()

    if policy["type"] == "rotate":
        out = rotate_image(out, policy["degrees"])

    elif policy["type"] == "flip":
        out = flip_image(out, policy["mode"])

    elif policy["type"] == "zoom":
        out = zoom_in_image(out, policy["zoom_factor"])

    elif policy["type"] == "rotate_flip":
        out = rotate_image(out, policy["degrees"])
        out = flip_image(out, policy["mode"])

    elif policy["type"] == "rotate_zoom":
        out = rotate_image(out, policy["degrees"])
        out = zoom_in_image(out, policy["zoom_factor"])

    elif policy["type"] == "flip_zoom":
        out = flip_image(out, policy["mode"])
        out = zoom_in_image(out, policy["zoom_factor"])

    elif policy["type"] == "rotate_flip_zoom":
        out = rotate_image(out, policy["degrees"])
        out = flip_image(out, policy["mode"])
        out = zoom_in_image(out, policy["zoom_factor"])

    else:
        raise ValueError(f"Unknown policy type: {policy['type']}")

    return out


def make_policy(rng: random.Random) -> dict:
    """
    Randomly choose one policy for an augmented sequence copy.
    This policy is then applied consistently to all frames in that sequence.
    """
    policy_type = rng.choice([
        "rotate",
        "flip",
        "zoom",
        "rotate_flip",
        "rotate_zoom",
        "flip_zoom",
        "rotate_flip_zoom",
    ])

    degrees = rng.choice([-ROTATE_DEGREES, ROTATE_DEGREES])
    zoom_factor = rng.choice(ZOOM_FACTORS)

    return {
        "type": policy_type,
        "degrees": degrees,
        "mode": FLIP_MODE,
        "zoom_factor": zoom_factor,
    }


# =========================================================
# DISCOVERY
# =========================================================
def find_sequence_dirs(base_dir: str):
    """
    Find all sequence directories that contain a 'frames' subdirectory.
    Assumes structure like:
        base_dir / outer_dir / inner_dir / frames / *.png
    """
    sequence_dirs = []

    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Input base directory not found: {base_dir}")

    for outer_dir in base.iterdir():
        if not outer_dir.is_dir():
            continue

        for inner_dir in outer_dir.iterdir():
            if not inner_dir.is_dir():
                continue

            frames_dir = inner_dir / "frames"
            if frames_dir.is_dir():
                image_files = sorted(
                    [
                        p for p in frames_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in VALID_EXTS
                    ]
                )
                if image_files:
                    sequence_dirs.append(inner_dir)

    return sequence_dirs


# =========================================================
# PER-SEQUENCE PROCESSING
# =========================================================
def process_one_sequence(sequence_dir_str: str, input_base_str: str, output_base_str: str,
                         num_aug_per_sequence: int, global_seed: int):
    """
    Process one sequence folder.
    Creates num_aug_per_sequence augmented copies, each with a single policy
    applied to all frames in that sequence.
    """
    sequence_dir = Path(sequence_dir_str)
    input_base = Path(input_base_str)
    output_base = Path(output_base_str)

    rel_seq = sequence_dir.relative_to(input_base)
    frames_dir = sequence_dir / "frames"

    frame_files = sorted(
        [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    )

    if not frame_files:
        return {
            "sequence": str(sequence_dir),
            "status": "skipped",
            "reason": "No valid image files found in frames/"
        }

    # Deterministic per-sequence RNG
    seq_seed = global_seed + (hash(str(rel_seq)) & 0xFFFFFFFF)
    rng = random.Random(seq_seed)

    # Make sure base output directory exists
    output_base.mkdir(parents=True, exist_ok=True)

    created = []
    policy_logs = []

    try:
        for aug_idx in range(1, num_aug_per_sequence + 1):
            policy = make_policy(rng)

            # Example output:
            # OUTPUT_BASE / outer / inner__aug_01 / frames / file.png
            aug_rel_seq = rel_seq.parent / f"{rel_seq.name}__aug_{aug_idx:02d}"
            out_seq_dir = output_base / aug_rel_seq
            out_frames_dir = out_seq_dir / "frames"
            out_frames_dir.mkdir(parents=True, exist_ok=True)

            for img_path in frame_files:
                out_path = out_frames_dir / img_path.name

                with Image.open(img_path) as img:
                    # Preserve mode as much as possible
                    aug_img = apply_policy(img, policy)
                    aug_img.save(out_path)

            created.append(str(out_seq_dir))
            policy_logs.append({
                "augmented_sequence_dir": str(out_seq_dir),
                "policy": policy
            })

        return {
            "sequence": str(sequence_dir),
            "status": "done",
            "num_input_frames": len(frame_files),
            "num_augmented_sequences_created": len(created),
            "created_dirs": created,
            "policies": policy_logs
        }

    except Exception as e:
        return {
            "sequence": str(sequence_dir),
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"INPUT_BASE_DIR  : {INPUT_BASE_DIR}")
    print(f"OUTPUT_BASE_DIR : {OUTPUT_BASE_DIR}")
    print(f"NUM_AUG_PER_SEQUENCE : {NUM_AUG_PER_SEQUENCE}")
    print(f"MAX_WORKERS : {MAX_WORKERS}")
    print(f"GLOBAL_SEED : {GLOBAL_SEED}")
    print()

    sequence_dirs = find_sequence_dirs(INPUT_BASE_DIR)
    print(f"Found {len(sequence_dirs)} sequence directories with frames/")

    if not sequence_dirs:
        print("No valid sequences found. Exiting.")
        return

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    futures = []
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for seq_dir in sequence_dirs:
            futures.append(
                executor.submit(
                    process_one_sequence,
                    str(seq_dir),
                    INPUT_BASE_DIR,
                    OUTPUT_BASE_DIR,
                    NUM_AUG_PER_SEQUENCE,
                    GLOBAL_SEED
                )
            )

        for i, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            results.append(result)

            status = result.get("status", "unknown")
            seq = result.get("sequence", "unknown_sequence")

            if status == "done":
                print(f"[{i}/{len(futures)}] DONE   : {seq}")
            elif status == "skipped":
                print(f"[{i}/{len(futures)}] SKIPPED: {seq} | {result.get('reason', '')}")
            else:
                print(f"[{i}/{len(futures)}] ERROR  : {seq}")
                print(result.get("error", "Unknown error"))

    # Summary
    done_count = sum(1 for r in results if r["status"] == "done")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")

    total_augmented_sequences = sum(
        r.get("num_augmented_sequences_created", 0) for r in results if r["status"] == "done"
    )

    print("\n=========================================================")
    print("AUGMENTATION COMPLETE")
    print("=========================================================")
    print(f"Sequences found                 : {len(sequence_dirs)}")
    print(f"Sequences processed successfully: {done_count}")
    print(f"Sequences skipped               : {skipped_count}")
    print(f"Sequences failed                : {error_count}")
    print(f"Total augmented sequence dirs   : {total_augmented_sequences}")
    print(f"Saved under                     : {OUTPUT_BASE_DIR}")

    if error_count > 0:
        print("\nSome sequences failed. First few errors:\n")
        shown = 0
        for r in results:
            if r["status"] == "error":
                print(f"Sequence : {r['sequence']}")
                print(f"Error    : {r['error']}")
                print(r["traceback"])
                print("-" * 80)
                shown += 1
                if shown >= 3:
                    break


if __name__ == "__main__":
    main()