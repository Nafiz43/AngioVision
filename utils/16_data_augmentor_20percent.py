#!/usr/bin/env python3

import os
import math
import random
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageEnhance

# =========================================================
# CONFIG
# =========================================================
INPUT_BASE_DIR = "/data/Deep_Angiography/DICOM_Sequence_Processed"
OUTPUT_BASE_DIR = "/data/Deep_Angiography/DICOM_Sequence_Processed_Augmented"

# Number of augmented versions to create PER sequence
# Final output per sequence:
#   1 original copy + NUM_AUG_PER_SEQUENCE augmented copies
NUM_AUG_PER_SEQUENCE = 4

# Keep only this fraction of frames from each sequence
# Example: 0.20 means keep 20% of frames
KEEP_FRAME_FRACTION = 0.20

# Always keep at least this many frames per sequence
MIN_FRAMES_TO_KEEP = 1

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

# Fill color used for rotation borders
FILL_COLOR = 0  # black for grayscale; works fine for RGB too as 0 -> black

# Brightness / Contrast jitter ranges
# factor = 1.0 means no change
BRIGHTNESS_RANGE = (0.90, 1.10)
CONTRAST_RANGE = (0.90, 1.10)

# Gamma jitter range
# gamma < 1 brightens, gamma > 1 darkens
GAMMA_RANGE = (0.90, 1.10)

# Slight Gaussian noise sigma range (in pixel intensity units)
GAUSSIAN_NOISE_SIGMA_RANGE = (2.0, 6.0)

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


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """
    Apply brightness jitter.
    factor=1.0 means unchanged.
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    """
    Apply contrast jitter.
    factor=1.0 means unchanged.
    """
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def adjust_gamma(img: Image.Image, gamma: float) -> Image.Image:
    """
    Apply gamma correction.
    gamma < 1 brightens, gamma > 1 darkens.
    """
    if gamma <= 0:
        raise ValueError("Gamma must be > 0")

    arr = np.asarray(img).astype(np.float32)

    # Normalize to [0, 1]
    arr_norm = arr / 255.0
    arr_gamma = np.power(arr_norm, gamma)
    arr_out = np.clip(arr_gamma * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(arr_out)


def add_gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    """
    Add slight Gaussian noise.
    sigma is in pixel intensity units.
    """
    arr = np.asarray(img).astype(np.float32)

    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape)
    noisy = arr + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy)


def apply_policy(img: Image.Image, policy: dict, np_rng: np.random.Generator) -> Image.Image:
    """
    Apply the same augmentation policy to an image.
    """
    out = img.copy()

    if policy.get("rotate", False):
        out = rotate_image(out, policy["degrees"])

    if policy.get("zoom", False):
        out = zoom_in_image(out, policy["zoom_factor"])

    if policy.get("brightness", False):
        out = adjust_brightness(out, policy["brightness_factor"])

    if policy.get("contrast", False):
        out = adjust_contrast(out, policy["contrast_factor"])

    if policy.get("gamma", False):
        out = adjust_gamma(out, policy["gamma_value"])

    if policy.get("gaussian_noise", False):
        out = add_gaussian_noise(out, policy["noise_sigma"], np_rng)

    return out


def make_policy(rng: random.Random) -> dict:
    """
    Randomly choose one policy for an augmented sequence copy.
    This policy is then applied consistently to all selected frames in that sequence.

    Flip has been removed completely.
    """
    policy_type = rng.choice([
        "rotate",
        "zoom",
        "brightness_contrast",
        "gamma_noise",
        "rotate_zoom",
        "rotate_brightness_contrast",
        "zoom_gamma_noise",
        "rotate_zoom_brightness_contrast",
        "rotate_zoom_gamma_noise",
        "brightness_contrast_gamma_noise",
        "rotate_zoom_brightness_contrast_gamma_noise",
    ])

    policy = {
        "type": policy_type,
        "rotate": False,
        "zoom": False,
        "brightness": False,
        "contrast": False,
        "gamma": False,
        "gaussian_noise": False,
        "degrees": rng.choice([-ROTATE_DEGREES, ROTATE_DEGREES]),
        "zoom_factor": rng.choice(ZOOM_FACTORS),
        "brightness_factor": rng.uniform(*BRIGHTNESS_RANGE),
        "contrast_factor": rng.uniform(*CONTRAST_RANGE),
        "gamma_value": rng.uniform(*GAMMA_RANGE),
        "noise_sigma": rng.uniform(*GAUSSIAN_NOISE_SIGMA_RANGE),
    }

    if "rotate" in policy_type:
        policy["rotate"] = True
    if "zoom" in policy_type:
        policy["zoom"] = True
    if "brightness_contrast" in policy_type:
        policy["brightness"] = True
        policy["contrast"] = True
    if "gamma_noise" in policy_type:
        policy["gamma"] = True
        policy["gaussian_noise"] = True

    return policy


# =========================================================
# FRAME SAMPLING HELPERS
# =========================================================
def select_frame_subset(frame_files, keep_fraction=0.20, min_frames=1):
    """
    Uniformly sample a subset of frames across the full sequence.

    Example:
      - 100 frames, keep_fraction=0.20 -> keep 20 frames
      - 7 frames, keep_fraction=0.20 -> keep max(1, ceil(1.4)) = 2 frames

    Returns a sorted list of frame paths.
    """
    n = len(frame_files)
    if n == 0:
        return []

    keep_n = max(min_frames, int(math.ceil(n * keep_fraction)))
    keep_n = min(keep_n, n)

    if keep_n == n:
        return list(frame_files)

    # Uniformly spaced indices from 0 to n-1
    indices = np.linspace(0, n - 1, num=keep_n, dtype=int)
    # Deduplicate in rare small-n cases
    indices = sorted(set(indices.tolist()))

    # If deduplication reduced count, pad deterministically
    if len(indices) < keep_n:
        chosen = set(indices)
        for i in range(n):
            if i not in chosen:
                indices.append(i)
                chosen.add(i)
                if len(indices) == keep_n:
                    break
        indices = sorted(indices)

    return [frame_files[i] for i in indices]


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
# FILE COPY HELPERS
# =========================================================
def copy_original_sequence_subset(sequence_dir: Path, input_base: Path, output_base: Path,
                                  selected_frame_files):
    """
    Copy only the selected subset of original frames into the output directory.

    Output example:
        INPUT  : base/outer/inner/frames/*.png
        OUTPUT : out/outer/inner/frames/<selected subset only>
    """
    rel_seq = sequence_dir.relative_to(input_base)
    out_seq_dir = output_base / rel_seq
    out_frames_dir = out_seq_dir / "frames"
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    for img_path in selected_frame_files:
        out_path = out_frames_dir / img_path.name
        with Image.open(img_path) as img:
            img.save(out_path)

    return str(out_seq_dir)


# =========================================================
# PER-SEQUENCE PROCESSING
# =========================================================
def process_one_sequence(sequence_dir_str: str, input_base_str: str, output_base_str: str,
                         num_aug_per_sequence: int, global_seed: int):
    """
    Process one sequence folder.
    Creates:
      - 1 original copied sequence containing only 20% of frames
      - num_aug_per_sequence augmented copies using only that same 20% subset
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

    # Deterministic numpy RNG for Gaussian noise
    np_rng_master = np.random.default_rng(seq_seed)

    output_base.mkdir(parents=True, exist_ok=True)

    created = []
    policy_logs = []

    try:
        # -------------------------------------------------
        # Select only 20% of frames uniformly across sequence
        # -------------------------------------------------
        selected_frame_files = select_frame_subset(
            frame_files,
            keep_fraction=KEEP_FRAME_FRACTION,
            min_frames=MIN_FRAMES_TO_KEEP
        )

        # -------------------------------------------------
        # 1) Copy original subset into output
        # -------------------------------------------------
        original_out_dir = copy_original_sequence_subset(
            sequence_dir,
            input_base,
            output_base,
            selected_frame_files
        )
        created.append(original_out_dir)
        policy_logs.append({
            "sequence_dir": original_out_dir,
            "policy": {"type": "original_copy_subset_only"},
            "num_frames_saved": len(selected_frame_files)
        })

        # -------------------------------------------------
        # 2) Create augmented copies using only selected subset
        # -------------------------------------------------
        for aug_idx in range(1, num_aug_per_sequence + 1):
            policy = make_policy(rng)

            # Example output:
            # OUTPUT_BASE / outer / inner__aug_01 / frames / file.png
            aug_rel_seq = rel_seq.parent / f"{rel_seq.name}__aug_{aug_idx:02d}"
            out_seq_dir = output_base / aug_rel_seq
            out_frames_dir = out_seq_dir / "frames"
            out_frames_dir.mkdir(parents=True, exist_ok=True)

            # Per-augmentation deterministic numpy RNG
            aug_seed = int(np_rng_master.integers(0, 2**32 - 1))
            np_rng = np.random.default_rng(aug_seed)

            for img_path in selected_frame_files:
                out_path = out_frames_dir / img_path.name

                with Image.open(img_path) as img:
                    aug_img = apply_policy(img, policy, np_rng)
                    aug_img.save(out_path)

            created.append(str(out_seq_dir))
            policy_logs.append({
                "sequence_dir": str(out_seq_dir),
                "policy": policy,
                "num_frames_saved": len(selected_frame_files)
            })

        return {
            "sequence": str(sequence_dir),
            "status": "done",
            "num_input_frames": len(frame_files),
            "num_selected_frames": len(selected_frame_files),
            "keep_fraction": KEEP_FRAME_FRACTION,
            "num_original_sequences_created": 1,
            "num_augmented_sequences_created": num_aug_per_sequence,
            "num_total_output_sequences_created": 1 + num_aug_per_sequence,
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
    print(f"INPUT_BASE_DIR              : {INPUT_BASE_DIR}")
    print(f"OUTPUT_BASE_DIR             : {OUTPUT_BASE_DIR}")
    print(f"NUM_AUG_PER_SEQUENCE        : {NUM_AUG_PER_SEQUENCE}")
    print(f"KEEP_FRAME_FRACTION         : {KEEP_FRAME_FRACTION}")
    print(f"FINAL COPIES PER SEQUENCE   : {1 + NUM_AUG_PER_SEQUENCE} (1 original + {NUM_AUG_PER_SEQUENCE} augmented)")
    print(f"MAX_WORKERS                 : {MAX_WORKERS}")
    print(f"GLOBAL_SEED                 : {GLOBAL_SEED}")
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
                print(
                    f"[{i}/{len(futures)}] DONE   : {seq} | "
                    f"selected {result.get('num_selected_frames', 0)}/{result.get('num_input_frames', 0)} frames"
                )
            elif status == "skipped":
                print(f"[{i}/{len(futures)}] SKIPPED: {seq} | {result.get('reason', '')}")
            else:
                print(f"[{i}/{len(futures)}] ERROR  : {seq}")
                print(result.get("error", "Unknown error"))

    # Summary
    done_count = sum(1 for r in results if r["status"] == "done")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")

    total_original_sequences = sum(
        r.get("num_original_sequences_created", 0) for r in results if r["status"] == "done"
    )
    total_augmented_sequences = sum(
        r.get("num_augmented_sequences_created", 0) for r in results if r["status"] == "done"
    )
    total_output_sequences = sum(
        r.get("num_total_output_sequences_created", 0) for r in results if r["status"] == "done"
    )
    total_input_frames = sum(
        r.get("num_input_frames", 0) for r in results if r["status"] == "done"
    )
    total_selected_frames = sum(
        r.get("num_selected_frames", 0) for r in results if r["status"] == "done"
    )

    print("\n=========================================================")
    print("AUGMENTATION COMPLETE")
    print("=========================================================")
    print(f"Sequences found                  : {len(sequence_dirs)}")
    print(f"Sequences processed successfully : {done_count}")
    print(f"Sequences skipped                : {skipped_count}")
    print(f"Sequences failed                 : {error_count}")
    print(f"Original sequence dirs copied    : {total_original_sequences}")
    print(f"Augmented sequence dirs created  : {total_augmented_sequences}")
    print(f"Total output sequence dirs       : {total_output_sequences}")
    print(f"Total input frames seen          : {total_input_frames}")
    print(f"Total frames kept/saved per copy : {total_selected_frames}")
    if total_input_frames > 0:
        pct = 100.0 * total_selected_frames / total_input_frames
        print(f"Effective kept fraction          : {pct:.2f}%")
    print(f"Saved under                      : {OUTPUT_BASE_DIR}")

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