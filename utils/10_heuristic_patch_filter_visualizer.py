#!/usr/bin/env python3
"""
hardcoded_patch_grid_visualizer_parallel.py

Uses 5 hardcoded frame paths, processes them in parallel with ProcessPoolExecutor,
draws heuristic informative/uninformative patch overlays, and saves the
5 processed images side-by-side.

Run:
    python3 hardcoded_patch_grid_visualizer_parallel.py \
        --output_image /data/Deep_Angiography/sample_patch_grid.png \
        --patch_size 64 \
        --informative_percent 20 \
        --target_height 512 \
        --max_workers 5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm


# =========================
# HARDCODED INPUT FRAMES
# =========================
HARDCODED_FRAME_PATHS = [
    "/data/Deep_Angiography/DICOM_Sequence_Processed/03_DSA 3 LD/2.16.840.1.113883.3.16.90729317817737798748724286231292096160/frames/2.16.840.1.113883.3.16.90729317817737798748724286231292096160_frame_0005.png",
    "/data/Deep_Angiography/DICOM_Sequence_Processed/gzeAvoZBK5/2.16.840.1.113883.3.16.27887747233506821641442574601125740969/frames/2.16.840.1.113883.3.16.27887747233506821641442574601125740969_frame_0016.png",
    "/data/Deep_Angiography/DICOM_Sequence_Processed/received/2.16.840.1.113883.3.16.206533095497994768887122049484591788029/frames/2.16.840.1.113883.3.16.206533095497994768887122049484591788029_frame_0007.png",
    "/data/Deep_Angiography/DICOM_Sequence_Processed/14_DSA 3/2.16.840.1.113883.3.16.229884264127170088524043314078208850582/frames/2.16.840.1.113883.3.16.229884264127170088524043314078208850582_frame_0008.png",
    "/data/Deep_Angiography/DICOM_Sequence_Processed/DICOM/2.16.840.1.113883.3.16.229149697742230994696452930366730877174/frames/2.16.840.1.113883.3.16.229149697742230994696452930366730877174_frame_0011.png",
]


# =========================
# CONFIGURATION DEFAULTS
# =========================
DEFAULT_TARGET_HEIGHT = 512
DEFAULT_MAX_WORKERS = 5

DEFAULT_PATCH_SIZE = 64
DEFAULT_INFORMATIVE_PERCENT = 20.0
DEFAULT_DARK_PIXEL_THRESHOLD = 20
DEFAULT_DARK_PIXEL_RATIO_REJECT = 0.85
DEFAULT_MIN_VARIANCE = 10.0
DEFAULT_MIN_EDGE_DENSITY = 0.01
DEFAULT_MIN_ENTROPY = 2.0
DEFAULT_ALPHA = 0.30

# Score weights
WEIGHT_VARIANCE = 1.0
WEIGHT_ENTROPY = 1.0
WEIGHT_EDGE_DENSITY = 1.5

# Drawing
GREEN = (0, 255, 0)
RED = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
LINE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process 5 hardcoded frames in parallel and save them side-by-side."
    )
    parser.add_argument(
        "--output_image",
        type=str,
        required=True,
        help="Path to save the final side-by-side output image"
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=DEFAULT_TARGET_HEIGHT,
        help=f"Resize each output image to this height before stacking (default: {DEFAULT_TARGET_HEIGHT})"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of worker processes (default: {DEFAULT_MAX_WORKERS})"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f"Patch size (default: {DEFAULT_PATCH_SIZE})"
    )
    parser.add_argument(
        "--informative_percent",
        type=float,
        default=DEFAULT_INFORMATIVE_PERCENT,
        help=f"Top percentage of valid patches to mark as informative (default: {DEFAULT_INFORMATIVE_PERCENT})"
    )
    parser.add_argument(
        "--dark_pixel_threshold",
        type=int,
        default=DEFAULT_DARK_PIXEL_THRESHOLD,
        help=f"Dark pixel threshold (default: {DEFAULT_DARK_PIXEL_THRESHOLD})"
    )
    parser.add_argument(
        "--dark_pixel_ratio_reject",
        type=float,
        default=DEFAULT_DARK_PIXEL_RATIO_REJECT,
        help=f"Reject patch if dark pixel ratio >= this value (default: {DEFAULT_DARK_PIXEL_RATIO_REJECT})"
    )
    parser.add_argument(
        "--min_variance",
        type=float,
        default=DEFAULT_MIN_VARIANCE,
        help=f"Minimum variance threshold (default: {DEFAULT_MIN_VARIANCE})"
    )
    parser.add_argument(
        "--min_edge_density",
        type=float,
        default=DEFAULT_MIN_EDGE_DENSITY,
        help=f"Minimum edge density threshold (default: {DEFAULT_MIN_EDGE_DENSITY})"
    )
    parser.add_argument(
        "--min_entropy",
        type=float,
        default=DEFAULT_MIN_ENTROPY,
        help=f"Minimum entropy threshold (default: {DEFAULT_MIN_ENTROPY})"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Overlay alpha in [0,1] (default: {DEFAULT_ALPHA})"
    )
    return parser.parse_args()


def compute_entropy(gray_patch: np.ndarray) -> float:
    hist = cv2.calcHist([gray_patch], [0], None, [256], [0, 256]).ravel()
    hist_sum = hist.sum()
    if hist_sum == 0:
        return 0.0
    prob = hist / hist_sum
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def compute_edge_density(gray_patch: np.ndarray) -> float:
    edges = cv2.Canny(gray_patch, 50, 150)
    return float(np.count_nonzero(edges)) / float(edges.size)


def dark_pixel_ratio(gray_patch: np.ndarray, dark_threshold: int) -> float:
    return float(np.mean(gray_patch <= dark_threshold))


def normalize_array(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    vmin = np.min(values)
    vmax = np.max(values)
    if math.isclose(float(vmin), float(vmax)):
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def extract_patch_features(gray_patch: np.ndarray, dark_threshold: int) -> dict:
    variance = float(np.var(gray_patch))
    entropy = compute_entropy(gray_patch)
    edge_density = compute_edge_density(gray_patch)
    dark_ratio = dark_pixel_ratio(gray_patch, dark_threshold)

    return {
        "variance": variance,
        "entropy": entropy,
        "edge_density": edge_density,
        "dark_ratio": dark_ratio,
    }


def classify_patches(
    image_bgr: np.ndarray,
    patch_size: int,
    informative_percent: float,
    dark_threshold: int,
    dark_ratio_reject: float,
    min_variance: float,
    min_edge_density: float,
    min_entropy: float,
) -> list[dict]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    patches: list[dict] = []

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = gray[y:min(y + patch_size, h), x:min(x + patch_size, w)]

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            feats = extract_patch_features(patch, dark_threshold)

            trivial = (
                feats["dark_ratio"] >= dark_ratio_reject
                or feats["variance"] < min_variance
                or feats["edge_density"] < min_edge_density
                or feats["entropy"] < min_entropy
            )

            patches.append({
                "x": x,
                "y": y,
                "w": patch_size,
                "h": patch_size,
                "variance": feats["variance"],
                "entropy": feats["entropy"],
                "edge_density": feats["edge_density"],
                "dark_ratio": feats["dark_ratio"],
                "trivial": trivial,
                "score": None,
                "label": None,
            })

    valid_indices = [i for i, p in enumerate(patches) if not p["trivial"]]

    if valid_indices:
        variances = np.array([patches[i]["variance"] for i in valid_indices], dtype=np.float32)
        entropies = np.array([patches[i]["entropy"] for i in valid_indices], dtype=np.float32)
        edge_densities = np.array([patches[i]["edge_density"] for i in valid_indices], dtype=np.float32)

        var_norm = normalize_array(variances)
        ent_norm = normalize_array(entropies)
        edge_norm = normalize_array(edge_densities)

        scores = (
            WEIGHT_VARIANCE * var_norm
            + WEIGHT_ENTROPY * ent_norm
            + WEIGHT_EDGE_DENSITY * edge_norm
        )

        for idx, score in zip(valid_indices, scores):
            patches[idx]["score"] = float(score)

        num_informative = max(1, int(round(len(valid_indices) * informative_percent / 100.0)))
        ranked_valid = sorted(valid_indices, key=lambda i: patches[i]["score"], reverse=True)
        informative_set = set(ranked_valid[:num_informative])

        for i, patch in enumerate(patches):
            if patch["trivial"]:
                patch["label"] = "uninformative"
            elif i in informative_set:
                patch["label"] = "informative"
            else:
                patch["label"] = "uninformative"
    else:
        for patch in patches:
            patch["label"] = "uninformative"

    return patches


def draw_patch_overlay(image_bgr: np.ndarray, patches: list[dict], alpha: float) -> np.ndarray:
    alpha = max(0.0, min(1.0, alpha))

    base = image_bgr.copy()
    overlay = image_bgr.copy()

    for patch in patches:
        x, y, w, h = patch["x"], patch["y"], patch["w"], patch["h"]
        color = GREEN if patch["label"] == "informative" else RED
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

    blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)

    for patch in patches:
        x, y, w, h = patch["x"], patch["y"], patch["w"], patch["h"]
        color = GREEN if patch["label"] == "informative" else RED
        cv2.rectangle(blended, (x, y), (x + w, y + h), color, LINE_THICKNESS)

    return blended


def add_legend(image_bgr: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    overlay = out.copy()

    box_x, box_y = 20, 20
    box_w, box_h = 320, 85

    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)

    cv2.rectangle(out, (35, 35), (55, 55), GREEN, -1)
    cv2.putText(out, "Informative patch", (70, 50), FONT, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

    cv2.rectangle(out, (35, 65), (55, 85), RED, -1)
    cv2.putText(out, "Uninformative patch", (70, 80), FONT, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

    return out


def resize_to_target_height(image_bgr: np.ndarray, target_height: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h == target_height:
        return image_bgr
    scale = target_height / float(h)
    target_width = max(1, int(round(w * scale)))
    return cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


def process_single_frame(task: dict) -> dict:
    frame_path = Path(task["frame_path"])
    image_bgr = cv2.imread(str(frame_path))

    if image_bgr is None:
        return {
            "success": False,
            "frame_path": str(frame_path),
            "index": task["index"],
            "error": f"Failed to read image: {frame_path}",
        }

    patches = classify_patches(
        image_bgr=image_bgr,
        patch_size=task["patch_size"],
        informative_percent=task["informative_percent"],
        dark_threshold=task["dark_pixel_threshold"],
        dark_ratio_reject=task["dark_pixel_ratio_reject"],
        min_variance=task["min_variance"],
        min_edge_density=task["min_edge_density"],
        min_entropy=task["min_entropy"],
    )

    overlay = draw_patch_overlay(
        image_bgr=image_bgr,
        patches=patches,
        alpha=task["alpha"],
    )
    overlay = add_legend(overlay)
    overlay = resize_to_target_height(overlay, task["target_height"])

    return {
        "success": True,
        "frame_path": str(frame_path),
        "index": task["index"],
        "image": overlay,
    }


def main():
    args = parse_args()
    output_path = Path(args.output_image)

    # Validate hardcoded paths first
    frame_paths = [Path(p) for p in HARDCODED_FRAME_PATHS]
    for p in frame_paths:
        if not p.exists():
            raise FileNotFoundError(f"Hardcoded frame not found: {p}")

    tasks = []
    for idx, frame_path in enumerate(frame_paths):
        tasks.append({
            "index": idx,
            "frame_path": str(frame_path),
            "patch_size": args.patch_size,
            "informative_percent": args.informative_percent,
            "dark_pixel_threshold": args.dark_pixel_threshold,
            "dark_pixel_ratio_reject": args.dark_pixel_ratio_reject,
            "min_variance": args.min_variance,
            "min_edge_density": args.min_edge_density,
            "min_entropy": args.min_entropy,
            "alpha": args.alpha,
            "target_height": args.target_height,
        })

    max_workers = max(1, min(args.max_workers, len(tasks)))
    results = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_frame, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
            result = future.result()
            if not result["success"]:
                print(f"[WARN] {result['error']}")
                continue
            results[result["index"]] = result

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        raise RuntimeError("No valid images were processed successfully.")

    output_images = [r["image"] for r in valid_results]
    combined = np.hstack(output_images)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), combined)
    if not ok:
        raise IOError(f"Failed to save output image to: {output_path}")

    print(f"\nSaved combined image to: {output_path}\n")
    print("Used hardcoded frames:")
    for i, r in enumerate(valid_results, start=1):
        print(f"{i}. {r['frame_path']}")


if __name__ == "__main__":
    main()