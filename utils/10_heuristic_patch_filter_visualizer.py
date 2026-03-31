#!/usr/bin/env python3
"""
heuristic_patch_filter_visualizer.py

Given an input image, this script:
1. Splits it into patches
2. Computes heuristic scores for each patch
3. Classifies patches as informative / uninformative
4. Draws:
   - green filled transparent patches for informative regions
   - red filled transparent patches for uninformative regions
5. Saves the visualization

Heuristics used:
- variance
- entropy
- edge density
- dark pixel ratio

Example:
    python3 heuristic_patch_filter_visualizer.py \
        --input_image /path/to/frame.png \
        --output_image /path/to/frame_patch_overlay.png \
        --patch_size 64 \
        --informative_percent 20
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


# =========================
# CONFIGURATION DEFAULTS
# =========================
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
        description="Visualize informative vs uninformative patches using heuristic scoring."
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output_image",
        type=str,
        required=True,
        help="Path to save overlay image"
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
        help=f"Pixel intensity threshold below which a pixel is considered dark (default: {DEFAULT_DARK_PIXEL_THRESHOLD})"
    )
    parser.add_argument(
        "--dark_pixel_ratio_reject",
        type=float,
        default=DEFAULT_DARK_PIXEL_RATIO_REJECT,
        help=f"Reject a patch if dark pixel ratio >= this value (default: {DEFAULT_DARK_PIXEL_RATIO_REJECT})"
    )
    parser.add_argument(
        "--min_variance",
        type=float,
        default=DEFAULT_MIN_VARIANCE,
        help=f"Minimum variance for a patch to be considered non-trivial (default: {DEFAULT_MIN_VARIANCE})"
    )
    parser.add_argument(
        "--min_edge_density",
        type=float,
        default=DEFAULT_MIN_EDGE_DENSITY,
        help=f"Minimum edge density for a patch to be considered non-trivial (default: {DEFAULT_MIN_EDGE_DENSITY})"
    )
    parser.add_argument(
        "--min_entropy",
        type=float,
        default=DEFAULT_MIN_ENTROPY,
        help=f"Minimum entropy for a patch to be considered non-trivial (default: {DEFAULT_MIN_ENTROPY})"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Transparency for filled patch overlays, between 0 and 1 (default: {DEFAULT_ALPHA})"
    )
    return parser.parse_args()


def compute_entropy(gray_patch: np.ndarray) -> float:
    """Compute Shannon entropy of a grayscale patch."""
    hist = cv2.calcHist([gray_patch], [0], None, [256], [0, 256]).ravel()
    hist_sum = hist.sum()
    if hist_sum == 0:
        return 0.0
    prob = hist / hist_sum
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def compute_edge_density(gray_patch: np.ndarray) -> float:
    """Compute fraction of edge pixels using Canny."""
    edges = cv2.Canny(gray_patch, 50, 150)
    return float(np.count_nonzero(edges)) / float(edges.size)


def dark_pixel_ratio(gray_patch: np.ndarray, dark_threshold: int) -> float:
    """Fraction of pixels that are near-black."""
    return float(np.mean(gray_patch <= dark_threshold))


def normalize_array(values: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0, 1]."""
    if len(values) == 0:
        return values
    vmin = np.min(values)
    vmax = np.max(values)
    if math.isclose(float(vmin), float(vmax)):
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def extract_patch_features(gray_patch: np.ndarray, dark_threshold: int) -> dict:
    """Compute heuristic features for one patch."""
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
    """
    Split image into patches, compute features, reject clearly uninformative ones,
    then rank valid patches and mark top percentage as informative.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    patches: list[dict] = []

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = gray[y:min(y + patch_size, h), x:min(x + patch_size, w)]

            # Skip incomplete border patches
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


def draw_patch_overlay(
    image_bgr: np.ndarray,
    patches: list[dict],
    alpha: float = DEFAULT_ALPHA
) -> np.ndarray:
    """
    Draw filled transparent green/red patch overlays plus borders.
    Green = informative
    Red   = uninformative
    """
    alpha = max(0.0, min(1.0, alpha))

    base = image_bgr.copy()
    overlay = image_bgr.copy()

    # Fill full patches on overlay
    for patch in patches:
        x, y, w, h = patch["x"], patch["y"], patch["w"], patch["h"]
        color = GREEN if patch["label"] == "informative" else RED
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

    # Blend so original image remains visible
    blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)

    # Draw borders on top
    for patch in patches:
        x, y, w, h = patch["x"], patch["y"], patch["w"], patch["h"]
        color = GREEN if patch["label"] == "informative" else RED
        cv2.rectangle(blended, (x, y), (x + w, y + h), color, LINE_THICKNESS)

    return blended


def add_legend(image_bgr: np.ndarray) -> np.ndarray:
    """Add legend to image."""
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


def main():
    args = parse_args()

    input_path = Path(args.input_image)
    output_path = Path(args.output_image)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {input_path}")

    patches = classify_patches(
        image_bgr=image_bgr,
        patch_size=args.patch_size,
        informative_percent=args.informative_percent,
        dark_threshold=args.dark_pixel_threshold,
        dark_ratio_reject=args.dark_pixel_ratio_reject,
        min_variance=args.min_variance,
        min_edge_density=args.min_edge_density,
        min_entropy=args.min_entropy,
    )

    overlay = draw_patch_overlay(
        image_bgr=image_bgr,
        patches=patches,
        alpha=args.alpha,
    )
    overlay = add_legend(overlay)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), overlay)
    if not ok:
        raise IOError(f"Failed to save output image to: {output_path}")

    total = len(patches)
    informative = sum(1 for p in patches if p["label"] == "informative")
    uninformative = sum(1 for p in patches if p["label"] == "uninformative")

    print(f"Input image       : {input_path}")
    print(f"Output image      : {output_path}")
    print(f"Patch size        : {args.patch_size}")
    print(f"Total patches     : {total}")
    print(f"Informative       : {informative}")
    print(f"Uninformative     : {uninformative}")
    print(f"Overlay alpha     : {args.alpha}")


if __name__ == "__main__":
    main()


python3 heuristic_patch_filter_visualizer.py \
    --input_image /data/Deep_Angiography/DICOM_Sequence_Processed/0AVNTO~C/2.16.840.1.113883.3.16.242948424383568667903940832500591782968/frames/2.16.840.1.113883.3.16.242948424383568667903940832500591782968_frame_0001.png \
    --output_image /data/Deep_Angiography/DICOM-metadata-stats/sample_frame_overlay.png \
    --patch_size 64 \
    --informative_percent 20


# /data/Deep_Angiography/DICOM_Sequence_Processed/0AVNTO~C/2.16.840.1.113883.3.16.242948424383568667903940832500591782968/frames/2.16.840.1.113883.3.16.242948424383568667903940832500591782968_frame_0001.png
