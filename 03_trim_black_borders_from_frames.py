#!/usr/bin/env python3
"""
trim_black_borders.py

Recursively trims black padding from images under a root directory.
Overwrites images in-place (atomic replace).

Default root:
  /data/Deep_Angiography/DICOM_Sequence_Processed
"""

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def compute_bbox_nonblack(gray: np.ndarray, thresh: int) -> tuple[int, int, int, int] | None:
    """
    Return bbox (left, upper, right, lower) in PIL coords, or None if nothing found.
    """
    # mask of content pixels (not black)
    mask = gray > thresh
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    top, bottom = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())

    # PIL crop uses (left, upper, right_exclusive, lower_exclusive)
    return (left, top, right + 1, bottom + 1)


def trim_image_inplace(path: Path, thresh: int, margin: int, min_content_frac: float, dry_run: bool) -> bool:
    """
    Trims black borders from one image. Returns True if modified, False otherwise.
    """
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)  # handle orientation safely
            w, h = im.size

            # Convert to grayscale for detecting non-black region
            gray_im = im.convert("L")
            gray = np.array(gray_im)

            # If threshold wasn't provided well, a tiny adaptive boost helps:
            # use max of fixed thresh and a low percentile of non-zero values.
            nonzero = gray[gray > 0]
            if nonzero.size > 0:
                adaptive = int(np.percentile(nonzero, 1))  # very low percentile
                effective_thresh = max(thresh, min(adaptive, 30))
            else:
                effective_thresh = thresh

            bbox = compute_bbox_nonblack(gray, effective_thresh)
            if bbox is None:
                return False

            left, top, right, bottom = bbox

            # Optional margin (keep a small border so we don't clip real content)
            left = max(0, left - margin)
            top = max(0, top - margin)
            right = min(w, right + margin)
            bottom = min(h, bottom + margin)

            # Skip if crop would be basically the same
            if left == 0 and top == 0 and right == w and bottom == h:
                return False

            # Guard: if content region is tiny, likely mis-detection; skip
            crop_w, crop_h = right - left, bottom - top
            if (crop_w * crop_h) < (w * h * min_content_frac):
                return False

            cropped = im.crop((left, top, right, bottom))

            if dry_run:
                print(f"[DRY] {path}  ({w}x{h} -> {crop_w}x{crop_h})")
                return True

            # Atomic overwrite: write to temp file then replace
            suffix = path.suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(path.parent)) as tmp:
                tmp_path = Path(tmp.name)

            # Preserve format if possible
            save_kwargs = {}
            fmt = (im.format or "").upper()
            if fmt:
                save_kwargs["format"] = fmt

            # JPEG can't store alpha; if needed, convert safely
            if suffix in {".jpg", ".jpeg"} and cropped.mode in {"RGBA", "LA"}:
                cropped = cropped.convert("RGB")

            cropped.save(tmp_path, **save_kwargs)
            os.replace(tmp_path, path)

            print(f"[OK]  {path}  ({w}x{h} -> {crop_w}x{crop_h})")
            return True

    except Exception as e:
        print(f"[ERR] {path}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/data/Deep_Angiography/DICOM_Sequence_Processed"),
        help="Root directory to recursively process",
    )
    ap.add_argument(
        "--thresh",
        type=int,
        default=5,
        help="Pixel threshold (0-255). Pixels <= thresh are considered black padding.",
    )
    ap.add_argument(
        "--margin",
        type=int,
        default=2,
        help="Extra pixels to keep around detected content bbox.",
    )
    ap.add_argument(
        "--min-content-frac",
        type=float,
        default=0.02,
        help="Skip cropping if detected content area is smaller than this fraction of the image.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change, without overwriting files.",
    )
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    total = 0
    changed = 0

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            total += 1
            if trim_image_inplace(
                p,
                thresh=args.thresh,
                margin=args.margin,
                min_content_frac=args.min_content_frac,
                dry_run=args.dry_run,
            ):
                changed += 1

    print(f"\nDone. Images scanned: {total}, changed: {changed}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
