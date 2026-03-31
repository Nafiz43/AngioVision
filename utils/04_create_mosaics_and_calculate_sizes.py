#!/usr/bin/env python3
"""
create_mosaics_and_calculate_sizes.py

Merged functionality:
1. Discover per-sequence directories under a base path.
2. Create/reuse mosaic.png inside each sequence directory.
3. Read mosaic dimensions (width, height).
4. Save a CSV containing:
   - outer_dir
   - inner_dir
   - mosaic_path
   - width
   - height
   - num_frames_found
   - num_frames_selected
   - mosaic_ok
   - error

Behavior:
- If mosaic already exists and is valid -> skip regeneration
- If mosaic exists but is corrupted/unreadable -> regenerate
- If --overwrite_mosaic is passed -> always regenerate

Parallel version:
- Multiprocessing across sequence dirs (ProcessPoolExecutor)
- Multithreading inside each process for frame loading/decoding (ThreadPoolExecutor)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from PIL import Image, ImageOps
from tqdm import tqdm


# =========================================================
# EASY-TO-CHANGE DEFAULTS
# =========================================================
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed_Augmented")
DEFAULT_OUTPUT_CSV = Path("mosaic_sizes.csv")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_FRAMES_SUBDIR = "frames"
DEFAULT_MOSAIC_NAME = "mosaic.png"

DEFAULT_MAX_FRAMES = 500
DEFAULT_STRIDE = 1

DEFAULT_TILE_SIZE = (384, 384)
DEFAULT_MOSAIC_MAX_COLS = 6

DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)
DEFAULT_THREADS = 4


# =========================================================
# FRAME UTILITIES
# =========================================================
def list_frame_files(seq_dir: Path, frames_subdir: str = "frames") -> List[Path]:
    """
    Prefer frames in seq_dir/<frames_subdir>/*.{png,jpg,...}.
    Fall back to recursive search if that folder doesn't exist or is empty.
    """
    frames_dir = seq_dir / frames_subdir

    # Preferred: exactly seq_dir/<frames_subdir>/*
    if frames_dir.exists() and frames_dir.is_dir():
        frames = [
            p for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        frames.sort(key=lambda p: p.name)
        if frames:
            return frames

    # Fallback: find images anywhere under seq_dir
    frames = [
        p for p in seq_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    frames.sort(key=lambda p: p.as_posix())
    return frames


def pick_frames(frames: List[Path], max_frames: int, stride: int) -> List[Path]:
    if stride < 1:
        stride = 1

    sampled = frames[::stride]

    if max_frames and max_frames > 0 and len(sampled) > max_frames:
        if max_frames == 1:
            return [sampled[len(sampled) // 2]]

        step = (len(sampled) - 1) / (max_frames - 1)
        idxs = [round(i * step) for i in range(max_frames)]
        sampled = [sampled[i] for i in idxs]

    return sampled


# =========================================================
# SEQUENCE DISCOVERY
# =========================================================
def find_sequence_dirs(base_path: Path, frames_subdir: str) -> List[Path]:
    """
    Return directories under base_path that look like sequence folders.

    Definition:
    - Any directory D such that D/<frames_subdir>/ exists AND contains
      at least one image file.
    """
    seq_dirs: List[Path] = []

    for d in base_path.rglob("*"):
        if not d.is_dir():
            continue

        frames_dir = d / frames_subdir
        if not frames_dir.exists() or not frames_dir.is_dir():
            continue

        try:
            has_image = any(
                p.is_file() and p.suffix.lower() in IMAGE_EXTS
                for p in frames_dir.iterdir()
            )
        except PermissionError:
            continue

        if has_image:
            seq_dirs.append(d)

    return sorted(seq_dirs, key=lambda p: p.as_posix())


# =========================================================
# MOSAIC UTILITIES
# =========================================================
def _open_image_rgb(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None


def _fit_to_box(img: Image.Image, box: Tuple[int, int]) -> Image.Image:
    target_w, target_h = box
    w, h = img.size

    if w <= 0 or h <= 0:
        return Image.new("RGB", (target_w, target_h), (0, 0, 0))

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    return canvas


def create_mosaic_image(
    frame_paths: List[Path],
    out_path: Path,
    tile_size: Tuple[int, int] = (384, 384),
    cols: Optional[int] = None,
    max_cols: int = 6,
    threads: int = 4,
) -> Optional[Path]:
    """
    Create a single mosaic PNG at out_path from the provided frame_paths.
    Returns out_path on success, None on failure.
    """
    if not frame_paths:
        return None

    tiles: List[Image.Image] = []
    threads = max(1, int(threads))

    if threads == 1:
        for p in frame_paths:
            img = _open_image_rgb(p)
            if img is not None:
                tiles.append(img)
    else:
        with ThreadPoolExecutor(max_workers=threads) as tp:
            futs = [tp.submit(_open_image_rgb, p) for p in frame_paths]
            for f in futs:
                img = f.result()
                if img is not None:
                    tiles.append(img)

    if not tiles:
        return None

    n = len(tiles)

    if cols is None:
        cols = min(max_cols, max(1, int(math.ceil(math.sqrt(n)))))
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))

    tile_w, tile_h = tile_size
    mosaic_w = cols * tile_w
    mosaic_h = rows * tile_h

    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), (0, 0, 0))

    for idx, img in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        tile = _fit_to_box(img, (tile_w, tile_h))
        mosaic.paste(tile, (c * tile_w, r * tile_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(out_path, format="PNG", optimize=True)
    return out_path


def get_image_size(image_path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Returns (width, height, error).
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return width, height, None
    except Exception as e:
        return None, None, str(e)


# =========================================================
# RESULT STRUCTURE
# =========================================================
@dataclass
class MosaicResult:
    seq_dir: Path
    seq_rel: str
    outer_dir: str
    inner_dir: str
    num_frames_found: int
    num_frames_selected: int
    mosaic_path: Path
    mosaic_ok: bool
    width: Optional[int] = None
    height: Optional[int] = None
    error: Optional[str] = None


# =========================================================
# PER-SEQUENCE WORKER
# =========================================================
def _process_one_sequence_dir(payload: dict) -> MosaicResult:
    seq_dir: Path = payload["seq_dir"]
    base_path: Path = payload["base_path"]
    frames_subdir: str = payload["frames_subdir"]
    mosaic_name: str = payload["mosaic_name"]
    max_frames: int = payload["max_frames"]
    stride: int = payload["stride"]
    tile_size: Tuple[int, int] = payload["tile_size"]
    mosaic_cols: Optional[int] = payload["mosaic_cols"]
    mosaic_max_cols: int = payload["mosaic_max_cols"]
    overwrite_mosaic: bool = payload["overwrite_mosaic"]
    debug: bool = payload["debug"]
    threads: int = payload["threads"]

    seq_rel = seq_dir.relative_to(base_path).as_posix()
    rel_parts = seq_dir.relative_to(base_path).parts

    if len(rel_parts) >= 2:
        outer_dir = rel_parts[-2]
        inner_dir = rel_parts[-1]
    elif len(rel_parts) == 1:
        outer_dir = "UNKNOWN"
        inner_dir = rel_parts[-1]
    else:
        outer_dir = "UNKNOWN"
        inner_dir = seq_dir.name

    mosaic_path = seq_dir / mosaic_name

    # -------------------------------------------------------
    # SKIP LOGIC (SAFE + STRICT)
    # -------------------------------------------------------
    if mosaic_path.exists() and not overwrite_mosaic:
        width, height, size_err = get_image_size(mosaic_path)

        if size_err is None:
            if debug:
                print(f"[SKIP] Valid mosaic exists: {mosaic_path}", flush=True)

            return MosaicResult(
                seq_dir=seq_dir,
                seq_rel=seq_rel,
                outer_dir=outer_dir,
                inner_dir=inner_dir,
                num_frames_found=-1,       # skipped frame scan
                num_frames_selected=-1,    # skipped frame selection
                mosaic_path=mosaic_path,
                mosaic_ok=True,
                width=width,
                height=height,
                error=None,
            )
        else:
            if debug:
                print(f"[REGEN] Corrupted mosaic detected: {mosaic_path}", flush=True)

    # Only scan frames when we actually need to generate/regenerate
    frames = list_frame_files(seq_dir, frames_subdir=frames_subdir)
    selected = pick_frames(frames, max_frames=max_frames, stride=stride)

    if debug:
        print(
            f"[PROCESS] {seq_rel}: found={len(frames)}, selected={len(selected)}, mosaic={mosaic_path}",
            flush=True,
        )

    try:
        created = create_mosaic_image(
            frame_paths=selected,
            out_path=mosaic_path,
            tile_size=tile_size,
            cols=mosaic_cols,
            max_cols=mosaic_max_cols,
            threads=threads,
        )

        if created is None or not created.exists():
            return MosaicResult(
                seq_dir=seq_dir,
                seq_rel=seq_rel,
                outer_dir=outer_dir,
                inner_dir=inner_dir,
                num_frames_found=len(frames),
                num_frames_selected=len(selected),
                mosaic_path=mosaic_path,
                mosaic_ok=False,
                width=None,
                height=None,
                error="Mosaic creation returned None or file missing.",
            )

        width, height, size_err = get_image_size(created)

        return MosaicResult(
            seq_dir=seq_dir,
            seq_rel=seq_rel,
            outer_dir=outer_dir,
            inner_dir=inner_dir,
            num_frames_found=len(frames),
            num_frames_selected=len(selected),
            mosaic_path=mosaic_path,
            mosaic_ok=size_err is None,
            width=width,
            height=height,
            error=size_err,
        )

    except Exception as e:
        err = str(e)[:500]
        if debug:
            print(f"[DEBUG] Failed {seq_rel}: {err}", flush=True)

        return MosaicResult(
            seq_dir=seq_dir,
            seq_rel=seq_rel,
            outer_dir=outer_dir,
            inner_dir=inner_dir,
            num_frames_found=len(frames) if "frames" in locals() else 0,
            num_frames_selected=len(selected) if "selected" in locals() else 0,
            mosaic_path=mosaic_path,
            mosaic_ok=False,
            width=None,
            height=None,
            error=err,
        )


# =========================================================
# PARALLEL DRIVER
# =========================================================
def create_mosaics_and_collect_sizes_parallel(
    seq_dirs: List[Path],
    base_path: Path,
    frames_subdir: str,
    mosaic_name: str,
    max_frames: int,
    stride: int,
    tile_size: Tuple[int, int],
    mosaic_cols: Optional[int],
    mosaic_max_cols: int,
    overwrite_mosaic: bool,
    debug: bool,
    workers: int,
    threads: int,
) -> List[MosaicResult]:
    results: List[MosaicResult] = []
    workers = max(1, int(workers))
    threads = max(1, int(threads))

    payloads = []
    for sd in seq_dirs:
        payloads.append(
            {
                "seq_dir": sd,
                "base_path": base_path,
                "frames_subdir": frames_subdir,
                "mosaic_name": mosaic_name,
                "max_frames": max_frames,
                "stride": stride,
                "tile_size": tile_size,
                "mosaic_cols": mosaic_cols,
                "mosaic_max_cols": mosaic_max_cols,
                "overwrite_mosaic": overwrite_mosaic,
                "debug": debug,
                "threads": threads,
            }
        )

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process_one_sequence_dir, payload) for payload in payloads]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Processing sequences", unit="seq"):
            results.append(f.result())

    results.sort(key=lambda r: r.seq_rel)
    return results


# =========================================================
# CSV WRITER
# =========================================================
def write_results_csv(results: List[MosaicResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "outer_dir",
            "inner_dir",
            "mosaic_path",
            "width",
            "height",
            "num_frames_found",
            "num_frames_selected",
            "mosaic_ok",
            "error",
        ])

        for r in results:
            writer.writerow([
                r.outer_dir,
                r.inner_dir,
                str(r.mosaic_path),
                r.width,
                r.height,
                r.num_frames_found,
                r.num_frames_selected,
                r.mosaic_ok,
                r.error or "",
            ])


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create mosaics and save mosaic image dimensions to CSV."
    )

    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--frames_subdir", type=str, default=DEFAULT_FRAMES_SUBDIR)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)

    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)

    parser.add_argument("--limit", type=int, default=None, help="Process only first N sequence dirs")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of processes. Default: cpu_count - 1",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="Threads per process for frame loading/decoding",
    )

    parser.add_argument("--mosaic_name", type=str, default=DEFAULT_MOSAIC_NAME)
    parser.add_argument(
        "--tile_size",
        type=int,
        nargs=2,
        default=list(DEFAULT_TILE_SIZE),
        metavar=("W", "H"),
        help="Tile size (width height) for each frame in the mosaic",
    )
    parser.add_argument("--mosaic_cols", type=int, default=None, help="Fixed number of mosaic columns")
    parser.add_argument("--mosaic_max_cols", type=int, default=DEFAULT_MOSAIC_MAX_COLS)
    parser.add_argument("--overwrite_mosaic", action="store_true")

    args = parser.parse_args()

    if not args.base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {args.base_path}")

    print("=" * 80)
    print("Starting mosaic creation + size collection")
    print("=" * 80)
    print(f"Base path           : {args.base_path}")
    print(f"Frames subdir       : {args.frames_subdir}")
    print(f"Mosaic filename     : {args.mosaic_name}")
    print(f"Output CSV          : {args.output_csv.resolve()}")
    print(f"Workers             : {args.workers}")
    print(f"Threads / process   : {args.threads}")
    print(f"Max frames          : {args.max_frames}")
    print(f"Stride              : {args.stride}")
    print(f"Tile size           : {tuple(args.tile_size)}")
    print(f"Overwrite mosaics   : {args.overwrite_mosaic}")
    print("=" * 80)

    print("[INFO] Discovering sequence directories...")
    seq_dirs = find_sequence_dirs(args.base_path, frames_subdir=args.frames_subdir)

    if args.limit is not None:
        seq_dirs = seq_dirs[: max(0, args.limit)]

    print(f"[INFO] Total sequence directories found: {len(seq_dirs)}")

    if not seq_dirs:
        print("[WARN] No sequence directories found. Exiting.")
        return

    print("[INFO] Valid existing mosaics will be skipped.")
    print("[INFO] Corrupted mosaics will be regenerated.")
    print("[INFO] Mosaic files are/will be saved inside each sequence directory as:")
    print(f"       <sequence_dir>/{args.mosaic_name}")
    print("[INFO] CSV summary will be saved at:")
    print(f"       {args.output_csv.resolve()}")

    results = create_mosaics_and_collect_sizes_parallel(
        seq_dirs=seq_dirs,
        base_path=args.base_path,
        frames_subdir=args.frames_subdir,
        mosaic_name=args.mosaic_name,
        max_frames=args.max_frames,
        stride=args.stride,
        tile_size=(int(args.tile_size[0]), int(args.tile_size[1])),
        mosaic_cols=args.mosaic_cols,
        mosaic_max_cols=args.mosaic_max_cols,
        overwrite_mosaic=args.overwrite_mosaic,
        debug=args.debug,
        workers=args.workers,
        threads=args.threads,
    )

    write_results_csv(results, args.output_csv)

    ok = sum(1 for r in results if r.mosaic_ok and r.mosaic_path.exists())
    fail = len(results) - ok
    skipped = sum(1 for r in results if r.num_frames_found == -1 and r.mosaic_ok)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"[DONE] Total processed         : {len(results)}")
    print(f"[DONE] Successful mosaics      : {ok}")
    print(f"[DONE] Skipped existing valid  : {skipped}")
    print(f"[DONE] Failed                  : {fail}")
    print(f"[DONE] Output CSV saved to     : {args.output_csv.resolve()}")
    print(f"[DONE] Mosaic images saved in  : each sequence directory under {args.base_path}")

    if args.debug and fail:
        print("[DEBUG] Failed entries:")
        for r in results:
            if not r.mosaic_ok:
                print(f"  - {r.seq_rel} :: {r.error}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial mosaics and CSV may already exist.", file=sys.stderr)
        raise