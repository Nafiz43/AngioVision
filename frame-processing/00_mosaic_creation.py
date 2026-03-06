#!/usr/bin/env python3
"""
creating_mosaics.py

Stage 1 only: Discover INNER "sequence dirs" (any dir under base_path that contains
<frames_subdir>/ with images), sample frames, and create/reuse a mosaic image saved
inside each sequence dir (e.g., <seq_dir>/mosaic.png).

Parallel version:
- Multiprocessing across sequence dirs (ProcessPoolExecutor)
- Multithreading inside each process for frame loading/decoding (ThreadPoolExecutor)

No LLM calls. No CSV output (optional debug prints only).
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from PIL import Image, ImageOps
from tqdm import tqdm

# -----------------------------
# Defaults
# -----------------------------
# DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM_Sequence_Processed")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# Frame utilities
# -----------------------------
def list_frame_files(seq_dir: Path, frames_subdir: str = "frames") -> List[Path]:
    """
    Prefer frames in seq_dir/<frames_subdir>/*.{png,jpg,...}.
    Fall back to recursive search if that folder doesn't exist or is empty.
    """
    frames_dir = seq_dir / frames_subdir

    # 1) Preferred: exactly seq_dir/<frames_subdir>/*
    if frames_dir.exists() and frames_dir.is_dir():
        frames = [
            p for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        frames.sort(key=lambda p: p.name)
        if frames:
            return frames

    # 2) Fallback: find images anywhere under seq_dir
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


# -----------------------------
# Sequence-dir discovery (inner dirs)
# -----------------------------
def find_sequence_dirs(base_path: Path, frames_subdir: str) -> List[Path]:
    """
    Return directories under base_path that look like per-DICOM/per-sequence folders.

    Definition:
    - Any directory D such that D/<frames_subdir>/ exists AND contains at least one image file.
    """
    seq_dirs: List[Path] = []
    # rglob("*") is OK; discovery is usually not the slowest part vs image IO/resize.
    for d in base_path.rglob("*"):
        if not d.is_dir():
            continue
        frames_dir = d / frames_subdir
        if not frames_dir.exists() or not frames_dir.is_dir():
            continue
        try:
            has_image = any(
                (p.is_file() and p.suffix.lower() in IMAGE_EXTS)
                for p in frames_dir.iterdir()
            )
        except PermissionError:
            continue
        if has_image:
            seq_dirs.append(d)

    return sorted(seq_dirs, key=lambda p: p.as_posix())


# -----------------------------
# Mosaic/splicing utilities
# -----------------------------
def _open_image_rgb(path: Path) -> Optional[Image.Image]:
    # Keep this top-level for picklability and thread pool usage.
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
    threads: int = 24,
) -> Optional[Path]:
    """
    Create a single mosaic PNG at out_path from the provided frame_paths.
    Returns out_path on success, None on failure.

    Uses threads to load/decode/transpose/convert frames faster.
    """
    if not frame_paths:
        return None

    # Load frames concurrently (I/O + decode)
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

    # Resize+paste (CPU-ish). Keep single-threaded to avoid huge memory spikes.
    for idx, img in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        tile = _fit_to_box(img, (tile_w, tile_h))
        mosaic.paste(tile, (c * tile_w, r * tile_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(out_path, format="PNG", optimize=True)
    return out_path


# -----------------------------
# Stage-1 driver
# -----------------------------
@dataclass
class MosaicResult:
    seq_dir: Path
    seq_rel: str
    num_frames_found: int
    num_frames_selected: int
    mosaic_path: Path
    mosaic_ok: bool
    error: Optional[str] = None


def _process_one_sequence_dir(payload: dict) -> MosaicResult:
    """
    Worker function run in a separate process.
    Uses threads internally for loading frames.
    """
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

    frames = list_frame_files(seq_dir, frames_subdir=frames_subdir)
    selected = pick_frames(frames, max_frames=max_frames, stride=stride)

    seq_rel = seq_dir.relative_to(base_path).as_posix()
    mosaic_path = seq_dir / mosaic_name

    if debug:
        print(f"[DEBUG] {seq_rel}: found {len(frames)} frames, selected {len(selected)}", flush=True)

    if mosaic_path.exists() and not overwrite_mosaic:
        return MosaicResult(
            seq_dir=seq_dir,
            seq_rel=seq_rel,
            num_frames_found=len(frames),
            num_frames_selected=len(selected),
            mosaic_path=mosaic_path,
            mosaic_ok=True,
            error=None,
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
        ok = created is not None and created.exists()
        err = None if ok else "Mosaic creation returned None or file missing."
        return MosaicResult(
            seq_dir=seq_dir,
            seq_rel=seq_rel,
            num_frames_found=len(frames),
            num_frames_selected=len(selected),
            mosaic_path=mosaic_path,
            mosaic_ok=ok,
            error=err,
        )
    except Exception as e:
        err = str(e)[:300]
        if debug:
            print(f"[DEBUG] Failed mosaic for {seq_rel}: {err}", flush=True)
        return MosaicResult(
            seq_dir=seq_dir,
            seq_rel=seq_rel,
            num_frames_found=len(frames),
            num_frames_selected=len(selected),
            mosaic_path=mosaic_path,
            mosaic_ok=False,
            error=err,
        )


def create_mosaics_for_sequence_dirs_parallel(
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
    """
    Parallel driver:
    - processes: per seq_dir
    - threads: per-process frame loading
    """
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

    # Important: tqdm in parent process only
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process_one_sequence_dir, pl) for pl in payloads]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Creating mosaics", unit="seq"):
            results.append(f.result())

    # Keep stable order (optional)
    results.sort(key=lambda r: r.seq_rel)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Create mosaics for per-sequence frame dirs (parallel).")
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--frames_subdir", type=str, default="frames")
    parser.add_argument("--max_frames", type=int, default=144)
    parser.add_argument("--stride", type=int, default=2)

    parser.add_argument("--limit", type=int, default=None, help="Process only first N sequence dirs.")
    parser.add_argument("--debug", action="store_true")

    # Parallelism knobs
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of processes (one sequence dir per task). Default: cpu_count-1",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Threads per process for loading/decoding frames. Default: 4",
    )

    # Mosaic settings
    parser.add_argument("--mosaic_name", type=str, default="mosaic.png")
    parser.add_argument(
        "--tile_size",
        type=int,
        nargs=2,
        default=[384, 384],
        metavar=("W", "H"),
        help="Tile size (width height) for each frame in the mosaic",
    )
    parser.add_argument("--mosaic_cols", type=int, default=None, help="Fixed mosaic columns (default auto).")
    parser.add_argument("--mosaic_max_cols", type=int, default=6, help="Max cols when auto layout.")
    parser.add_argument("--overwrite_mosaic", action="store_true")

    args = parser.parse_args()

    if not args.base_path.exists():
        raise FileNotFoundError(args.base_path)

    seq_dirs = find_sequence_dirs(args.base_path, frames_subdir=args.frames_subdir)
    if args.limit is not None:
        seq_dirs = seq_dirs[: max(0, args.limit)]

    print(f"Discovered {len(seq_dirs)} sequence (inner) directories")
    print(f"Frames subdir: {args.frames_subdir}")
    print(f"Mosaic name: {args.mosaic_name}")
    print(f"Parallelism: workers={args.workers}, threads/process={args.threads}")

    results = create_mosaics_for_sequence_dirs_parallel(
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

    ok = sum(1 for r in results if r.mosaic_ok and r.mosaic_path.exists())
    fail = len(results) - ok
    print(f"Done ✔ mosaics ready: {ok}/{len(results)} (failed: {fail})")

    if args.debug and fail:
        for r in results:
            if not r.mosaic_ok:
                print(f"[DEBUG] FAIL {r.seq_rel} :: {r.error}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial mosaics are preserved.", file=sys.stderr)
        raise
