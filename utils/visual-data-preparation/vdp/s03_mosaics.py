"""
Step 03 — Create/reuse mosaic.png per sequence + record mosaic sizes.

mosaic.png lands inside each sequence directory (alongside frames/);
the sizes CSV goes to the run dir. Valid existing mosaics are skipped
unless cfg.overwrite_mosaic is set; corrupted ones are regenerated.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps
from tqdm import tqdm

from vdp.common import IMAGE_EXTENSIONS, find_sequence_dirs, write_csv


def _open_rgb(path: Path) -> Optional[Image.Image]:
    try:
        img = ImageOps.exif_transpose(Image.open(path))
        return img.convert("RGB") if img.mode != "RGB" else img
    except Exception:
        return None


def _fit_to_box(img: Image.Image, box: Tuple[int, int]) -> Image.Image:
    tw, th = box
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (tw, th), (0, 0, 0))
    scale = min(tw / w, th / h)
    resized = img.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    canvas.paste(resized, ((tw - resized.width) // 2, (th - resized.height) // 2))
    return canvas


def _pick_frames(frames: List[Path], max_frames: int) -> List[Path]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[len(frames) // 2]]
    step = (len(frames) - 1) / (max_frames - 1)
    return [frames[round(i * step)] for i in range(max_frames)]


def _image_size(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    try:
        with Image.open(path) as img:
            return img.size[0], img.size[1], None
    except Exception as e:
        return None, None, str(e)


def _process_sequence(seq_dir_str: str, tile: int, max_cols: int,
                      max_frames: int, overwrite: bool) -> Dict:
    seq_dir = Path(seq_dir_str)
    mosaic_path = seq_dir / "mosaic.png"
    row = {
        "outer_dir": seq_dir.parent.name, "inner_dir": seq_dir.name,
        "mosaic_path": str(mosaic_path), "width": None, "height": None,
        "num_frames_found": 0, "num_frames_selected": 0,
        "mosaic_ok": False, "error": "",
    }

    if mosaic_path.exists() and not overwrite:
        w, h, err = _image_size(mosaic_path)
        if err is None:
            row.update(width=w, height=h, mosaic_ok=True,
                       num_frames_found=-1, num_frames_selected=-1)
            return row
        # corrupted -> fall through and regenerate

    frames = sorted(
        p for p in (seq_dir / "frames").iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected = _pick_frames(frames, max_frames)
    row.update(num_frames_found=len(frames), num_frames_selected=len(selected))

    tiles = [img for p in selected if (img := _open_rgb(p)) is not None]
    if not tiles:
        row["error"] = "No readable frames"
        return row

    cols = max(1, min(max_cols, math.ceil(math.sqrt(len(tiles)))))
    rows_n = math.ceil(len(tiles) / cols)
    mosaic = Image.new("RGB", (cols * tile, rows_n * tile), (0, 0, 0))
    for idx, img in enumerate(tiles):
        mosaic.paste(_fit_to_box(img, (tile, tile)),
                     ((idx % cols) * tile, (idx // cols) * tile))

    try:
        mosaic.save(mosaic_path, format="PNG", optimize=True)
        w, h, err = _image_size(mosaic_path)
        row.update(width=w, height=h, mosaic_ok=err is None, error=err or "")
    except Exception as e:
        row["error"] = str(e)[:500]
    return row


def run(cfg, run_dir: Path) -> Dict:
    step_dir = run_dir / "03_mosaics"
    seq_dirs = find_sequence_dirs(Path(cfg.output_root))

    results: List[Dict] = []
    with tqdm(total=len(seq_dirs), unit="seq", desc="[03] Mosaics") as pbar:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futures = [
                ex.submit(_process_sequence, str(d), cfg.mosaic_tile_size,
                          cfg.mosaic_max_cols, cfg.mosaic_max_frames,
                          cfg.overwrite_mosaic)
                for d in seq_dirs
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
                pbar.update(1)

    results.sort(key=lambda r: (r["outer_dir"], r["inner_dir"]))
    write_csv(step_dir / "mosaic_sizes.csv",
              ["outer_dir", "inner_dir", "mosaic_path", "width", "height",
               "num_frames_found", "num_frames_selected", "mosaic_ok", "error"],
              results)

    ok = sum(1 for r in results if r["mosaic_ok"])
    skipped = sum(1 for r in results if r["num_frames_found"] == -1)
    summary = {"sequences": len(results), "mosaics_ok": ok,
               "reused_existing": skipped, "failed": len(results) - ok}
    print(f"[03] {summary}")
    return summary
