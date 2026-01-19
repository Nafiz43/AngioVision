#!/usr/bin/env python3
"""
extract_labels_from_frames.py

Automates extraction of structured clinical labels from angiography image sequences
using a multimodal LLM (Qwen-2.5VL) served via Ollama.

WORKFLOW (UPDATED: two-stage pipeline)
1) Mosaic creation stage:
   - Discover INNER "sequence dirs" (any dir under base_path that contains <frames_subdir>/ with images)
   - For each sequence dir, sample frames and create/reuse a mosaic image
   - Store the mosaic inside that sequence dir (e.g., <seq_dir>/mosaic.png)

2) LLM inference stage:
   - For each sequence dir, send ONLY the stored mosaic image to the model
   - Ask a fixed set of questions
   - Append results incrementally to a CSV (fault-tolerant)

Other:
- Samples representative frames (stride + max_frames)
- Requires strict JSON responses
"""

import argparse
import base64
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm
from PIL import Image, ImageOps
from utils.questions import QUESTIONS


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL_NAME = "qwen2.5vl:32b"
DEFAULT_TIMEOUT_S = 180

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BASE_PROMPT = """ROLE
You are a meticulous clinical information extraction engine for interventional radiology angiography image sequences.

WHY THIS MATTERS
Your output will be used to build a research-grade labeled dataset. High precision is more important than guessing.

SOURCE OF TRUTH
Use ONLY the provided image. Do not use outside medical knowledge.

TASK
Answer exactly ONE question:
Question: {QUESTION}

IMPORTANT CONTEXT (MOSAIC)
You are given ONE mosaic (spliced) image that contains multiple frames tiled in reading order (left-to-right, top-to-bottom).
Treat each tile as an individual frame.

STRICT RULES
1) Do not guess. If unclear, return “Not stated” or “Unclear”.
2) If evidence conflicts across tiles, return “Conflicting”.
3) Cite frames by filename when possible (choose from the provided filenames list).

OUTPUT FORMAT (JSON ONLY)
Return:
- answer
- confidence (0–100)
- evidence (≤3 short frame-based cues; reference filenames if possible)
- notes

IMAGE
A single mosaic image is attached.
"""


# -----------------------------
# CSV helpers
# -----------------------------
def ensure_csv_header(out_path: Path, columns: List[str]) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(out_path, index=False)


def append_csv_row(out_path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    ordered = {c: row.get(c) for c in columns}
    pd.DataFrame([ordered]).to_csv(out_path, mode="a", header=False, index=False)


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
            p
            for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        frames.sort(key=lambda p: p.name)
        if frames:
            return frames

    # 2) Fallback: find images anywhere under seq_dir
    frames = [
        p for p in seq_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
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


def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


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
    for d in base_path.rglob("*"):
        if not d.is_dir():
            continue
        frames_dir = d / frames_subdir
        if not frames_dir.exists() or not frames_dir.is_dir():
            continue
        has_image = any(
            (p.is_file() and p.suffix.lower() in IMAGE_EXTS) for p in frames_dir.iterdir()
        )
        if has_image:
            seq_dirs.append(d)

    return sorted(seq_dirs, key=lambda p: p.as_posix())


# -----------------------------
# Mosaic/splicing utilities
# -----------------------------
def _open_image_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


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
) -> Optional[Path]:
    """
    Create a single mosaic PNG at out_path from the provided frame_paths.
    Returns out_path on success, None on failure.
    """
    if not frame_paths:
        return None

    tiles: List[Image.Image] = []
    for p in frame_paths:
        try:
            tiles.append(_open_image_rgb(p))
        except Exception:
            continue

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


# -----------------------------
# Two-stage pipeline: stage 1 = make/store mosaics
# -----------------------------
@dataclass
class SequenceMosaicInfo:
    seq_dir: Path
    seq_rel: str
    num_frames_found: int
    selected_frames: List[Path]
    frame_names: List[str]
    mosaic_path: Path
    mosaic_ok: bool
    mosaic_error: Optional[str] = None


def prepare_mosaics(
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
) -> List[SequenceMosaicInfo]:
    """
    Stage 1:
    For each sequence directory, sample frames and create/reuse a mosaic image on disk.
    Returns a list of SequenceMosaicInfo objects (one per sequence directory).
    """
    infos: List[SequenceMosaicInfo] = []

    for seq_dir in tqdm(seq_dirs, desc="Creating mosaics", unit="seq"):
        frames = list_frame_files(seq_dir, frames_subdir=frames_subdir)
        selected = pick_frames(frames, max_frames=max_frames, stride=stride)

        seq_rel = seq_dir.relative_to(base_path).as_posix()
        mosaic_path = seq_dir / mosaic_name

        if debug:
            print(f"[DEBUG] {seq_rel}: found {len(frames)} frames, selected {len(selected)}")

        mosaic_ok = False
        mosaic_error: Optional[str] = None

        if mosaic_path.exists() and not overwrite_mosaic:
            mosaic_ok = True
        else:
            try:
                created = create_mosaic_image(
                    frame_paths=selected,
                    out_path=mosaic_path,
                    tile_size=tile_size,
                    cols=mosaic_cols,
                    max_cols=mosaic_max_cols,
                )
                mosaic_ok = created is not None and created.exists()
                if not mosaic_ok:
                    mosaic_error = "Mosaic creation returned None or file missing."
            except Exception as e:
                mosaic_ok = False
                mosaic_error = str(e)[:300]
                if debug:
                    print(f"[DEBUG] Failed to create mosaic for {seq_rel}: {mosaic_error}")

        infos.append(
            SequenceMosaicInfo(
                seq_dir=seq_dir,
                seq_rel=seq_rel,
                num_frames_found=len(frames),
                selected_frames=selected,
                frame_names=[p.name for p in selected],
                mosaic_path=mosaic_path,
                mosaic_ok=mosaic_ok,
                mosaic_error=mosaic_error,
            )
        )

    return infos


# -----------------------------
# Ollama helpers
# -----------------------------
def build_prompt(question: str, frame_names: List[str], mosaic_name: str) -> str:
    names_block = "\n".join(frame_names) if frame_names else "(none)"
    return (
        BASE_PROMPT.format(QUESTION=question)
        + f"\nMOSAIC FILENAME\n{mosaic_name}\n"
        + f"\nFRAME FILENAMES (tiles are in this order: left-to-right, top-to-bottom)\n{names_block}\n"
    )


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def ollama_chat_with_images(
    prompt: str,
    images_b64: List[str],
    model: str,
    url: str,
    timeout_s: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        ],
        "stream": False,
        "options": {"temperature": 0},
    }

    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


# -----------------------------
# Two-stage pipeline: stage 2 = send mosaics to LLM
# -----------------------------
def run_llm_over_mosaics(
    infos: List[SequenceMosaicInfo],
    out_path: Path,
    out_cols: List[str],
    model: str,
    url: str,
    timeout_s: int,
    delay: float,
    debug: bool,
) -> None:
    """
    Stage 2:
    For each sequence dir, read the stored mosaic and query the LLM for each question.
    Append results incrementally to CSV.
    """
    total_tasks = len(infos) * len(QUESTIONS)

    with tqdm(total=total_tasks, desc="Analyzing", unit="q") as pbar:
        for info in infos:
            # Load mosaic -> base64 (only if it exists and mosaic_ok)
            images_b64: List[str] = []
            if info.mosaic_ok and info.mosaic_path.exists():
                try:
                    images_b64 = [b64_image(info.mosaic_path)]
                except Exception as e:
                    images_b64 = []
                    if debug:
                        print(f"[DEBUG] Could not read mosaic {info.mosaic_path}: {str(e)[:300]}")
            else:
                if debug and info.mosaic_error:
                    print(f"[DEBUG] Mosaic not available for {info.seq_rel}: {info.mosaic_error}")

            for q in QUESTIONS:
                if not images_b64:
                    append_csv_row(
                        out_path,
                        {
                            "sequence_dir": info.seq_rel,
                            "num_frames_found": info.num_frames_found,
                            "num_frames_sent": 0,
                            "question": q,
                            "answer": "Not stated",
                            "confidence": 0,
                            "evidence": "[]",
                            "notes": "No usable mosaic image (or no usable frames).",
                        },
                        out_cols,
                    )
                    pbar.update(1)
                    continue

                try:
                    raw = ollama_chat_with_images(
                        build_prompt(q, info.frame_names, info.mosaic_path.name),
                        images_b64,
                        model,
                        url,
                        timeout_s,
                    )
                    parsed = safe_parse_json(raw)
                    if not parsed:
                        raise ValueError(f"Non-JSON model output: {raw[:200]}")

                    append_csv_row(
                        out_path,
                        {
                            "sequence_dir": info.seq_rel,
                            "num_frames_found": info.num_frames_found,
                            "num_frames_sent": len(images_b64),  # should be 1
                            "question": q,
                            "answer": parsed.get("answer"),
                            "confidence": parsed.get("confidence"),
                            "evidence": json.dumps(parsed.get("evidence", [])),
                            "notes": parsed.get("notes", f"Mosaic used: {info.mosaic_path.name}"),
                        },
                        out_cols,
                    )

                except Exception as e:
                    append_csv_row(
                        out_path,
                        {
                            "sequence_dir": info.seq_rel,
                            "num_frames_found": info.num_frames_found,
                            "num_frames_sent": len(images_b64),
                            "question": q,
                            "answer": "Unclear",
                            "confidence": 0,
                            "evidence": "[]",
                            "notes": f"Error: {str(e)[:200]}",
                        },
                        out_cols,
                    )

                pbar.update(1)
                if delay > 0:
                    time.sleep(delay)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen-2.5VL (via Ollama) on frame directories and extract labels."
    )
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max_frames", type=int, default=24)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.0)

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N sequence directories (after discovery + sort).",
    )

    parser.add_argument(
        "--frames_subdir",
        type=str,
        default="frames",
        help="Name of the subfolder inside each sequence_dir that contains frames",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info when frames are missing / before querying the model",
    )

    # Mosaic settings
    parser.add_argument(
        "--mosaic_name",
        type=str,
        default="mosaic.png",
        help="Filename for the saved mosaic image within each *sequence* (inner) directory",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        nargs=2,
        default=[384, 384],
        metavar=("W", "H"),
        help="Tile size (width height) for each frame in the mosaic",
    )
    parser.add_argument(
        "--mosaic_cols",
        type=int,
        default=None,
        help="Fixed number of columns in the mosaic (default: auto)",
    )
    parser.add_argument(
        "--mosaic_max_cols",
        type=int,
        default=6,
        help="Max columns when mosaic_cols is not set (auto layout)",
    )
    parser.add_argument(
        "--overwrite_mosaic",
        action="store_true",
        help="Recreate mosaic even if it already exists",
    )

    args = parser.parse_args()

    if not args.base_path.exists():
        raise FileNotFoundError(args.base_path)

    # Discover INNER sequence dirs (per DICOM/sequence)
    seq_dirs = find_sequence_dirs(args.base_path, frames_subdir=args.frames_subdir)

    if args.limit is not None:
        seq_dirs = seq_dirs[: max(0, args.limit)]

    out_path = args.out or (args.base_path / "frames_extracted_labels.csv")

    out_cols = [
        "sequence_dir",         # relative path under base_path
        "num_frames_found",
        "num_frames_sent",      # should be 1 (mosaic)
        "question",
        "answer",
        "confidence",
        "evidence",
        "notes",
    ]
    ensure_csv_header(out_path, out_cols)

    print(f"Discovered {len(seq_dirs)} sequence (inner) directories")
    print(f"Output CSV: {out_path}")
    print(f"Frames subdir: {args.frames_subdir}")
    print(
        f"Mosaic: {args.mosaic_name} | tile_size={tuple(args.tile_size)} | cols={args.mosaic_cols or 'auto'}"
    )

    # -----------------------------
    # Stage 1: Create + store mosaics
    # -----------------------------
    infos = prepare_mosaics(
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
    )

    ok_count = sum(1 for i in infos if i.mosaic_ok and i.mosaic_path.exists())
    print(f"Mosaics ready: {ok_count}/{len(infos)}")

    # -----------------------------
    # Stage 2: Send stored mosaics to LLM
    # -----------------------------
    total_tasks = len(infos) * len(QUESTIONS)
    print(f"Total questions: {total_tasks}")
    run_llm_over_mosaics(
        infos=infos,
        out_path=out_path,
        out_cols=out_cols,
        model=args.model,
        url=args.url,
        timeout_s=args.timeout,
        delay=args.delay,
        debug=args.debug,
    )

    print("Done ✔ Results saved incrementally.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results are preserved.", file=sys.stderr)
        raise
