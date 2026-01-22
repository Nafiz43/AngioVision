#!/usr/bin/env python3
"""
extract_labels_from_frames.py

Automates extraction of structured clinical labels from angiography image sequences
using a multimodal LLM (Qwen-2.5VL) served via Ollama.

FIX (IMPORTANT):
- Your dataset typically looks like:
    base_path/<outer_dir>/<dicom_uid_dir>/frames/*.png

  Older versions treated base_path's direct children as sequence dirs (outer_dir),
  then fell back to recursive search and accidentally collected ALL frames under that outer_dir.

- This version discovers INNER dicom_uid_dir folders as sequence dirs by finding directories
  that contain <frames_subdir>/ with at least one image.

ADDITIONAL UPDATES (per request):
- Adds CSV columns:
  1) timestamp_utc
  2) model_name
- Appends row-by-row to an existing CSV (no re-dumping full DF).
- Creates the CSV only if it doesn't already exist.
"""

import argparse
import base64
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

QUESTIONS = [
    "Which artery is catheterized?",
    "Is variant anatomy present?",
    "Is there evidence of hemorrhage or contrast extravasation in this sequence?",
    "Is there evidence of arterial or venous dissection?",
    "Is stenosis present in any visualized vessel?",
    "Is an endovascular stent visible in this sequence?",
]

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL_NAME = "qwen3-vl:32b"
DEFAULT_TIMEOUT_S = 180

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BASE_PROMPT = """ROLE
You are a meticulous clinical information extraction engine for interventional radiology angiography image sequences.

WHY THIS MATTERS
Your output will be used to build a research-grade labeled dataset. High precision is more important than guessing.

SOURCE OF TRUTH
Use ONLY the provided frames. Do not use outside medical knowledge.

TASK
Answer exactly ONE question:
Question: {QUESTION}

STRICT RULES
1) Do not guess. If unclear, return “Not stated” or “Unclear”.
2) If evidence conflicts across frames, return “Conflicting”.
3) Cite frames by filename when possible (choose from the provided filenames).

OUTPUT FORMAT (JSON ONLY)
Return:
- answer
- confidence (0–100)
- evidence (≤3 short frame-based cues; reference filenames if possible)
- notes

FRAMES
A set of angiography frames is attached.
"""


# -----------------------------
# CSV helpers (row-by-row append)
# -----------------------------
def ensure_csv_header(out_path: Path, columns: List[str]) -> None:
    """
    Create the output CSV with header only if it doesn't exist or is empty.
    Does NOT overwrite an existing CSV.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()


def append_csv_row(out_path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    """
    Append exactly one row to an existing CSV. If CSV doesn't exist, it will be created
    with the provided header first.
    """
    ensure_csv_header(out_path, columns)

    # enforce column order + fill missing keys with None
    ordered = {c: row.get(c) for c in columns}

    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writerow(ordered)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# -----------------------------
# Sequence discovery
# -----------------------------
def _has_images_in_dir(d: Path) -> bool:
    try:
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                return True
    except Exception:
        return False
    return False


def discover_sequence_dirs(base_path: Path, frames_subdir: str) -> List[Path]:
    """
    Find all seq_dir such that:
      <seq_dir>/<frames_subdir>/ exists AND contains at least one image.
    Returns list of seq_dir paths (parent of the frames folder).
    """
    seq_dirs: List[Path] = []
    seen = set()

    for frames_dir in base_path.rglob(frames_subdir):
        if not frames_dir.is_dir():
            continue
        if not _has_images_in_dir(frames_dir):
            continue

        seq_dir = frames_dir.parent
        if seq_dir == base_path:
            continue

        key = seq_dir.resolve()
        if key not in seen:
            seen.add(key)
            seq_dirs.append(seq_dir)

    seq_dirs.sort(key=lambda p: p.as_posix())
    return seq_dirs


def get_outer_and_inner(seq_dir: Path, base_path: Path) -> Tuple[str, str]:
    """
    Expected:
      base_path/<outer>/<inner>
    Best-effort:
      inner = seq_dir.name
      outer = first path component under base_path if possible
    """
    inner = seq_dir.name
    try:
        rel = seq_dir.relative_to(base_path)
        parts = rel.parts
        outer = parts[0] if len(parts) >= 2 else (seq_dir.parent.name if seq_dir.parent != base_path else "")
    except Exception:
        outer = seq_dir.parent.name
    return outer, inner


# -----------------------------
# Frame utilities
# -----------------------------
def list_frame_files(seq_dir: Path, frames_subdir: str = "frames") -> List[Path]:
    """
    Prefer frames in seq_dir/<frames_subdir>/*.{png,jpg,...}.
    Fall back to recursive search ONLY within seq_dir.
    """
    frames_dir = seq_dir / frames_subdir

    if frames_dir.exists() and frames_dir.is_dir():
        frames = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        frames.sort(key=lambda p: p.name)
        if frames:
            return frames

    frames = [p for p in seq_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
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
# Ollama helpers
# -----------------------------
def build_prompt(question: str, frame_names: List[str]) -> str:
    names_block = "\n".join(frame_names) if frame_names else "(none)"
    return BASE_PROMPT.format(QUESTION=question) + f"\nFRAME FILENAMES\n{names_block}\n"


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
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen-VL (via Ollama) on DICOM frame directories and extract labels."
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
        help="Process only the first N sequence dirs (sorted by path)",
    )

    parser.add_argument(
        "--frames_subdir",
        type=str,
        default="frames",
        help="Name of the subfolder inside each dicom dir that contains frames",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info when frames are missing / before querying the model",
    )

    args = parser.parse_args()

    if not args.base_path.exists():
        raise FileNotFoundError(args.base_path)

    # Discover INNER dicom uid dirs
    seq_dirs = discover_sequence_dirs(args.base_path, args.frames_subdir)

    if args.limit is not None:
        seq_dirs = seq_dirs[: max(0, args.limit)]

    out_path = args.out or (args.base_path / "frames_extracted_labels.csv")

    out_cols = [
        "timestamp_utc",
        "model_name",
        "outer_dir",
        "dicom_dir",
        "sequence_path",
        "num_frames_found",
        "num_frames_sent",
        "question",
        "answer",
        "confidence",
        "evidence",
        "notes",
    ]
    ensure_csv_header(out_path, out_cols)

    total_tasks = len(seq_dirs) * len(QUESTIONS)

    print(f"Discovered {len(seq_dirs)} DICOM sequence directories (inner dirs).")
    print(f"Total questions: {total_tasks}")
    print(f"Output CSV (append mode): {out_path}")
    print(f"Frames subdir: {args.frames_subdir}")
    print(f"Model: {args.model}")

    with tqdm(total=total_tasks, desc="Analyzing", unit="q") as pbar:
        for seq_dir in seq_dirs:
            outer, inner = get_outer_and_inner(seq_dir, args.base_path)

            frames = list_frame_files(seq_dir, frames_subdir=args.frames_subdir)
            selected = pick_frames(frames, args.max_frames, args.stride)

            if args.debug:
                print(
                    f"[DEBUG] outer={outer} inner={inner} | found={len(frames)} selected={len(selected)} | path={seq_dir}"
                )

            images_b64: List[str] = []
            for f in selected:
                try:
                    images_b64.append(b64_image(f))
                except Exception as e:
                    print(f"[WARN] Could not read {f}: {e}")

            frame_names = [p.name for p in selected]

            for q in QUESTIONS:
                ts = utc_now_iso()

                if not images_b64:
                    append_csv_row(
                        out_path,
                        {
                            "timestamp_utc": ts,
                            "model_name": args.model,
                            "outer_dir": outer,
                            "dicom_dir": inner,
                            "sequence_path": str(seq_dir),
                            "num_frames_found": len(frames),
                            "num_frames_sent": 0,
                            "question": q,
                            "answer": "Not stated",
                            "confidence": 0,
                            "evidence": "[]",
                            "notes": "No usable frames.",
                        },
                        out_cols,
                    )
                    pbar.update(1)
                    continue

                try:
                    raw = ollama_chat_with_images(
                        build_prompt(q, frame_names),
                        images_b64,
                        args.model,
                        args.url,
                        args.timeout,
                    )
                    parsed = safe_parse_json(raw)
                    if not parsed:
                        raise ValueError(f"Non-JSON model output: {raw[:200]}")

                    append_csv_row(
                        out_path,
                        {
                            "timestamp_utc": ts,
                            "model_name": args.model,
                            "outer_dir": outer,
                            "dicom_dir": inner,
                            "sequence_path": str(seq_dir),
                            "num_frames_found": len(frames),
                            "num_frames_sent": len(images_b64),
                            "question": q,
                            "answer": parsed.get("answer"),
                            "confidence": parsed.get("confidence"),
                            "evidence": json.dumps(parsed.get("evidence", [])),
                            "notes": parsed.get("notes", ""),
                        },
                        out_cols,
                    )

                except Exception as e:
                    append_csv_row(
                        out_path,
                        {
                            "timestamp_utc": ts,
                            "model_name": args.model,
                            "outer_dir": outer,
                            "dicom_dir": inner,
                            "sequence_path": str(seq_dir),
                            "num_frames_found": len(frames),
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
                if args.delay > 0:
                    time.sleep(args.delay)

    print("Done ✔ Results appended incrementally.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results are preserved.", file=sys.stderr)
        raise
