#!/usr/bin/env python3
"""
extract_labels_from_frames.py

Automates extraction of structured clinical labels from angiography image sequences
using a multimodal LLM (Qwen-2.5VL) served via Ollama.

Dataset layout assumed:
  base_path/<outer_dir>/<dicom_uid_dir>/frames/*.png

Key properties:
- Discovers INNER dicom_uid_dir folders as sequence dirs
- Reads frames ONLY per dicom directory
- Appends results row-by-row to a persistent CSV
- Adds timestamp + model name to every row
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

# -----------------------------
# Questions
# -----------------------------
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
DEFAULT_OUTPUT_ROOT_SUFFIX = "_Output"

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL_NAME = "qwen3-vl:32b"
DEFAULT_TIMEOUT_S = 180

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# -----------------------------
# Prompt
# -----------------------------
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
# CSV helpers (append-only)
# -----------------------------
def ensure_csv_header(out_path: Path, columns: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv_row(out_path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    ensure_csv_header(out_path, columns)
    ordered = {c: row.get(c) for c in columns}
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(ordered)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

# -----------------------------
# Sequence discovery
# -----------------------------
def _has_images_in_dir(d: Path) -> bool:
    try:
        return any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTS
            for p in d.iterdir()
        )
    except Exception:
        return False


def discover_sequence_dirs(base_path: Path, frames_subdir: str) -> List[Path]:
    seq_dirs = []
    seen = set()

    for frames_dir in base_path.rglob(frames_subdir):
        if not frames_dir.is_dir():
            continue
        if not _has_images_in_dir(frames_dir):
            continue

        seq_dir = frames_dir.parent
        key = seq_dir.resolve()
        if key not in seen:
            seen.add(key)
            seq_dirs.append(seq_dir)

    seq_dirs.sort(key=lambda p: p.as_posix())
    return seq_dirs


def get_outer_and_inner(seq_dir: Path, base_path: Path) -> Tuple[str, str]:
    inner = seq_dir.name
    try:
        rel = seq_dir.relative_to(base_path)
        outer = rel.parts[0] if len(rel.parts) >= 2 else ""
    except Exception:
        outer = seq_dir.parent.name
    return outer, inner

# -----------------------------
# Frame utilities
# -----------------------------
def list_frame_files(seq_dir: Path, frames_subdir: str) -> List[Path]:
    frames_dir = seq_dir / frames_subdir

    if frames_dir.exists():
        frames = [
            p for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if frames:
            return sorted(frames, key=lambda p: p.name)

    frames = [
        p for p in seq_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(frames, key=lambda p: p.as_posix())


def pick_frames(frames: List[Path], max_frames: int, stride: int) -> List[Path]:
    stride = max(1, stride)
    sampled = frames[::stride]

    if max_frames and len(sampled) > max_frames:
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
    names = "\n".join(frame_names) if frame_names else "(none)"
    return BASE_PROMPT.format(QUESTION=question) + f"\nFRAME FILENAMES\n{names}\n"


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
    return None


def ollama_chat_with_images(prompt, images_b64, model, url, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--frames_subdir", type=str, default="frames")
    parser.add_argument("--max_frames", type=int, default=24)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    seq_dirs = discover_sequence_dirs(args.base_path, args.frames_subdir)
    if args.limit:
        seq_dirs = seq_dirs[:args.limit]

    # ✅ OUTPUT ROOT FIX
    output_root = args.base_path.parent / (args.base_path.name + DEFAULT_OUTPUT_ROOT_SUFFIX)
    output_root.mkdir(parents=True, exist_ok=True)

    out_path = output_root / "frames_extracted_labels.csv"

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

    with tqdm(total=len(seq_dirs) * len(QUESTIONS), desc="Analyzing") as pbar:
        for seq_dir in seq_dirs:
            outer, inner = get_outer_and_inner(seq_dir, args.base_path)
            frames = list_frame_files(seq_dir, args.frames_subdir)
            selected = pick_frames(frames, args.max_frames, args.stride)

            images_b64 = [b64_image(p) for p in selected]
            frame_names = [p.name for p in selected]

            for q in QUESTIONS:
                ts = utc_now_iso()

                if not images_b64:
                    append_csv_row(out_path, {
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
                        "notes": "No usable frames",
                    }, out_cols)
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
                    parsed = safe_parse_json(raw) or {}

                    append_csv_row(out_path, {
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
                    }, out_cols)

                except Exception as e:
                    append_csv_row(out_path, {
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
                        "notes": str(e)[:200],
                    }, out_cols)

                pbar.update(1)
                if args.delay:
                    time.sleep(args.delay)

    print("Done ✔ CSV appended incrementally.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted — partial results preserved.", file=sys.stderr)
        raise
