#!/usr/bin/env python3
"""
extract_labels_from_frames.py

Automates extraction of structured clinical labels from angiography image sequences
using a multimodal LLM (Qwen-2.5VL) served via Ollama.

Key behavior:
- Iterates over sequence directories under base_path
- Looks for frames in <sequence_dir>/frames/*.{png,jpg,...}
  - If that folder doesn't exist or is empty, falls back to recursive search under sequence_dir
- Samples representative frames (stride + max_frames)
- Asks a fixed set of clinical questions
- Requires strict JSON responses
- Appends results incrementally to a CSV (fault-tolerant)
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm
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
# Frame utilities (UPDATED)
# -----------------------------
def list_frame_files(seq_dir: Path, frames_subdir: str = "frames") -> List[Path]:
    """
    Prefer frames in seq_dir/<frames_subdir>/*.{png,jpg,...}.
    Fall back to recursive search if that folder doesn't exist or is empty.
    """
    frames_dir = seq_dir / frames_subdir

    # 1) Preferred: exactly seq_dir/frames/*
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
# Ollama helpers
# -----------------------------
def build_prompt(question: str, frame_names: List[str]) -> str:
    """
    Include frame filenames so the model can cite them in evidence.
    """
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

    # Try to salvage a JSON object embedded in extra text
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

    # NEW: directory limit
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N frame directories (sorted by name)",
    )

    # NEW: frames subdir (default 'frames')
    parser.add_argument(
        "--frames_subdir",
        type=str,
        default="frames",
        help="Name of the subfolder inside each sequence_dir that contains frames",
    )

    # NEW: optional debug prints
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info when frames are missing / before querying the model",
    )

    args = parser.parse_args()

    if not args.base_path.exists():
        raise FileNotFoundError(args.base_path)

    # Each direct child directory of base_path is treated as a "sequence directory"
    seq_dirs = sorted(
        [p for p in args.base_path.iterdir() if p.is_dir()], key=lambda p: p.name
    )

    if args.limit is not None:
        seq_dirs = seq_dirs[: max(0, args.limit)]

    out_path = args.out or (args.base_path / "frames_extracted_labels.csv")

    out_cols = [
        "sequence_dir",
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

    print(f"Processing {len(seq_dirs)} directories")
    print(f"Total questions: {total_tasks}")
    print(f"Output CSV: {out_path}")
    print(f"Frames subdir: {args.frames_subdir}")

    with tqdm(total=total_tasks, desc="Analyzing", unit="q") as pbar:
        for seq_dir in seq_dirs:
            frames = list_frame_files(seq_dir, frames_subdir=args.frames_subdir)
            selected = pick_frames(frames, args.max_frames, args.stride)

            if args.debug:
                print(
                    f"[DEBUG] {seq_dir.name}: found {len(frames)} frames, selected {len(selected)}"
                )
                if len(frames) == 0:
                    try:
                        contents = [p.name for p in seq_dir.iterdir()]
                    except Exception:
                        contents = ["(unable to list)"]
                    print(f"[DEBUG] {seq_dir.name} contents: {contents[:30]}")

            images_b64: List[str] = []
            for f in selected:
                try:
                    images_b64.append(b64_image(f))
                except Exception as e:
                    print(f"[WARN] Could not read {f}: {e}")

            frame_names = [p.name for p in selected]

            for q in QUESTIONS:
                if not images_b64:
                    append_csv_row(
                        out_path,
                        {
                            "sequence_dir": seq_dir.name,
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
                            "sequence_dir": seq_dir.name,
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
                            "sequence_dir": seq_dir.name,
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

    print("Done ✔ Results saved incrementally.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results are preserved.", file=sys.stderr)
        raise
