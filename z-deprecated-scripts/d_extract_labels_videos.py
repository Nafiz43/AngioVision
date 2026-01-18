# PURPOSE:
# This script iterates over subdirectories under a base path, where each subdirectory represents
# one angiography image sequence (a folder of frame images). For each sequence, it samples a subset
# of frames (using stride + max_frames), sends those frames along with a fixed list of clinical
# questions to a multimodal LLM (Qwen-2.5VL) served via Ollama, parses the model’s JSON-only answers,
# and incrementally writes one CSV row per (sequence, question) containing answer/confidence/evidence.


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


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path(
    "/data/Deep_Angiography/DICOM2Video/Code/Data_Processing/videos_out"
)
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL_NAME = "qwen2.5vl:32b"
DEFAULT_TIMEOUT_S = 180

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

QUESTIONS = [
    "Which artery is catheterized?",
    "Is variant anatomy present?",
    "Is there evidence of hemorrhage or contrast extravasation in this sequence?",
    "Is there evidence of arterial or venous dissection?",
    "Is stenosis present in any visualized vessel?",
    "Is an endovascular stent visible in this sequence?",
]

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
3) Cite frames by filename when possible.

OUTPUT FORMAT (JSON ONLY)
Return:
- answer
- confidence (0–100)
- evidence (≤3 short frame-based cues)
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
# Frame utilities
# -----------------------------
def list_frame_files(seq_dir: Path) -> List[Path]:
    frames = [
        p for p in seq_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    frames.sort(key=lambda p: p.name)
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
def build_prompt(question: str) -> str:
    return BASE_PROMPT.format(QUESTION=question)


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
            return json.loads(text[start:end + 1])
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

    # ✅ NEW: directory limit
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N frame directories (sorted by name)",
    )

    args = parser.parse_args()

    if not args.base_path.exists():
        raise FileNotFoundError(args.base_path)

    seq_dirs = sorted([p for p in args.base_path.iterdir() if p.is_dir()],
                      key=lambda p: p.name)

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

    with tqdm(total=total_tasks, desc="Analyzing", unit="q") as pbar:
        for seq_dir in seq_dirs:
            frames = list_frame_files(seq_dir)
            selected = pick_frames(frames, args.max_frames, args.stride)

            images_b64 = []
            for f in selected:
                try:
                    images_b64.append(b64_image(f))
                except Exception as e:
                    print(f"[WARN] Could not read {f}: {e}")

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
                        build_prompt(q),
                        images_b64,
                        args.model,
                        args.url,
                        args.timeout,
                    )
                    parsed = safe_parse_json(raw)
                    if not parsed:
                        raise ValueError("Non-JSON model output")

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
