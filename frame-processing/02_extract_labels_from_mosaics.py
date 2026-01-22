#!/usr/bin/env python3
"""
extract_labels_from_mosaics.py

Stage 2 only: Discover INNER sequence dirs (same rule as stage 1),
read an EXISTING mosaic image from each sequence dir, and query an LLM (Ollama)
for each anatomical-level question.

CSV behavior (FINAL)
- Adds: Timestamp, Model Name
- Removes: mosaic_file column
- Appends row-by-row (never rewrites the full file)
- If CSV exists, new rows are appended
- Output directory is ALWAYS:
    <base_path>_Output/
"""

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
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
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

# Default model if user does not pass --model
DEFAULT_MODEL_NAME = "qwen3-vl:32b"
# "qwen3-vl:32b"

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
3) Cite frames by filename when possible.

OUTPUT FORMAT (JSON ONLY)
Return:
- answer
- confidence (0–100)
- evidence (≤3 short cues)
- notes
"""


# -----------------------------
# CSV helpers (append-only)
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
# Directory discovery
# -----------------------------
def find_sequence_dirs(base_path: Path, frames_subdir: str) -> List[Path]:
    seq_dirs: List[Path] = []
    for d in base_path.rglob("*"):
        if not d.is_dir():
            continue
        frames_dir = d / frames_subdir
        if not frames_dir.exists():
            continue
        if any(p.suffix.lower() in IMAGE_EXTS for p in frames_dir.iterdir()):
            seq_dirs.append(d)
    return sorted(seq_dirs, key=lambda p: p.as_posix())


# -----------------------------
# Helpers
# -----------------------------
def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def build_prompt(question: str) -> str:
    return BASE_PROMPT.format(QUESTION=question)


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text.strip())
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
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


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class SequenceMosaicInfo:
    seq_dir: Path
    seq_rel: str
    mosaic_path: Path
    ok: bool
    error: Optional[str] = None


def load_mosaics(seq_dirs, base_path, mosaic_name):
    infos = []
    for d in seq_dirs:
        rel = d.relative_to(base_path).as_posix()
        mp = d / mosaic_name
        infos.append(
            SequenceMosaicInfo(d, rel, mp, mp.exists(), None if mp.exists() else "Missing mosaic")
        )
    return infos


# -----------------------------
# Main processing loop
# -----------------------------
def run_llm(infos, out_path, columns, model, url, timeout, delay):
    total = len(infos) * len(QUESTIONS)

    with tqdm(total=total, desc="Analyzing mosaics", unit="q") as pbar:
        for info in infos:
            images = [b64_image(info.mosaic_path)] if info.ok else []

            for q in QUESTIONS:
                row = {
                    "Timestamp": utc_timestamp(),
                    "Model Name": model,
                    "sequence_dir": info.seq_rel,
                    "question": q,
                }

                if not images:
                    row.update(
                        dict(answer="Not stated", confidence=0, evidence="[]", notes=info.error)
                    )
                else:
                    try:
                        raw = ollama_chat_with_images(build_prompt(q), images, model, url, timeout)
                        parsed = safe_parse_json(raw)
                        if not parsed:
                            raise ValueError("Non-JSON response")
                        row.update(
                            dict(
                                answer=parsed.get("answer"),
                                confidence=parsed.get("confidence"),
                                evidence=json.dumps(parsed.get("evidence", [])),
                                notes=parsed.get("notes"),
                            )
                        )
                    except Exception as e:
                        row.update(
                            dict(answer="Unclear", confidence=0, evidence="[]", notes=str(e)[:200])
                        )

                append_csv_row(out_path, row, columns)
                pbar.update(1)
                if delay:
                    time.sleep(delay)


# -----------------------------
# Entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)

    # ✅ Model can be passed as an argument; defaults to qwen3-vl:32b if not provided
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)

    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--frames_subdir", default="frames")
    parser.add_argument("--mosaic_name", default="mosaic.png")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    seq_dirs = find_sequence_dirs(args.base_path, args.frames_subdir)
    if args.limit:
        seq_dirs = seq_dirs[:args.limit]
    infos = load_mosaics(seq_dirs, args.base_path, args.mosaic_name)

    # 🔥 OUTPUT PATH FIX (as requested)
    output_root = Path(f"{args.base_path}_Output")
    out_csv = output_root / "mosaics_extracted_labels.csv"

    columns = [
        "Timestamp",
        "Model Name",
        "sequence_dir",
        "question",
        "answer",
        "confidence",
        "evidence",
        "notes",
    ]

    ensure_csv_header(out_csv, columns)

    print(f"Sequences found: {len(seq_dirs)}")
    print(f"Output CSV: {out_csv}")
    print(f"Model: {args.model}")

    run_llm(
        infos=infos,
        out_path=out_csv,
        columns=columns,
        model=args.model,
        url=args.url,
        timeout=args.timeout,
        delay=args.delay,
    )

    print("Done ✔ Incremental results preserved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
