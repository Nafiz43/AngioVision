#!/usr/bin/env python3
"""
extract_labels_from_mosaics.py

Stage 2 only: Discover INNER sequence dirs (same rule as stage 1),
read an EXISTING mosaic image from each sequence dir, and query an LLM (Ollama)
for each anatomical-level question. Appends results incrementally to a CSV.

This script does NOT create mosaics. Use creating_mosaics.py first (or ensure mosaics exist).
"""

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# -----------------------------
# Questions (keep local or import from your utils/questions.py)
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
# Mosaic + prompt helpers
# -----------------------------
def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


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


@dataclass
class SequenceMosaicInfo:
    seq_dir: Path
    seq_rel: str
    mosaic_path: Path
    mosaic_ok: bool
    mosaic_error: Optional[str] = None


def load_sequence_mosaics(
    seq_dirs: List[Path],
    base_path: Path,
    mosaic_name: str,
    debug: bool,
) -> List[SequenceMosaicInfo]:
    infos: List[SequenceMosaicInfo] = []
    for seq_dir in seq_dirs:
        seq_rel = seq_dir.relative_to(base_path).as_posix()
        mosaic_path = seq_dir / mosaic_name
        if mosaic_path.exists() and mosaic_path.is_file():
            infos.append(SequenceMosaicInfo(seq_dir, seq_rel, mosaic_path, True, None))
        else:
            err = f"Missing mosaic: expected {mosaic_path.name} in {seq_rel}"
            if debug:
                print(f"[DEBUG] {err}")
            infos.append(SequenceMosaicInfo(seq_dir, seq_rel, mosaic_path, False, err))
    return infos


# -----------------------------
# Main LLM loop (mosaics only)
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
    total_tasks = len(infos) * len(QUESTIONS)

    with tqdm(total=total_tasks, desc="Analyzing mosaics", unit="q") as pbar:
        for info in infos:
            images_b64: List[str] = []
            if info.mosaic_ok:
                try:
                    images_b64 = [b64_image(info.mosaic_path)]
                except Exception as e:
                    images_b64 = []
                    if debug:
                        print(f"[DEBUG] Could not read {info.mosaic_path}: {str(e)[:300]}")

            # NOTE: stage-2 only doesn't inherently know frame filenames;
            # if you want them here, you can store them during stage-1 in a sidecar file.
            frame_names: List[str] = []

            for q in QUESTIONS:
                if not images_b64:
                    append_csv_row(
                        out_path,
                        {
                            "sequence_dir": info.seq_rel,
                            "mosaic_file": info.mosaic_path.name,
                            "question": q,
                            "answer": "Not stated",
                            "confidence": 0,
                            "evidence": "[]",
                            "notes": info.mosaic_error or "No usable mosaic image.",
                        },
                        out_cols,
                    )
                    pbar.update(1)
                    continue

                try:
                    raw = ollama_chat_with_images(
                        build_prompt(q, frame_names, info.mosaic_path.name),
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
                            "mosaic_file": info.mosaic_path.name,
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
                            "mosaic_file": info.mosaic_path.name,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ollama on existing per-sequence mosaics.")
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--out", type=Path, default=None)

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--delay", type=float, default=0.0)

    parser.add_argument("--frames_subdir", type=str, default="frames")
    parser.add_argument("--mosaic_name", type=str, default="mosaic.png")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if not args.base_path.exists():
        raise FileNotFoundError(args.base_path)

    seq_dirs = find_sequence_dirs(args.base_path, frames_subdir=args.frames_subdir)
    if args.limit is not None:
        seq_dirs = seq_dirs[: max(0, args.limit)]

    infos = load_sequence_mosaics(
        seq_dirs=seq_dirs,
        base_path=args.base_path,
        mosaic_name=args.mosaic_name,
        debug=args.debug,
    )

    out_path = args.out or (args.base_path / "mosaics_extracted_labels.csv")
    out_cols = [
        "sequence_dir",
        "mosaic_file",
        "question",
        "answer",
        "confidence",
        "evidence",
        "notes",
    ]
    ensure_csv_header(out_path, out_cols)

    ok = sum(1 for i in infos if i.mosaic_ok)
    print(f"Discovered {len(seq_dirs)} sequence (inner) directories")
    print(f"Mosaics found: {ok}/{len(infos)}")
    print(f"Output CSV: {out_path}")

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
