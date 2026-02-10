#!/usr/bin/env python3
"""
extract_frame_text_then_aggregate.py

Stage 2: For each sequence directory containing frames/, process EACH frame image
with a video-language model (via Ollama) to extract visual information as text,
then aggregate all per-frame outputs into a final per-sequence summary.

REQUIREMENTS MET
1) Pass each frame to a VLM: qwen-3-vl:8b (default)
2) Ask model to extract visual info from the image in textual format; treat as sequence;
   process each image independently.
3) After each image is processed, run another LLM call to aggregate all textual outputs.
4) Save all intermediate (per-frame) responses AND final aggregated response in CSV.
5) Create a separate CSV per sequence.

OUTPUT LAYOUT
<base_path>_Output/<sequence_rel>/sequence_summary.csv

CSV IS APPEND-ONLY PER SEQUENCE (never rewrites; preserves partial progress)

PROGRESS/LOADER
- Outer tqdm: sequences
- Inner tqdm: frames per sequence
"""

import argparse
import base64
import json
import re
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
# Defaults
# -----------------------------
# ✅ Updated default base path to your Validation dataset location
DEFAULT_BASE_PATH = Path(
    "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed"
)
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL_NAME = "qwen3-vl:8b"
DEFAULT_TIMEOUT_S = 180

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FRAMES_SUBDIR_DEFAULT = "frames"

# -----------------------------
# Prompts
# -----------------------------
FRAME_PROMPT = """ROLE
You are a meticulous clinical VISUAL information extraction engine for interventional radiology angiography FRAME images.

WHY THIS MATTERS
Your output will be used to build a research-grade labeled dataset. High precision is more important than guessing.

SOURCE OF TRUTH
Use ONLY the provided image. Do not use outside medical knowledge.

IMPORTANT CONTEXT
- This image is ONE frame from a longer angiography SEQUENCE.
- Process THIS frame independently. Do NOT assume what happens in other frames.
- If something is unclear in THIS frame, say so.

STRICT RULES
1) Do not guess. If unclear, return "Unclear" or "Not stated".
2) If the image quality prevents assessment, say so.
3) Be concise and factual.

OUTPUT FORMAT (JSON ONLY)
Return a single JSON object with:
- frame_description: 2-5 sentences describing what is visible (devices, vessels, contrast, motion artifacts)
- salient_findings: list of up to 6 short bullets (strings)
- uncertainty: list of up to 4 items that are unclear (strings)
- confidence: integer 0-100
"""

AGGREGATION_PROMPT = """ROLE
You aggregate per-frame extraction results for an interventional radiology angiography SEQUENCE.

GOAL
Given a list of per-frame JSON outputs (each processed independently), produce a single SEQUENCE-LEVEL summary.

STRICT RULES
1) Do not invent details not supported by the per-frame outputs.
2) If frames disagree, mark as "Conflicting".
3) If evidence is insufficient, say "Unclear" / "Not stated".
4) Prefer high-precision, low-recall: it's okay to be unsure.

OUTPUT FORMAT (JSON ONLY)
Return a single JSON object with:
- sequence_summary: 3-6 sentences combining consistent info across frames
- consistent_findings: list of up to 8 short bullets (strings)
- temporal_changes: list of up to 6 items describing changes across frames (strings) or [] if none
- conflicts: list of up to 6 conflicts across frames (strings) or []
- open_questions: list of up to 6 uncertainties that remain (strings)
- overall_confidence: integer 0-100
"""

# -----------------------------
# CSV helpers (append-only)
# -----------------------------
CSV_COLUMNS = [
    "Timestamp",
    "Model Name",
    "sequence_dir",
    "row_type",          # FRAME or AGGREGATED
    "frame_index",
    "frame_file",
    "raw_response",
    "parsed_json",
    "notes",
]


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
    """
    A "sequence dir" is any directory under base_path that contains:
      - <frames_subdir>/ with at least one image file.
    """
    seq_dirs: List[Path] = []
    for d in base_path.rglob("*"):
        if not d.is_dir():
            continue
        frames_dir = d / frames_subdir
        if not frames_dir.exists() or not frames_dir.is_dir():
            continue
        if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in frames_dir.iterdir()):
            seq_dirs.append(d)
    return sorted(seq_dirs, key=lambda p: p.as_posix())


# -----------------------------
# Helpers
# -----------------------------
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(t[start : end + 1])
            except Exception:
                return None
        return None


def ollama_chat(prompt: str, model: str, url: str, timeout: int, images_b64: Optional[List[str]] = None) -> str:
    msg: Dict[str, Any] = {"role": "user", "content": prompt}
    if images_b64:
        msg["images"] = images_b64

    payload = {
        "model": model,
        "messages": [msg],
        "stream": False,
        "options": {"temperature": 0},
    }

    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def list_frame_files(frames_dir: Path) -> List[Path]:
    files = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name)


def sanitize_relpath(rel: str) -> str:
    rel = rel.replace("\\", "/")
    parts = []
    for seg in rel.split("/"):
        seg2 = re.sub(r"[^A-Za-z0-9._-]+", "_", seg).strip("_")
        parts.append(seg2 if seg2 else "_")
    return "/".join(parts)


@dataclass
class SeqInfo:
    seq_dir: Path
    seq_rel: str
    frames_dir: Path


# -----------------------------
# Main processing
# -----------------------------
def process_sequence(
    info: SeqInfo,
    out_csv: Path,
    model: str,
    url: str,
    timeout: int,
    delay: float,
    max_frames: Optional[int],
) -> None:
    ensure_csv_header(out_csv, CSV_COLUMNS)

    frame_files = list_frame_files(info.frames_dir)
    if max_frames is not None:
        frame_files = frame_files[:max_frames]

    per_frame_jsons: List[Dict[str, Any]] = []

    # ✅ Inner loader: per-frame progress for THIS sequence
    for idx, frame_path in enumerate(
        tqdm(
            frame_files,
            desc=f"Frames | {info.seq_rel}",
            unit="frame",
            leave=False,
        )
    ):
        row = {
            "Timestamp": utc_timestamp(),
            "Model Name": model,
            "sequence_dir": info.seq_rel,
            "row_type": "FRAME",
            "frame_index": idx,
            "frame_file": frame_path.name,
        }

        try:
            raw = ollama_chat(
                prompt=FRAME_PROMPT,
                model=model,
                url=url,
                timeout=timeout,
                images_b64=[b64_image(frame_path)],
            )
            parsed = safe_parse_json(raw)
            if parsed is None:
                row.update(
                    {
                        "raw_response": raw,
                        "parsed_json": "",
                        "notes": "Non-JSON response",
                    }
                )
            else:
                per_frame_jsons.append(parsed)
                row.update(
                    {
                        "raw_response": raw,
                        "parsed_json": json.dumps(parsed, ensure_ascii=False),
                        "notes": "",
                    }
                )
        except Exception as e:
            row.update(
                {
                    "raw_response": "",
                    "parsed_json": "",
                    "notes": f"ERROR: {str(e)[:200]}",
                }
            )

        append_csv_row(out_csv, row, CSV_COLUMNS)

        if delay:
            time.sleep(delay)

    # Aggregate
    agg_row = {
        "Timestamp": utc_timestamp(),
        "Model Name": model,
        "sequence_dir": info.seq_rel,
        "row_type": "AGGREGATED",
        "frame_index": "",
        "frame_file": "__AGGREGATED__",
    }

    if not per_frame_jsons:
        agg_row.update(
            {
                "raw_response": "",
                "parsed_json": "",
                "notes": "No valid per-frame JSON outputs; aggregation skipped",
            }
        )
        append_csv_row(out_csv, agg_row, CSV_COLUMNS)
        return

    try:
        frames_blob = json.dumps(per_frame_jsons, ensure_ascii=False)
        agg_input = AGGREGATION_PROMPT + "\n\nPER-FRAME OUTPUTS (JSON LIST):\n" + frames_blob

        raw_agg = ollama_chat(
            prompt=agg_input,
            model=model,
            url=url,
            timeout=timeout,
            images_b64=None,
        )
        parsed_agg = safe_parse_json(raw_agg)
        if parsed_agg is None:
            agg_row.update(
                {
                    "raw_response": raw_agg,
                    "parsed_json": "",
                    "notes": "Non-JSON aggregation response",
                }
            )
        else:
            agg_row.update(
                {
                    "raw_response": raw_agg,
                    "parsed_json": json.dumps(parsed_agg, ensure_ascii=False),
                    "notes": "",
                }
            )
    except Exception as e:
        agg_row.update(
            {
                "raw_response": "",
                "parsed_json": "",
                "notes": f"ERROR: {str(e)[:200]}",
            }
        )

    append_csv_row(out_csv, agg_row, CSV_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--frames_subdir", type=str, default=FRAMES_SUBDIR_DEFAULT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--limit_seqs", type=int, default=None, help="Process only first N sequences")
    parser.add_argument("--max_frames", type=int, default=None, help="Process only first N frames per sequence")
    args = parser.parse_args()

    seq_dirs = find_sequence_dirs(args.base_path, args.frames_subdir)
    if args.limit_seqs is not None:
        seq_dirs = seq_dirs[: args.limit_seqs]

    output_root = Path(f"{args.base_path}_Output")
    print(f"Sequences found: {len(seq_dirs)}")
    print(f"Input base_path: {args.base_path}")
    print(f"Output root: {output_root}")
    print(f"Model: {args.model}")
    print(f"Ollama URL: {args.url}")

    infos: List[SeqInfo] = []
    for d in seq_dirs:
        rel = d.relative_to(args.base_path).as_posix()
        infos.append(SeqInfo(seq_dir=d, seq_rel=rel, frames_dir=d / args.frames_subdir))

    # ✅ Outer loader: sequence progress
    with tqdm(total=len(infos), desc="Processing sequences", unit="seq") as pbar:
        for info in infos:
            safe_rel = sanitize_relpath(info.seq_rel)
            out_csv = output_root / safe_rel / "sequence_summary.csv"

            process_sequence(
                info=info,
                out_csv=out_csv,
                model=args.model,
                url=args.url,
                timeout=args.timeout,
                delay=args.delay,
                max_frames=args.max_frames,
            )

            pbar.update(1)

    print("Done ✔ Per-sequence CSVs written (append-only).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
