#!/usr/bin/env python3
"""
extract_labels_from_validation_mosaics.py

Goal
- Read the validation CSV (row-by-row).
- For each row, locate the mosaic for that row's UID under:
    /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed
- Ask the row's *own* Question to the mosaic (ONLY for validated UID+Question pairs).
- Append results incrementally to an output CSV (never rewrite).

Key features
- Works even if UID directories are nested (falls back to rglob search).
- Append-only CSV with resume support (skips rows already processed using input_row_index).
- Keeps ALL original validation columns + adds model output columns.

Example
python extract_labels_from_validation_mosaics.py \
  --base_path /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed \
  --in_csv /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/Test_Data_2026_02_01_v01.csv \
  --model qwen3-vl:32b
"""

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path(
    "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed"
)
DEFAULT_IN_CSV = Path(
    "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/Test_Data_2026_02_01_v01.csv"
)

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL_NAME = "qwen3-vl:8b"
DEFAULT_TIMEOUT_S = 2400

# Mosaic filename (inside each UID directory)
DEFAULT_MOSAIC_NAME = "mosaic.png"

# Columns in your validation CSV
DEFAULT_UID_COL = "UID"
DEFAULT_QUESTION_COL = "Question"

# If your mosaic isn't directly in UID dir, you can set this (rare).
# e.g., if mosaic is in <uid_dir>/some_subdir/mosaic.png
DEFAULT_MOSAIC_RELATIVE_DIR = ""  # empty => look in UID dir first

# -----------------------------
# Prompt
# -----------------------------
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
# Helpers
# -----------------------------
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def build_prompt(question: str) -> str:
    return BASE_PROMPT.format(QUESTION=question)


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Accepts either pure JSON or JSON embedded in extra text.
    """
    try:
        return json.loads(text.strip())
    except Exception:
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
    timeout: int,
) -> str:
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
# CSV append-only helpers
# -----------------------------
def ensure_csv_header(out_path: Path, columns: List[str]) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(out_path, index=False)


def append_csv_row(out_path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    ordered = {c: row.get(c) for c in columns}
    pd.DataFrame([ordered]).to_csv(out_path, mode="a", header=False, index=False)


def load_already_done_indices(out_path: Path, index_col: str = "input_row_index") -> Set[int]:
    """
    Resume behavior:
    - If output CSV exists, read only the input_row_index column and skip those rows.
    """
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()

    try:
        done = pd.read_csv(out_path, usecols=[index_col])
        # Drop NaNs safely
        vals = done[index_col].dropna().astype(int).tolist()
        return set(vals)
    except Exception:
        # If output is malformed or missing column, do not skip anything
        return set()


# -----------------------------
# UID -> mosaic resolution
# -----------------------------
@dataclass
class ResolvedMosaic:
    uid: str
    seq_dir: Optional[Path]
    mosaic_path: Optional[Path]
    ok: bool
    error: Optional[str] = None


def resolve_uid_dir(base_path: Path, uid: str) -> Optional[Path]:
    """
    Try common patterns:
    1) base_path/UID
    2) search nested: first directory whose name == UID
    """
    direct = base_path / uid
    if direct.exists() and direct.is_dir():
        return direct

    # fallback: search nested (can be expensive; but OK for validation subset)
    try:
        for p in base_path.rglob("*"):
            if p.is_dir() and p.name == uid:
                return p
    except Exception:
        return None

    return None


def resolve_mosaic_for_uid(
    base_path: Path,
    uid: str,
    mosaic_name: str,
    mosaic_relative_dir: str = "",
) -> ResolvedMosaic:
    uid_dir = resolve_uid_dir(base_path, uid)
    if not uid_dir:
        return ResolvedMosaic(uid=uid, seq_dir=None, mosaic_path=None, ok=False, error="UID directory not found")

    # primary expected location
    candidate = uid_dir / mosaic_relative_dir / mosaic_name if mosaic_relative_dir else uid_dir / mosaic_name
    if candidate.exists():
        return ResolvedMosaic(uid=uid, seq_dir=uid_dir, mosaic_path=candidate, ok=True, error=None)

    # fallback: search within uid_dir for mosaic_name
    try:
        hits = list(uid_dir.rglob(mosaic_name))
        if hits:
            return ResolvedMosaic(uid=uid, seq_dir=uid_dir, mosaic_path=hits[0], ok=True, error=None)
    except Exception:
        pass

    return ResolvedMosaic(uid=uid, seq_dir=uid_dir, mosaic_path=None, ok=False, error="Missing mosaic.png")


# -----------------------------
# Main processing loop
# -----------------------------
def run_validation_rows(
    df: pd.DataFrame,
    base_path: Path,
    uid_col: str,
    question_col: str,
    out_csv: Path,
    columns: List[str],
    model: str,
    url: str,
    timeout: int,
    delay: float,
    mosaic_name: str,
    mosaic_relative_dir: str,
    limit: Optional[int],
    skip_done: bool,
) -> None:
    ensure_csv_header(out_csv, columns)

    done_indices = load_already_done_indices(out_csv) if skip_done else set()

    # Validate required columns
    if uid_col not in df.columns:
        raise ValueError(f"Input CSV missing UID column: {uid_col}")
    if question_col not in df.columns:
        raise ValueError(f"Input CSV missing Question column: {question_col}")

    # TQDM over row count (respect limit)
    total_rows = len(df) if limit is None else min(len(df), limit)

    with tqdm(total=total_rows, desc="Validating mosaics", unit="row") as pbar:
        processed = 0

        for idx, row_in in df.iterrows():
            if limit is not None and processed >= limit:
                break

            input_row_index = int(idx)

            # Resume skip
            if skip_done and input_row_index in done_indices:
                pbar.update(1)
                processed += 1
                continue

            uid = str(row_in.get(uid_col, "")).strip()
            question = str(row_in.get(question_col, "")).strip()

            out_row: Dict[str, Any] = {}

            # Copy ALL original validation columns into output row
            for c in df.columns:
                out_row[c] = row_in.get(c)

            # Add our metadata
            out_row["Timestamp"] = utc_timestamp()
            out_row["Model Name"] = model
            out_row["input_row_index"] = input_row_index
            out_row["uid"] = uid
            out_row["question"] = question

            # Basic guards
            if not uid or uid.lower() == "nan":
                out_row.update(
                    dict(
                        sequence_dir="",
                        mosaic_path="",
                        answer="Not stated",
                        confidence=0,
                        evidence="[]",
                        notes="Missing UID in this row",
                    )
                )
                append_csv_row(out_csv, out_row, columns)
                pbar.update(1)
                processed += 1
                continue

            if not question or question.lower() == "nan":
                out_row.update(
                    dict(
                        sequence_dir="",
                        mosaic_path="",
                        answer="Not stated",
                        confidence=0,
                        evidence="[]",
                        notes="Missing Question in this row",
                    )
                )
                append_csv_row(out_csv, out_row, columns)
                pbar.update(1)
                processed += 1
                continue

            # Resolve mosaic
            resolved = resolve_mosaic_for_uid(
                base_path=base_path,
                uid=uid,
                mosaic_name=mosaic_name,
                mosaic_relative_dir=mosaic_relative_dir,
            )

            if not resolved.ok or not resolved.mosaic_path or not resolved.seq_dir:
                out_row.update(
                    dict(
                        sequence_dir=str(resolved.seq_dir) if resolved.seq_dir else "",
                        mosaic_path="",
                        answer="Not stated",
                        confidence=0,
                        evidence="[]",
                        notes=resolved.error or "Could not resolve mosaic",
                    )
                )
                append_csv_row(out_csv, out_row, columns)
                pbar.update(1)
                processed += 1
                continue

            out_row["sequence_dir"] = resolved.seq_dir.relative_to(base_path).as_posix() if resolved.seq_dir else ""
            out_row["mosaic_path"] = str(resolved.mosaic_path)

            # Query model
            try:
                images = [b64_image(resolved.mosaic_path)]
                raw = ollama_chat_with_images(build_prompt(question), images, model, url, timeout)
                parsed = safe_parse_json(raw)
                if not parsed:
                    raise ValueError("Non-JSON response")

                out_row.update(
                    dict(
                        answer=parsed.get("answer"),
                        confidence=parsed.get("confidence"),
                        evidence=json.dumps(parsed.get("evidence", [])),
                        notes=parsed.get("notes"),
                    )
                )
            except Exception as e:
                out_row.update(
                    dict(
                        answer="Unclear",
                        confidence=0,
                        evidence="[]",
                        notes=str(e)[:200],
                    )
                )

            append_csv_row(out_csv, out_row, columns)

            if delay:
                time.sleep(delay)

            pbar.update(1)
            processed += 1


# -----------------------------
# Entrypoint
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--in_csv", type=Path, default=DEFAULT_IN_CSV)

    parser.add_argument("--uid_col", type=str, default=DEFAULT_UID_COL)
    parser.add_argument("--question_col", type=str, default=DEFAULT_QUESTION_COL)

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--delay", type=float, default=0.0)

    parser.add_argument("--mosaic_name", type=str, default=DEFAULT_MOSAIC_NAME)
    parser.add_argument("--mosaic_relative_dir", type=str, default=DEFAULT_MOSAIC_RELATIVE_DIR)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true", help="Do NOT skip already-processed rows")

    args = parser.parse_args()

    if not args.in_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {args.in_csv}")
    if not args.base_path.exists():
        raise FileNotFoundError(f"Base path not found: {args.base_path}")

    df = pd.read_csv(args.in_csv)

    # Output CSV location (same convention you used)
    output_root = Path(f"{args.base_path}_Output")
    out_csv = output_root / "validation_mosaics_llm_labels.csv"

    # Output columns = all input columns + our added columns
    added_cols = [
        "Timestamp",
        "Model Name",
        "input_row_index",
        "uid",
        "question",
        "sequence_dir",
        "mosaic_path",
        "answer",
        "confidence",
        "evidence",
        "notes",
    ]
    columns = list(df.columns) + [c for c in added_cols if c not in df.columns]

    print(f"Input CSV: {args.in_csv}")
    print(f"Rows in input: {len(df)}")
    print(f"Base path: {args.base_path}")
    print(f"Output CSV: {out_csv}")
    print(f"Model: {args.model}")
    print(f"UID col: {args.uid_col} | Question col: {args.question_col}")
    print(f"Mosaic: {args.mosaic_name} (relative dir: '{args.mosaic_relative_dir or ''}')")
    print(f"Resume enabled: {not args.no_resume}")

    run_validation_rows(
        df=df,
        base_path=args.base_path,
        uid_col=args.uid_col,
        question_col=args.question_col,
        out_csv=out_csv,
        columns=columns,
        model=args.model,
        url=args.url,
        timeout=args.timeout,
        delay=args.delay,
        mosaic_name=args.mosaic_name,
        mosaic_relative_dir=args.mosaic_relative_dir,
        limit=args.limit,
        skip_done=(not args.no_resume),
    )

    print("Done ✔ Incremental results preserved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
