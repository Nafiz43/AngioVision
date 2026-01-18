#!/usr/bin/env python3
"""
extract_sequences_from_reports.py

Reads a CSV containing consolidated radiology/IR reports and, for each report,
asks an LLM (via Ollama) to:
  - estimate number of sequences/runs described
  - partition the report into sequence-level verbatim chunks
  - provide rationale + confidence per chunk

Writes ONE JSON file per report into a new output directory.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from tqdm import tqdm

# -----------------------------
# Defaults / Configuration
# -----------------------------
DEFAULT_CSV_PATH = Path("/data/Deep_Angiography/Reports/Report_List_v01_01.csv")
REPORT_COL = "radrpt"

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL_NAME = "llama3:8b"
DEFAULT_TIMEOUT_S = 180

# New prompt for sequence extraction
BASE_PROMPT = """You are given a single radiology/angiography procedure report (free text). The report is a consolidated narrative built from findings across multiple image sequences. Each sequence corresponds to one DICOM sequence (one “run”), and the report may describe findings sequence-by-sequence.

TASK
1) Identify how many distinct sequences (“runs”) are described in the report.
2) Partition the report into sequence-level chunks.
3) For each sequence, output ONLY the exact wording copied from the original report (verbatim). Do not paraphrase, rewrite, or normalize. Preserve punctuation and line breaks as much as possible.
4) For each sequence chunk, provide a rationale explaining why that text belongs to that sequence.

IMPORTANT RULES
- Use only information present in the report. Do not invent sequence numbers, findings, or timing.
- If the report explicitly labels sequences (e.g., “Run 1,” “Sequence 2,” “AP view,” “DSA run,” “Series,” “Injection,” etc.), follow those labels.
- If the report does NOT explicitly label sequences, infer boundaries using textual cues such as:
  • transitions (“Next…”, “Subsequently…”, “Then…”, “Additional run…”, “A second angiogram…”)  
  • changes in target vessel/anatomic territory (e.g., switching from ICA to ECA, left to right, different artery name)
  • changes in projection/view (AP, lateral, oblique), phase (arterial/venous), or technique (catheter position, injection site)
  • separate angiograms described as distinct events
- If boundaries are ambiguous, keep larger chunks and mark them as uncertain. Do NOT over-split.
- Every character in the extracted text must appear in the original report (verbatim). No added words.

OUTPUT FORMAT (STRICT JSON ONLY)
Return a single JSON object with this schema:

{{
  "num_sequences_estimate": <integer or null>,
  "sequences": [
    {{
      "sequence_id": "Sequence 1",
      "verbatim_text": "<direct copy-paste from report>",
      "rationale": [
        "<brief evidence point 1>",
        "<brief evidence point 2>"
      ],
      "confidence": "high|medium|low"
    }}
  ],
  "notes": [
    "List any ambiguities, overlaps, or places where the report does not clearly separate sequences."
  ]
}}

NOW PROCESS THIS REPORT:
<<<REPORT_TEXT
{REPORT_TEXT}
REPORT_TEXT>>>
"""



# -----------------------------
# Helpers
# -----------------------------
def build_prompt(report_text: str) -> str:
    return BASE_PROMPT.format(REPORT_TEXT=report_text)


def ollama_chat(prompt: str, model: str, url: str, timeout_s: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON even if the model wrapped it with extra text."""
    if not text:
        return None

    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting the largest JSON object
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def sanitize_filename(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]", "", s)
    return s[:max_len] if s else "report"


def choose_output_name(row: pd.Series, idx: int) -> str:
    """
    Use a stable identifier if available; otherwise fall back to row index.
    You can add more candidate columns here if your CSV has them.
    """
    candidate_cols = ["study_id", "StudyID", "accession", "AccessionNumber", "mrn", "MRN", "patient_id", "PatientID"]
    for c in candidate_cols:
        if c in row.index:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                return sanitize_filename(v)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                return sanitize_filename(str(v))

    return f"row_{idx:06d}"


def validate_sequence_json(obj: Dict[str, Any]) -> bool:
    """Light validation of the expected schema."""
    if not isinstance(obj, dict):
        return False
    if "sequences" not in obj or not isinstance(obj["sequences"], list):
        return False
    # Allow empty sequences list, but still must exist.
    return True


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to write one JSON per report (default: <csv>_sequences_json/)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--delay", type=float, default=0.0)

    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)

    if REPORT_COL not in df.columns:
        raise ValueError(f"Missing column: {REPORT_COL}")

    if args.limit is not None:
        df = df.iloc[: args.limit]

    out_dir = args.out_dir or args.csv.with_name(f"{args.csv.stem}_sequences_json")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(df)} reports")
    print(f"Output directory: {out_dir}")
    print(f"Model: {args.model}")
    print(f"Ollama URL: {args.url}")

    with tqdm(total=len(df), desc="Extracting sequences", unit="report") as pbar:
        for idx, row in df.iterrows():
            report = row.get(REPORT_COL)
            base_name = choose_output_name(row, idx)
            out_path = out_dir / f"{base_name}.json"

            # Skip if already exists (optional behavior; comment out if undesired)
            # if out_path.exists():
            #     pbar.update(1)
            #     continue

            if not isinstance(report, str) or not report.strip():
                # Save a minimal JSON so every row still produces a file
                minimal = {
                    "row_index": int(idx),
                    "error": "Empty or missing report",
                    "num_sequences_estimate": None,
                    "sequences": [],
                    "notes": ["Empty or missing report"],
                }
                out_path.write_text(json.dumps(minimal, indent=2), encoding="utf-8")
                pbar.update(1)
                continue

            try:
                prompt = build_prompt(report)
                raw = ollama_chat(
                    prompt=prompt,
                    model=args.model,
                    url=args.url,
                    timeout_s=args.timeout,
                )
                parsed = safe_parse_json(raw)
                if not parsed or not validate_sequence_json(parsed):
                    raise ValueError("Invalid JSON or missing expected fields")

                # Attach bookkeeping info (doesn't change extracted content)
                parsed["_meta"] = {
                    "row_index": int(idx),
                    "source_csv": str(args.csv),
                    "report_col": REPORT_COL,
                    "model": args.model,
                }

                out_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

            except Exception as e:
                err = {
                    "row_index": int(idx),
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "raw_model_output": raw if "raw" in locals() else "",
                    "num_sequences_estimate": None,
                    "sequences": [],
                    "notes": ["Model call failed or returned invalid JSON."],
                }
                out_path.write_text(json.dumps(err, indent=2), encoding="utf-8")

            pbar.update(1)
            if args.delay > 0:
                time.sleep(args.delay)

    print("Done ✔ One JSON file per report has been written.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — any JSON files already written are preserved.", file=sys.stderr)
        raise
