#!/usr/bin/env python3
"""
extract_sequences_from_reports_using_GT.py

Reads:
  1) A consolidated reports CSV (contains free-text report in column `radrpt`
     and identifier column `Anon Acc #`)
  2) A Ground Truth (GT) CSV that contains, per `Anon Acc #`:
        - the true number of sequences
        - the true DICOM SeriesInstanceUIDs (sequence IDs)

For each report row, this script:
  - Looks up the true sequence count + ordered SeriesInstanceUID list from GT
  - Prompts an LLM (via Ollama) to extract EXACT (verbatim) text chunks from the report
    corresponding to each GT sequence, IN ORDER.
  - If the model cannot extract text for a GT sequence, it must output:
      "No text could be extracted for that sequence"
    for that sequence's verbatim_text.

Writes ONE JSON file per report.

Key guarantees in the saved JSON:
  - Uses the ACTUAL sequence count from GT (not an estimate) when GT is available
  - Includes the ACTUAL SeriesInstanceUIDs from GT
  - Ensures output has exactly N sequences (pads/truncates if needed)

If a report’s `Anon Acc #` is missing from GT:
  - Falls back to the older behavior (model estimates num + chunks),
    and marks the JSON with a note that GT was missing.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# -----------------------------
# Defaults / Configuration
# -----------------------------
DEFAULT_REPORTS_CSV_PATH = Path("/data/Deep_Angiography/Reports/Report_List_v01_01.csv")
DEFAULT_GT_CSV_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed/consolidated_metadata_GT.csv")

REPORT_COL = "radrpt"
ANON_ACC_COL = "Anon Acc #"

# --- GT columns (expected) ---
# If your GT file uses different names, update these.
GT_NUMSEQ_COL_CANDIDATES = ["Number of Sequences", "Number_of_Sequences", "num_sequences", "numSequences"]
GT_UIDS_COL_CANDIDATES = ["SeriesInstanceUIDs", "SeriesInstanceUID", "Series_Instance_UIDs"]

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL_NAME = "thewindmom/llama3-med42-8b"
DEFAULT_TIMEOUT_S = 180

# -----------------------------
# Prompts
# -----------------------------
BASE_PROMPT_WITH_GT = """You are given:
(1) A radiology/angiography procedure report (free text).
(2) Ground-truth (GT) information stating EXACTLY how many DICOM sequences (“runs”) exist for this study,
    and the ordered list of SeriesInstanceUIDs (one per sequence).

Your job is to extract, from the report, verbatim text corresponding to EACH GT sequence, in the SAME ORDER.

GROUND TRUTH
- num_sequences (N): {NUM_SEQUENCES}
- ordered SeriesInstanceUIDs:
{UID_LIST_BULLETS}

TASK
For i = 1..N (in the exact order above):
1) Extract the report text that belongs to GT Sequence i.
2) Output ONLY exact wording copied from the original report (verbatim). Do not paraphrase, rewrite, normalize, or invent.
3) If you cannot find any report text that clearly maps to GT Sequence i, set:
   verbatim_text = "No text could be extracted for that sequence"

BOUNDARY GUIDANCE (use ONLY report content)
- If the report explicitly labels runs/sequences (e.g., “Run 1”, “DSA run”, “Series”, “Injection”, “AP view”, etc.), follow that.
- If not explicitly labeled, infer boundaries using cues like:
  • transitions (“Next…”, “Subsequently…”, “Then…”, “Additional run…”, “A second angiogram…”)
  • change in vessel/anatomic territory (e.g., ICA→ECA, left→right, different artery)
  • change in projection/view (AP/lateral/oblique), phase, catheter position/injection site
- If ambiguous, DO NOT over-split. Prefer larger chunks and mark confidence low.

HARD RULES
- Every character you output in verbatim_text must appear in the report EXACTLY.
- Do not include any text that is not from the report.
- Produce EXACTLY N sequence entries (one per GT SeriesInstanceUID), even if some have no extractable text.

OUTPUT FORMAT (STRICT JSON ONLY — no extra keys, no commentary)
Return a single JSON object with this schema:

{{
  "num_sequences_actual": {NUM_SEQUENCES},
  "sequence_instance_uids": [{UID_LIST_JSON}],
  "sequences": [
    {{
      "sequence_number": 1,
      "sequence_instance_uid": "<the UID for sequence 1>",
      "verbatim_text": "<direct copy-paste from report OR 'No text could be extracted for that sequence'>",
      "rationale": [
        "<brief evidence point 1>",
        "<brief evidence point 2>"
      ],
      "confidence": "high|medium|low"
    }}
  ],
  "notes": [
    "List any ambiguities/overlaps."
  ]
}}

NOW PROCESS THIS REPORT (verbatim):
<<<REPORT_TEXT
{REPORT_TEXT}
REPORT_TEXT>>>
"""

# Fallback prompt when GT missing (keeps your original behavior, slightly tightened)
BASE_PROMPT_FALLBACK = """You are given a single radiology/angiography procedure report (free text).
The report is a consolidated narrative built from findings across multiple image sequences.
Each sequence corresponds to one DICOM sequence (one “run”).

TASK
1) Identify how many distinct sequences (“runs”) are described in the report.
2) Partition the report into sequence-level chunks.
3) For each sequence, output ONLY the exact wording copied from the original report (verbatim).
   Do not paraphrase, rewrite, or normalize. Preserve punctuation and line breaks as much as possible.
4) For each sequence chunk, provide a rationale explaining why that text belongs to that sequence.

IMPORTANT RULES
- Use only information present in the report. Do not invent sequence numbers, findings, or timing.
- If the report explicitly labels sequences (e.g., “Run 1,” “Sequence 2,” “AP view,” “DSA run,” “Series,” “Injection,” etc.), follow those labels.
- If not explicitly labeled, infer boundaries using textual cues such as transitions, vessel territory change, view/projection change, technique change.
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

    try:
        return json.loads(text)
    except Exception:
        pass

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


def normalize_scalar(v) -> Optional[str]:
    """Turn CSV cell into a JSON-friendly scalar string (or None)."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return str(v)


def choose_output_name(row: pd.Series, idx: int) -> str:
    """Prefer Anon Acc # for stable naming; else fall back to row index."""
    if ANON_ACC_COL in row.index:
        v = normalize_scalar(row.get(ANON_ACC_COL))
        if v:
            return sanitize_filename(v)
    return f"row_{idx:06d}"


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_uids_cell(uids_cell: Any) -> List[str]:
    """
    GT often stores SeriesInstanceUIDs as a comma-separated string.
    We split on commas and strip whitespace.
    """
    if uids_cell is None:
        return []
    if isinstance(uids_cell, float) and pd.isna(uids_cell):
        return []
    if isinstance(uids_cell, list):
        return [str(x).strip() for x in uids_cell if str(x).strip()]
    s = str(uids_cell).strip()
    if not s:
        return []
    # Split on comma
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def build_prompt_with_gt(report_text: str, uids: List[str]) -> str:
    n = len(uids)
    uid_list_bullets = "\n".join([f"- Sequence {i+1}: {uids[i]}" for i in range(n)])
    uid_list_json = ", ".join([json.dumps(u) for u in uids])  # quoted strings
    return BASE_PROMPT_WITH_GT.format(
        NUM_SEQUENCES=n,
        UID_LIST_BULLETS=uid_list_bullets,
        UID_LIST_JSON=uid_list_json,
        REPORT_TEXT=report_text,
    )


def build_prompt_fallback(report_text: str) -> str:
    return BASE_PROMPT_FALLBACK.format(REPORT_TEXT=report_text)


def validate_gt_style_output(obj: Dict[str, Any], expected_n: int) -> bool:
    if not isinstance(obj, dict):
        return False
    if "num_sequences_actual" not in obj:
        return False
    if "sequence_instance_uids" not in obj or not isinstance(obj["sequence_instance_uids"], list):
        return False
    if "sequences" not in obj or not isinstance(obj["sequences"], list):
        return False
    # Allow model to be imperfect; we will repair, but require basic structure.
    return True


def repair_and_force_gt_alignment(
    parsed: Dict[str, Any],
    uids: List[str],
) -> Dict[str, Any]:
    """
    Force:
      - num_sequences_actual = len(uids)
      - sequence_instance_uids = uids
      - sequences has exactly N entries ordered 1..N, each with correct UID
      - if missing/short, pad with placeholder entries
      - if long, truncate
    """
    n = len(uids)
    parsed["num_sequences_actual"] = n
    parsed["sequence_instance_uids"] = list(uids)

    seqs = parsed.get("sequences", [])
    if not isinstance(seqs, list):
        seqs = []

    repaired: List[Dict[str, Any]] = []
    for i in range(n):
        uid = uids[i]
        entry = None
        if i < len(seqs) and isinstance(seqs[i], dict):
            entry = dict(seqs[i])  # copy
        else:
            entry = {}

        entry["sequence_number"] = i + 1
        entry["sequence_instance_uid"] = uid

        vt = entry.get("verbatim_text")
        if not isinstance(vt, str) or not vt.strip():
            entry["verbatim_text"] = "No text could be extracted for that sequence"

        rat = entry.get("rationale")
        if not isinstance(rat, list):
            entry["rationale"] = []
        else:
            # coerce items to strings
            entry["rationale"] = [str(x) for x in rat][:6]

        conf = entry.get("confidence")
        if conf not in ("high", "medium", "low"):
            entry["confidence"] = "low"

        repaired.append(entry)

    parsed["sequences"] = repaired

    notes = parsed.get("notes")
    if not isinstance(notes, list):
        parsed["notes"] = []
    else:
        parsed["notes"] = [str(x) for x in notes][:50]

    return parsed


def validate_fallback_output(obj: Dict[str, Any]) -> bool:
    """Light validation of the fallback schema."""
    return isinstance(obj, dict) and "sequences" in obj and isinstance(obj["sequences"], list)


def build_gt_lookup(gt_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Any]], str, str]:
    """
    Returns:
      lookup: anon_acc -> {"uids": [...], "num_sequences": int}
      num_col: chosen num-sequences column name
      uids_col: chosen uids column name
    """
    if ANON_ACC_COL not in gt_df.columns:
        raise ValueError(f"GT CSV missing required key column '{ANON_ACC_COL}'")

    num_col = find_first_existing_column(gt_df, GT_NUMSEQ_COL_CANDIDATES)
    uids_col = find_first_existing_column(gt_df, GT_UIDS_COL_CANDIDATES)

    if uids_col is None:
        raise ValueError(f"GT CSV missing SeriesInstanceUIDs column (candidates: {GT_UIDS_COL_CANDIDATES})")

    # num_col is helpful but not strictly required if we can derive from UID list
    lookup: Dict[str, Dict[str, Any]] = {}

    for _, r in gt_df.iterrows():
        key = normalize_scalar(r.get(ANON_ACC_COL))
        if not key:
            continue
        uids = parse_uids_cell(r.get(uids_col))
        # Prefer GT count if present and valid, else derive from UID list
        n = None
        if num_col is not None:
            raw_n = r.get(num_col)
            try:
                if raw_n is not None and not (isinstance(raw_n, float) and pd.isna(raw_n)):
                    n = int(raw_n)
            except Exception:
                n = None
        if n is None:
            n = len(uids)
        # If n conflicts with parsed list length, trust the UID list (it is what you need for IDs)
        if len(uids) != n and len(uids) > 0:
            n = len(uids)

        lookup[key] = {"uids": uids, "num_sequences": n}

    return lookup, (num_col or ""), uids_col


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_csv", type=Path, default=DEFAULT_REPORTS_CSV_PATH)
    parser.add_argument("--gt_csv", type=Path, default=DEFAULT_GT_CSV_PATH)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to write one JSON per report (default: <reports_csv>_sequences_json/)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--delay", type=float, default=0.0)

    args = parser.parse_args()

    if not args.reports_csv.exists():
        raise FileNotFoundError(args.reports_csv)
    if not args.gt_csv.exists():
        raise FileNotFoundError(args.gt_csv)

    reports_df = pd.read_csv(args.reports_csv)
    gt_df = pd.read_csv(args.gt_csv)

    if REPORT_COL not in reports_df.columns:
        raise ValueError(f"Reports CSV missing required column: {REPORT_COL}")
    if ANON_ACC_COL not in reports_df.columns:
        raise ValueError(f"Reports CSV missing required key column: {ANON_ACC_COL}")

    gt_lookup, gt_num_col, gt_uids_col = build_gt_lookup(gt_df)

    if args.limit is not None:
        reports_df = reports_df.iloc[: args.limit]

    out_dir = args.out_dir or args.reports_csv.with_name(f"{args.reports_csv.stem}_sequences_json")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(reports_df)} reports")
    print(f"Reports CSV: {args.reports_csv}")
    print(f"GT CSV: {args.gt_csv}")
    print(f"GT columns used: key='{ANON_ACC_COL}', num='{gt_num_col or '[derived]'}', uids='{gt_uids_col}'")
    print(f"Output directory: {out_dir}")
    print(f"Model: {args.model}")
    print(f"Ollama URL: {args.url}")

    with tqdm(total=len(reports_df), desc="Extracting sequences", unit="report") as pbar:
        for idx, row in reports_df.iterrows():
            report_text = row.get(REPORT_COL)
            report_text = report_text if isinstance(report_text, str) else ""
            report_text = report_text.strip()

            anon_acc = normalize_scalar(row.get(ANON_ACC_COL))
            base_name = sanitize_filename(anon_acc) if anon_acc else choose_output_name(row, idx)
            out_path = out_dir / f"{base_name}.json"

            # Always write something even if report empty
            if not report_text:
                minimal = {
                    "row_index": int(idx),
                    ANON_ACC_COL: anon_acc,
                    REPORT_COL: report_text,
                    "error": "Empty or missing report",
                    "num_sequences_actual": None,
                    "sequence_instance_uids": [],
                    "sequences": [],
                    "notes": ["Empty or missing report"],
                    "_meta": {
                        "source_reports_csv": str(args.reports_csv),
                        "source_gt_csv": str(args.gt_csv),
                        "report_col": REPORT_COL,
                        "anon_acc_col": ANON_ACC_COL,
                        "model": args.model,
                    },
                }
                out_path.write_text(json.dumps(minimal, indent=2), encoding="utf-8")
                pbar.update(1)
                continue

            raw = ""
            try:
                gt_entry = gt_lookup.get(anon_acc) if anon_acc else None

                if gt_entry and gt_entry.get("uids"):
                    uids: List[str] = gt_entry["uids"]
                    prompt = build_prompt_with_gt(report_text, uids)
                    raw = ollama_chat(prompt=prompt, model=args.model, url=args.url, timeout_s=args.timeout)
                    parsed = safe_parse_json(raw)
                    if not parsed or not validate_gt_style_output(parsed, expected_n=len(uids)):
                        raise ValueError("Invalid JSON or missing expected GT fields")

                    parsed = repair_and_force_gt_alignment(parsed, uids)

                    # Include requested source fields
                    parsed[REPORT_COL] = report_text
                    parsed[ANON_ACC_COL] = anon_acc

                    parsed["_meta"] = {
                        "row_index": int(idx),
                        "source_reports_csv": str(args.reports_csv),
                        "source_gt_csv": str(args.gt_csv),
                        "report_col": REPORT_COL,
                        "anon_acc_col": ANON_ACC_COL,
                        "model": args.model,
                        "gt_num_sequences_source_col": gt_num_col or None,
                        "gt_uids_source_col": gt_uids_col,
                        "gt_lookup_status": "found",
                    }

                else:
                    # Fallback if GT missing for this Anon Acc #
                    prompt = build_prompt_fallback(report_text)
                    raw = ollama_chat(prompt=prompt, model=args.model, url=args.url, timeout_s=args.timeout)
                    parsed = safe_parse_json(raw)
                    if not parsed or not validate_fallback_output(parsed):
                        raise ValueError("Invalid JSON or missing expected fallback fields")

                    # Attach requested source fields
                    parsed[REPORT_COL] = report_text
                    parsed[ANON_ACC_COL] = anon_acc

                    # Also add explicit note that GT was missing
                    notes = parsed.get("notes")
                    if not isinstance(notes, list):
                        notes = []
                    notes.insert(0, "GT not found for this Anon Acc #; model estimate used.")
                    parsed["notes"] = notes

                    parsed["_meta"] = {
                        "row_index": int(idx),
                        "source_reports_csv": str(args.reports_csv),
                        "source_gt_csv": str(args.gt_csv),
                        "report_col": REPORT_COL,
                        "anon_acc_col": ANON_ACC_COL,
                        "model": args.model,
                        "gt_lookup_status": "missing",
                    }

                out_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

            except Exception as e:
                err = {
                    "row_index": int(idx),
                    ANON_ACC_COL: anon_acc,
                    REPORT_COL: report_text,
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "raw_model_output": raw,
                    "num_sequences_actual": None,
                    "sequence_instance_uids": [],
                    "sequences": [],
                    "notes": ["Model call failed or returned invalid JSON."],
                    "_meta": {
                        "source_reports_csv": str(args.reports_csv),
                        "source_gt_csv": str(args.gt_csv),
                        "report_col": REPORT_COL,
                        "anon_acc_col": ANON_ACC_COL,
                        "model": args.model,
                    },
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
