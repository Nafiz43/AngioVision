"""
extract_sequences_and_sequence_QA_using_GT.py

PIPELINE (two-stage, both via Ollama LLM):
1) Sequence extraction (uses Ground Truth):
   - Reads Reports CSV with columns:
       * "Anon Acc #"
       * "radrpt"
   - Reads GT CSV with columns (names can vary; see candidates below):
       * "Anon Acc #"
       * "Number of Sequences" (optional if UIDs present)
       * "SOPInstanceUID" / "SOPInstanceUIDs" (comma-separated list)
   - For each report:
       * Looks up GT-ordered SOPInstanceUIDs and N
       * Prompts LLM to extract EXACT verbatim text for each of N sequences in order
       * If a sequence can’t be extracted: verbatim_text = "No text could be extracted for that sequence"
       * Saves JSON with actual GT N and UIDs

2) Per-sequence QA (second LLM call per sequence):
   For each extracted sequence chunk, asks 5 questions.
   **UPDATED:** QA prompt now includes BOTH:
     - verbatim_text (the ONLY source for clinical findings + evidence)
     - rationale (for boundary/context only, NOT evidence)

STRICTNESS / SAFETY:
- QA findings must be based ONLY on the provided sequence verbatim_text.
- Evidence must be copied verbatim from the sequence text (or empty list if none).
- Rationale is for context only; do NOT use it as evidence.

OUTPUT:
- One JSON file per report.

USAGE EXAMPLE:
python extract_sequences_and_sequence_QA_using_GT.py \
  --reports_csv /data/Deep_Angiography/Reports/Report_List_v01_01.csv \
  --gt_csv /data/Deep_Angiography/DICOM_Sequence_Processed/consolidated_metadata_GT.csv \
  --out_dir /data/Deep_Angiography/Reports/Report_List_v01_01_sequences_json \
  --model thewindmom/llama3-med42-8b
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
DEFAULT_GT_CSV_PATH = Path("/data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_GT.csv")

REPORT_COL = "radrpt"
ANON_ACC_COL = "Anon Acc #"

# --- GT columns (expected; script will choose first match) ---
GT_NUMSEQ_COL_CANDIDATES = ["Number of Sequences", "Number_of_Sequences", "num_sequences", "numSequences"]

# ✅ Updated: prefer SOP columns, but keep Series columns as backward-compatible fallbacks
GT_UIDS_COL_CANDIDATES = [
    "SOPInstanceUIDs",
    "SOPInstanceUID",
    "SOP_Instance_UIDs",
    "SOP_Instance_UID",
    # backward-compat:
    "SeriesInstanceUIDs",
    "SeriesInstanceUID",
    "Series_Instance_UIDs",
]

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL_NAME = "thiagomoraes/medgemma-27b-it:Q4_K_S"
DEFAULT_TIMEOUT_S = 1800

# Per-sequence QA questions (in the order you requested)
QA_QUESTIONS = [
    "Is variant anatomy present?",
    "Is there evidence of hemorrhage or contrast extravasation in this sequence?",
    "Is there evidence of arterial or venous dissection?",
    "Is stenosis present in any visualized vessel?",
    "Is an endovascular stent visible in this sequence?",
]

# -----------------------------
# Prompts
# -----------------------------
BASE_PROMPT_WITH_GT = """You are given:
(1) A radiology/angiography procedure report (free text).
(2) Ground-truth (GT) information stating EXACTLY how many DICOM instances (“runs”) exist for this study,
    and the ordered list of SOPInstanceUIDs (one per item).

Your job is to extract, from the report, verbatim text corresponding to EACH GT item, in the SAME ORDER.

GROUND TRUTH
- num_sequences (N): {NUM_SEQUENCES}
- ordered SOPInstanceUIDs:
{UID_LIST_BULLETS}

TASK
For i = 1..N (in the exact order above):
1) Extract the report text that belongs to GT item i.
2) Output ONLY exact wording copied from the original report (verbatim). Do not paraphrase, rewrite, normalize, or invent.
3) If you cannot find any report text that clearly maps to GT item i, set:
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
- Produce EXACTLY N sequence entries (one per GT SOPInstanceUID), even if some have no extractable text.

OUTPUT FORMAT (STRICT JSON ONLY — no extra keys, no commentary)
Return a single JSON object with this schema:

{{
  "num_sequences_actual": {NUM_SEQUENCES},
  "sequence_instance_uids": [{UID_LIST_JSON}],
  "sequences": [
    {{
      "sequence_number": 1,
      "sequence_instance_uid": "<the UID for item 1>",
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

# Per-sequence QA prompt (SECOND LLM CALL) — UPDATED to include rationale too
BASE_PROMPT_SEQUENCE_QA = """You are given:
(1) A SINGLE extracted sequence-level chunk from a radiology/angiography report (verbatim text).
(2) The extraction rationale used to map that chunk to this sequence (context only).

You must answer the questions below using the provided sequence text and the rationale.

CRITICAL RULES
- Use ONLY the provided sequence text for clinical findings. Do NOT use outside knowledge.
- The rationale may be used ONLY to understand sequence boundaries/context, but NOT as evidence for medical findings.
- If the sequence text does not contain enough information, answer "n/a".
- Provide "evidence" as an array of SHORT verbatim snippets copied EXACTLY from the sequence text that justify your answer.
  If answer is "n/a", evidence MUST be an empty array [].
- Do not include any evidence that is not copied verbatim from the sequence text.
- Keep answers concise. Prefer "yes", "no", or "n/a" when possible.
- Confidence must be one of: "high", "medium", "low".
  * "high" only if evidence directly supports the answer clearly.
  * "medium" if evidence is suggestive but not definitive.
  * "low" if answer is "n/a" or evidence is weak/indirect.
QUESTIONS
{QUESTIONS_BULLETS}

OUTPUT FORMAT (STRICT JSON ONLY — no extra keys, no commentary)
Return a single JSON object with this schema:

{{
  "qa": [
    {{
      "question": "<one of the questions exactly>",
      "answer": "yes|no|n/a",
      "confidence": "high|medium|low",
      "evidence": ["<verbatim snippet 1>", "<verbatim snippet 2>"],
      "notes": "<short note about ambiguity or why n/a>"
    }}
  ]
}}

RATIONALE (context only; NOT evidence for findings):
<<<RATIONALE
{RATIONALE_TEXT}
RATIONALE>>>

NOW ANALYZE THIS SEQUENCE CHUNK (verbatim):
<<<SEQUENCE_TEXT
{SEQUENCE_TEXT}
SEQUENCE_TEXT>>>
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
    data = r.json()
    content = data.get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Ollama returned empty content. Response keys: {list(data.keys())}")
    return content


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
    """
    Turn CSV cell into a stable string key (or None).
    Fixes common mismatch where "12345" vs "12345.0" prevents GT lookup.
    """
    if v is None:
        return None

    if isinstance(v, float):
        if pd.isna(v):
            return None
        if float(v).is_integer():
            return str(int(v))
        return str(v).strip()

    if isinstance(v, int):
        return str(v)

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if re.fullmatch(r"\d+\.0", s):
            return s[:-2]
        return s

    s = str(v).strip()
    if not s:
        return None
    if re.fullmatch(r"\d+\.0", s):
        return s[:-2]
    return s


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
    GT often stores UIDs as a comma-separated string.
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


def build_prompt_sequence_qa(sequence_text: str, rationale: Optional[List[str]] = None) -> str:
    bullets = "\n".join([f"- {q}" for q in QA_QUESTIONS])

    rat_list = rationale if isinstance(rationale, list) else []
    rat_list = [str(x).strip() for x in rat_list if str(x).strip()]
    rationale_text = "\n".join([f"- {x}" for x in rat_list]) if rat_list else "N/A"

    return BASE_PROMPT_SEQUENCE_QA.format(
        QUESTIONS_BULLETS=bullets,
        SEQUENCE_TEXT=sequence_text,
        RATIONALE_TEXT=rationale_text,
    )


def validate_gt_style_output(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if "num_sequences_actual" not in obj:
        return False
    if "sequence_instance_uids" not in obj or not isinstance(obj["sequence_instance_uids"], list):
        return False
    if "sequences" not in obj or not isinstance(obj["sequences"], list):
        return False
    return True


def repair_and_force_gt_alignment(parsed: Dict[str, Any], uids: List[str]) -> Dict[str, Any]:
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
        entry: Dict[str, Any]
        if i < len(seqs) and isinstance(seqs[i], dict):
            entry = dict(seqs[i])
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
            entry["rationale"] = [str(x) for x in rat][:6]

        conf = entry.get("confidence")
        if conf not in ("high", "medium", "low"):
            entry["confidence"] = "low"

        # Ensure QA placeholder exists (filled later)
        if "qa" not in entry or not isinstance(entry.get("qa"), list):
            entry["qa"] = []

        repaired.append(entry)

    parsed["sequences"] = repaired

    notes = parsed.get("notes")
    if not isinstance(notes, list):
        parsed["notes"] = []
    else:
        parsed["notes"] = [str(x) for x in notes][:50]

    return parsed


def validate_fallback_output(obj: Dict[str, Any]) -> bool:
    return isinstance(obj, dict) and "sequences" in obj and isinstance(obj["sequences"], list)


def build_gt_lookup(gt_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Any]], str, str]:
    """
    Returns:
      lookup: anon_acc -> {"uids": [...], "num_sequences": int}
      num_col: chosen num-sequences column name (or "")
      uids_col: chosen uids column name
    """
    if ANON_ACC_COL not in gt_df.columns:
        raise ValueError(f"GT CSV missing required key column '{ANON_ACC_COL}'")

    num_col = find_first_existing_column(gt_df, GT_NUMSEQ_COL_CANDIDATES)
    uids_col = find_first_existing_column(gt_df, GT_UIDS_COL_CANDIDATES)

    if uids_col is None:
        raise ValueError(
            f"GT CSV missing UID list column (candidates: {GT_UIDS_COL_CANDIDATES})"
        )

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

        # If mismatch, trust UID list (you need actual IDs)
        if len(uids) != n and len(uids) > 0:
            n = len(uids)

        lookup[key] = {"uids": uids, "num_sequences": n}

    return lookup, (num_col or ""), uids_col


def validate_qa_output(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    qa = obj.get("qa")
    if not isinstance(qa, list):
        return False
    return True


def coerce_and_repair_qa(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ensures we return exactly one QA entry per question in QA_QUESTIONS order.
    If model output is missing/invalid, fills n/a.
    Also enforces field types and confidence enum.
    """
    def unknown_entry(q: str) -> Dict[str, Any]:
        return {
            "question": q,
            "answer": "n/a",
            "confidence": "low",
            "evidence": [],
            "notes": "Insufficient information in provided sequence text.",
        }

    if not obj or not isinstance(obj, dict) or not isinstance(obj.get("qa"), list):
        return [unknown_entry(q) for q in QA_QUESTIONS]

    # Build map by exact question text
    out_map: Dict[str, Dict[str, Any]] = {}
    for item in obj.get("qa", []):
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        if isinstance(q, str) and q.strip():
            out_map[q.strip()] = item

    repaired: List[Dict[str, Any]] = []
    for q in QA_QUESTIONS:
        item = out_map.get(q)
        if not isinstance(item, dict):
            repaired.append(unknown_entry(q))
            continue

        ans = item.get("answer")
        if not isinstance(ans, str) or not ans.strip():
            ans = "n/a"
        ans = ans.strip()

        conf = item.get("confidence")
        if conf not in ("high", "medium", "low"):
            conf = "low"

        ev = item.get("evidence")
        if not isinstance(ev, list):
            ev = []
        else:
            ev = [str(x) for x in ev if isinstance(x, (str, int, float))][:8]
            ev = [e for e in ev if e.strip()]

        notes = item.get("notes")
        if not isinstance(notes, str):
            notes = ""

        repaired.append(
            {
                "question": q,
                "answer": ans,
                "confidence": conf,
                "evidence": ev,
                "notes": notes.strip(),
            }
        )

    return repaired


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

    # LLM / Ollama
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)

    # Optional pacing
    parser.add_argument("--delay_between_reports", type=float, default=0.0)
    parser.add_argument("--delay_between_sequences", type=float, default=0.0)

    # If you ever want to disable QA stage
    parser.add_argument("--skip_qa", action="store_true", help="If set, only extract sequences (no QA).")

    # Force tqdm to show (helpful for non-TTY environments / log capture)
    parser.add_argument("--force_tqdm", action="store_true", help="Force tqdm to render even if not a TTY.")

    args = parser.parse_args()

    if not args.reports_csv.exists():
        raise FileNotFoundError(args.reports_csv)
    if not args.gt_csv.exists():
        raise FileNotFoundError(args.gt_csv)

    # Force key column to string to avoid 12345 vs 12345.0 mismatches
    reports_df = pd.read_csv(args.reports_csv, dtype={ANON_ACC_COL: str})
    gt_df = pd.read_csv(args.gt_csv, dtype={ANON_ACC_COL: str})

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
    if args.skip_qa:
        print("QA stage: SKIPPED")
    else:
        print("QA stage: ENABLED (verbatim_text + rationale provided to QA; evidence must come from verbatim_text)")
    if args.force_tqdm:
        print("tqdm: FORCED ON")

    disable_bar = False if args.force_tqdm else (not sys.stderr.isatty())

    with tqdm(
        total=len(reports_df),
        desc="Processing reports",
        unit="report",
        dynamic_ncols=True,
        file=sys.stderr,
        disable=disable_bar,
    ) as pbar:
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

            raw_extract = ""
            try:
                gt_entry = gt_lookup.get(anon_acc) if anon_acc else None

                if gt_entry and gt_entry.get("uids"):
                    uids: List[str] = gt_entry["uids"]
                    prompt = build_prompt_with_gt(report_text, uids)
                    raw_extract = ollama_chat(prompt=prompt, model=args.model, url=args.url, timeout_s=args.timeout)
                    parsed = safe_parse_json(raw_extract)
                    if not parsed or not validate_gt_style_output(parsed):
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
                        "qa_enabled": (not args.skip_qa),
                    }

                else:
                    # Fallback if GT missing for this Anon Acc #
                    prompt = build_prompt_fallback(report_text)
                    raw_extract = ollama_chat(prompt=prompt, model=args.model, url=args.url, timeout_s=args.timeout)
                    parsed = safe_parse_json(raw_extract)
                    if not parsed or not validate_fallback_output(parsed):
                        raise ValueError("Invalid JSON or missing expected fallback fields")

                    # Attach requested source fields
                    parsed[REPORT_COL] = report_text
                    parsed[ANON_ACC_COL] = anon_acc

                    # Add explicit note that GT was missing
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
                        "qa_enabled": (not args.skip_qa),
                    }

                    # Ensure 'qa' exists for each sequence in fallback too (so downstream is consistent)
                    if isinstance(parsed.get("sequences"), list):
                        for s in parsed["sequences"]:
                            if isinstance(s, dict) and "qa" not in s:
                                s["qa"] = []

                # -----------------------------
                # Stage 2: Per-sequence QA (UPDATED: uses verbatim_text + rationale)
                # -----------------------------
                if not args.skip_qa:
                    if "sequences" in parsed and isinstance(parsed["sequences"], list):
                        for _, seq in enumerate(parsed["sequences"]):
                            if not isinstance(seq, dict):
                                continue

                            seq_text = seq.get("verbatim_text", "")
                            if not isinstance(seq_text, str):
                                seq_text = ""

                            # If no extractable text, fill n/a answers without calling the model
                            if seq_text.strip() == "" or seq_text.strip() == "No text could be extracted for that sequence":
                                seq["qa"] = [
                                    {
                                        "question": q,
                                        "answer": "n/a",
                                        "confidence": "low",
                                        "evidence": [],
                                        "notes": "No extractable sequence text available.",
                                    }
                                    for q in QA_QUESTIONS
                                ]
                                if args.delay_between_sequences > 0:
                                    time.sleep(args.delay_between_sequences)
                                continue

                            raw_qa = ""
                            try:
                                qa_prompt = build_prompt_sequence_qa(seq_text, seq.get("rationale"))
                                raw_qa = ollama_chat(
                                    prompt=qa_prompt,
                                    model=args.model,
                                    url=args.url,
                                    timeout_s=args.timeout,
                                )
                                qa_parsed = safe_parse_json(raw_qa)
                                if not qa_parsed or not validate_qa_output(qa_parsed):
                                    raise ValueError("Invalid QA JSON")

                                seq["qa"] = coerce_and_repair_qa(qa_parsed)

                            except Exception as qe:
                                seq["qa_error"] = f"{type(qe).__name__}: {str(qe)[:300]}"
                                seq["qa_raw_model_output"] = raw_qa
                                seq["qa"] = [
                                    {
                                        "question": q,
                                        "answer": "n/a",
                                        "confidence": "low",
                                        "evidence": [],
                                        "notes": "QA failed or returned invalid JSON.",
                                    }
                                    for q in QA_QUESTIONS
                                ]

                            if args.delay_between_sequences > 0:
                                time.sleep(args.delay_between_sequences)

                # Write final JSON for this report
                out_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

            except Exception as e:
                err = {
                    "row_index": int(idx),
                    ANON_ACC_COL: anon_acc,
                    REPORT_COL: report_text,
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "raw_model_output_extraction": raw_extract,
                    "num_sequences_actual": None,
                    "sequence_instance_uids": [],
                    "sequences": [],
                    "notes": ["Pipeline failed (extraction stage)."],
                    "_meta": {
                        "source_reports_csv": str(args.reports_csv),
                        "source_gt_csv": str(args.gt_csv),
                        "report_col": REPORT_COL,
                        "anon_acc_col": ANON_ACC_COL,
                        "model": args.model,
                        "qa_enabled": (not args.skip_qa),
                    },
                }
                out_path.write_text(json.dumps(err, indent=2), encoding="utf-8")

            pbar.update(1)
            if args.delay_between_reports > 0:
                time.sleep(args.delay_between_reports)

    print("Done ✔ One JSON file per report has been written.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — any JSON files already written are preserved.", file=sys.stderr)
        raise
