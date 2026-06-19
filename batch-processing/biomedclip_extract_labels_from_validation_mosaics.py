#!/usr/bin/env python3
"""
extract_labels_from_validation_mosaics_biomedclip.py

Goal
- Read the validation CSV (row-by-row).
- For each row, locate the mosaic for that row's UID under:
    /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed
- Use BioMedCLIP (NOT a language model) to answer the row's own Question via zero-shot classification.
- Append results incrementally to an output CSV (never rewrite).
- NEVER output "Unsupported question" (always select an option-set).

This version is tailored to YOUR exact question set (from your message):
A) Catheter location questions (multi-class vessel name)
   - "Please identify the artery in which the catheter is located..."
   - "Please identify the artery in which the sheath is located..."
   - "Please identify the artery in which the sheath tip is located..."
B) Yes/No questions
   - arterial abnormality / acute abnormality / acute injury / vascular aberrancy / active extravasation
C) Embolized artery name (multi-class vessel name)
   - "What is the name of the dominant artery that has been embolized..."

Progress / loader
- tqdm with live postfix: UID + shortened Question + status

Output extras (debuggable mappings)
- matched_question
- match_score
- option_strategy

Example
python extract_labels_from_validation_mosaics_biomedclip.py \
  --base_path /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed \
  --in_csv /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/Test_Data_2026_02_01_v01.csv \
  --biomed_model microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
  --device cuda
"""

import argparse
import json
import os
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:
    import torch
except ImportError:
    print("ERROR: Required packages not found. Install with:")
    print("  pip install -U torch torchvision open_clip_torch transformers")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.biomedclip_utils import biomedclip_classify, load_biomedclip_model
from shared.csv_helpers import append_csv_row, ensure_csv_header
from shared.sequence_utils import load_already_done_indices, resolve_mosaic_for_uid
from shared.text_utils import utc_timestamp

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path(
    "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed"
)
DEFAULT_IN_CSV = Path(
    "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/Test_Data_2026_02_01_v01.csv"
)

DEFAULT_MOSAIC_NAME = "mosaic.png"
DEFAULT_UID_COL = "UID"
DEFAULT_QUESTION_COL = "Question"
DEFAULT_MOSAIC_RELATIVE_DIR = ""

DEFAULT_BIOMED_MODEL = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEFAULT_FUZZY_THRESHOLD = 0.80


# -----------------------------
# Option sets (covers your pasted label space)
# -----------------------------
YESNO_OPTIONS = ["Yes", "No", "Unclear"]

VESSEL_OPTIONS = [
    "Aorta",
    "Celiac artery",
    "Superior mesenteric artery",
    "Inferior mesenteric artery",
    "Right hepatic artery",
    "Splenic artery",
    "Gastroduodenal artery",
    "Right renal artery",
    "Left renal artery",
    "Right internal iliac artery",
    "Right external iliac artery",
    "Left external iliac artery",
    "Left colic artery",
    "Unclear / other vessel",
    "No catheter visible",
]


# -----------------------------
# EXACT question templates from your message
# -----------------------------
Q_CATHTIP = "Please identify the artery in which the catheter is located during this angiogram. Do not state anything except the catheter location."
Q_SHEATH = "Please identify the artery in which the sheath is located during this angiogram. Do not state anything except the catheter location."
Q_SHEATH_TIP = "Please identify the artery in which the sheath tip is located during this angiogram. Do not state anything except the catheter location."

Q_ABN = "Is there an arterial abnormality in this angiogram? Please state yes or no."
Q_ACUTE_ABN = "Is there an acute arterial abnormality in this angiogram? Please state yes or no."
Q_ACUTE_INJURY = "Is there an acute arterial injury in this angiogram? Please state yes or no."
Q_VASC_ABERR = "Is there a vascular aberrancy demonstrated in this angiogram? Please state yes or no."
Q_EXTRAV = "Is there active arterial extravasation in this angiogram? Please state yes or no."

Q_EMBOL = "What is the name of the dominant artery that has been embolized in this angiogram?"


QUESTIONS_WITH_OPTIONS: List[Dict[str, Any]] = [
    {"question": Q_CATHTIP, "options": VESSEL_OPTIONS},
    {"question": Q_SHEATH, "options": VESSEL_OPTIONS},
    {"question": Q_SHEATH_TIP, "options": VESSEL_OPTIONS},
    {"question": Q_ABN, "options": YESNO_OPTIONS},
    {"question": Q_ACUTE_ABN, "options": YESNO_OPTIONS},
    {"question": Q_ACUTE_INJURY, "options": YESNO_OPTIONS},
    {"question": Q_VASC_ABERR, "options": YESNO_OPTIONS},
    {"question": Q_EXTRAV, "options": YESNO_OPTIONS},
    {"question": Q_EMBOL, "options": VESSEL_OPTIONS},
]


def _norm_q(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())


QUESTION_TO_OPTIONS: Dict[str, List[str]] = {_norm_q(d["question"]): list(d["options"]) for d in QUESTIONS_WITH_OPTIONS}
CONFIG_QUESTIONS_NORM = list(QUESTION_TO_OPTIONS.keys())


# -----------------------------
# Helpers
# -----------------------------
def short_text(s: str, n: int = 60) -> str:
    s = str(s or "").strip()
    return (s[:n] + "...") if len(s) > n else s


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def looks_like_yesno(question_norm: str) -> bool:
    starters = (
        "is ", "are ", "was ", "were ", "do ", "does ", "did ",
        "has ", "have ", "had ", "can ", "could ", "should ", "will ", "would ",
    )
    return question_norm.startswith(starters)


def select_options_for_question(
    question_raw: str,
    fuzzy_threshold: float,
) -> Tuple[List[str], str, float, str]:
    """Always returns an option list, plus matched_question, match_score, option_strategy."""
    qn = _norm_q(question_raw)

    if qn in QUESTION_TO_OPTIONS:
        return QUESTION_TO_OPTIONS[qn], qn, 1.0, "exact_template_match"

    best_key = ""
    best_score = -1.0
    for k in CONFIG_QUESTIONS_NORM:
        sc = similarity(qn, k)
        if sc > best_score:
            best_score = sc
            best_key = k

    if best_score >= fuzzy_threshold and best_key in QUESTION_TO_OPTIONS:
        return QUESTION_TO_OPTIONS[best_key], best_key, float(best_score), "fuzzy_template_match"

    vessel_keywords = [
        "catheter", "sheath", "sheath tip", "embolized", "embolised",
        "dominant artery", "artery in which", "which artery", "which vessel",
        "catheter location", "vessel", "artery", "aorta", "iliac",
        "mesenteric", "celiac", "coeliac", "renal", "hepatic",
        "splenic", "gastroduodenal", "extravasation",
    ]
    if ("identify" in qn and "artery" in qn) or any(kw in qn for kw in ["catheter", "sheath", "embolized", "embolised"]):
        return VESSEL_OPTIONS, "vessel_location_or_embolized", float(best_score), "heuristic_vessel"

    if looks_like_yesno(qn) or "please state yes or no" in qn:
        return YESNO_OPTIONS, "yes/no/unclear", float(best_score), "heuristic_yesno"

    return YESNO_OPTIONS, "yes/no/unclear", float(best_score), "fallback_yesno"


# -----------------------------
# Main processing loop
# -----------------------------
def run_validation_rows_biomedclip(
    df: pd.DataFrame,
    base_path: Path,
    uid_col: str,
    question_col: str,
    out_csv: Path,
    columns: List[str],
    biomed_model_name: str,
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    device: torch.device,
    delay: float,
    mosaic_name: str,
    mosaic_relative_dir: str,
    limit: Optional[int],
    skip_done: bool,
    fuzzy_threshold: float,
) -> None:
    ensure_csv_header(out_csv, columns)
    done_indices = load_already_done_indices(out_csv) if skip_done else set()

    if uid_col not in df.columns:
        raise ValueError(f"Input CSV missing UID column: {uid_col}")
    if question_col not in df.columns:
        raise ValueError(f"Input CSV missing Question column: {question_col}")

    total_rows = len(df) if limit is None else min(len(df), limit)

    with tqdm(
        total=total_rows,
        desc="BioMedCLIP validating mosaics",
        unit="row",
        dynamic_ncols=True,
    ) as pbar:
        processed = 0

        for idx, row_in in df.iterrows():
            if limit is not None and processed >= limit:
                break

            input_row_index = int(idx)
            uid = str(row_in.get(uid_col, "")).strip()
            question = str(row_in.get(question_col, "")).strip()

            pbar.set_postfix_str(f"UID={uid} | Q='{short_text(question)}' | status=INIT")

            if skip_done and input_row_index in done_indices:
                pbar.set_postfix_str(f"UID={uid} | status=SKIP(resume)")
                pbar.update(1)
                processed += 1
                continue

            out_row: Dict[str, Any] = {}
            for c in df.columns:
                out_row[c] = row_in.get(c)

            out_row["Timestamp"] = utc_timestamp()
            out_row["Model Name"] = biomed_model_name
            out_row["input_row_index"] = input_row_index
            out_row["uid"] = uid
            out_row["question"] = question

            if not uid or uid.lower() == "nan":
                pbar.set_postfix_str("status=FAIL(missing UID)")
                out_row.update(dict(
                    sequence_dir="", mosaic_path="",
                    matched_question="", match_score=0.0, option_strategy="missing_uid",
                    answer="Not stated", confidence=0, evidence="[]", all_scores="{}", notes="Missing UID in this row",
                ))
                append_csv_row(out_csv, out_row, columns)
                pbar.update(1)
                processed += 1
                continue

            if not question or question.lower() == "nan":
                pbar.set_postfix_str(f"UID={uid} | status=FAIL(missing Question)")
                out_row.update(dict(
                    sequence_dir="", mosaic_path="",
                    matched_question="", match_score=0.0, option_strategy="missing_question",
                    answer="Not stated", confidence=0, evidence="[]", all_scores="{}", notes="Missing Question in this row",
                ))
                append_csv_row(out_csv, out_row, columns)
                pbar.update(1)
                processed += 1
                continue

            pbar.set_postfix_str(f"UID={uid} | status=RESOLVE_MOSAIC")
            resolved = resolve_mosaic_for_uid(
                base_path=base_path, uid=uid,
                mosaic_name=mosaic_name, mosaic_relative_dir=mosaic_relative_dir,
            )

            if not resolved.ok or not resolved.mosaic_path or not resolved.seq_dir:
                pbar.set_postfix_str(f"UID={uid} | status=MISSING_MOSAIC")
                out_row.update(dict(
                    sequence_dir=str(resolved.seq_dir) if resolved.seq_dir else "",
                    mosaic_path="",
                    matched_question="", match_score=0.0, option_strategy="missing_mosaic",
                    answer="Not stated", confidence=0, evidence="[]", all_scores="{}",
                    notes=resolved.error or "Could not resolve mosaic",
                ))
                append_csv_row(out_csv, out_row, columns)
                pbar.update(1)
                processed += 1
                continue

            out_row["sequence_dir"] = resolved.seq_dir.relative_to(base_path).as_posix()
            out_row["mosaic_path"] = str(resolved.mosaic_path)

            options, matched_q, score, strategy = select_options_for_question(question, fuzzy_threshold)
            out_row["matched_question"] = matched_q
            out_row["match_score"] = round(float(score), 4)
            out_row["option_strategy"] = strategy

            pbar.set_postfix_str(f"UID={uid} | status=INFER({strategy})")

            try:
                result = biomedclip_classify(
                    image_path=resolved.mosaic_path,
                    question=question,
                    options=options,
                    model=model,
                    preprocess=preprocess,
                    tokenizer=tokenizer,
                    device=device,
                )

                out_row.update(dict(
                    answer=result.get("answer"),
                    confidence=result.get("confidence"),
                    evidence=json.dumps(result.get("evidence", [])),
                    all_scores=json.dumps(result.get("all_scores", {})),
                    notes=f"{result.get('notes')} | strategy={strategy} | matched='{matched_q}' | score={score:.3f}",
                ))
                pbar.set_postfix_str(f"UID={uid} | status=DONE")
            except Exception as e:
                pbar.set_postfix_str(f"UID={uid} | status=ERROR")
                out_row.update(dict(
                    answer="Error", confidence=0, evidence="[]", all_scores="{}",
                    notes=str(e)[:200],
                ))

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
    parser.add_argument("--biomed_model", type=str, default=DEFAULT_BIOMED_MODEL,
                        help="BioMedCLIP HF id like microsoft/... (or open_clip id hf-hub:...)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda', 'cpu', or None for auto")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--mosaic_name", type=str, default=DEFAULT_MOSAIC_NAME)
    parser.add_argument("--mosaic_relative_dir", type=str, default=DEFAULT_MOSAIC_RELATIVE_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true", help="Do NOT skip already-processed rows")
    parser.add_argument("--fuzzy_threshold", type=float, default=DEFAULT_FUZZY_THRESHOLD,
                        help="Fuzzy match threshold (0..1).")
    parser.add_argument("--model", type=str, default=None,
                        help="(alias) BioMedCLIP model id. Prefer --biomed_model.")

    args = parser.parse_args()

    if args.model and (args.biomed_model == DEFAULT_BIOMED_MODEL):
        args.biomed_model = args.model

    if not args.in_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {args.in_csv}")
    if not args.base_path.exists():
        raise FileNotFoundError(f"Base path not found: {args.base_path}")

    df = pd.read_csv(args.in_csv)

    output_root = Path(f"{args.base_path}_Output")
    out_csv = output_root / "validation_mosaics_biomedclip_labels.csv"

    added_cols = [
        "Timestamp", "Model Name", "input_row_index", "uid", "question",
        "sequence_dir", "mosaic_path", "matched_question", "match_score",
        "option_strategy", "answer", "confidence", "evidence", "all_scores", "notes",
    ]
    columns = list(df.columns) + [c for c in added_cols if c not in df.columns]

    print(f"Input CSV: {args.in_csv}")
    print(f"Rows in input: {len(df)}")
    print(f"Base path: {args.base_path}")
    print(f"Output CSV: {out_csv}")
    print(f"BioMedCLIP Model: {args.biomed_model}")
    print(f"UID col: {args.uid_col} | Question col: {args.question_col}")
    print(f"Mosaic: {args.mosaic_name} (relative dir: '{args.mosaic_relative_dir or ''}')")
    print(f"Resume enabled: {not args.no_resume}")
    print(f"Fuzzy threshold: {args.fuzzy_threshold}")
    print(f"Configured templates: {len(QUESTION_TO_OPTIONS)} (tailored to your exact question list)")

    model, preprocess, tokenizer, device = load_biomedclip_model(args.biomed_model, args.device)

    run_validation_rows_biomedclip(
        df=df, base_path=args.base_path,
        uid_col=args.uid_col, question_col=args.question_col,
        out_csv=out_csv, columns=columns,
        biomed_model_name=args.biomed_model,
        model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
        delay=args.delay, mosaic_name=args.mosaic_name,
        mosaic_relative_dir=args.mosaic_relative_dir,
        limit=args.limit, skip_done=(not args.no_resume),
        fuzzy_threshold=args.fuzzy_threshold,
    )

    print("Done. Incremental results preserved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
