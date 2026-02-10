#!/usr/bin/env python3
"""
infer_finetuned_clip_on_validation_mosaics.py

Goal
- Use a *fine-tuned* HuggingFace CLIP checkpoint (saved by finetune_clip_on_mosaics.py)
  to label validation mosaics row-by-row, based on each row's UID + Question.
- Zero-shot *classification* using the fine-tuned CLIP:
    score(image, text_prompt(option_i)) and choose argmax.

Input
- Validation CSV with columns at least: UID, Question
- Mosaic images under base_path/UID/.../mosaic.png (nested UID dirs supported)
- Fine-tuned checkpoint directory: <base_path>_CLIP_FT_Output/best or /last

Output
- Append-only CSV:
    <base_path>_CLIP_FT_Output/validation_mosaics_clip_ft_labels.csv

Key features
- tqdm loader with live status
- resume support (skip already-processed rows by input_row_index)
- never emits "Unsupported question" (always routes to an option set)
- adds debug columns: matched_question, match_score, option_strategy

Example
python infer_finetuned_clip_on_validation_mosaics.py \
  --base_path /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed \
  --in_csv /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/Test_Data_2026_02_01_v01.csv \
  --checkpoint /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed_CLIP_FT_Output/best \
  --device cuda
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path(
    "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed"
)
DEFAULT_IN_CSV = Path(
    "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/Test_Data_2026_02_01_v01.csv"
)
DEFAULT_UID_COL = "UID"
DEFAULT_QUESTION_COL = "Question"
DEFAULT_MOSAIC_NAME = "mosaic.png"
DEFAULT_MOSAIC_RELATIVE_DIR = ""  # empty => look in UID dir first
DEFAULT_TIMEOUT_S = 0.0
DEFAULT_FUZZY_THRESHOLD = 0.80


# -----------------------------
# Your EXACT questions (from your message)
# -----------------------------
Q_CATHTIP = (
    "Please identify the artery in which the catheter is located during this angiogram. "
    "Do not state anything except the catheter location."
)
Q_SHEATH = (
    "Please identify the artery in which the sheath is located during this angiogram. "
    "Do not state anything except the catheter location."
)
Q_SHEATH_TIP = (
    "Please identify the artery in which the sheath tip is located during this angiogram. "
    "Do not state anything except the catheter location."
)

Q_ABN = "Is there an arterial abnormality in this angiogram? Please state yes or no."
Q_ACUTE_ABN = "Is there an acute arterial abnormality in this angiogram? Please state yes or no."
Q_ACUTE_INJURY = "Is there an acute arterial injury in this angiogram? Please state yes or no."
Q_VASC_ABERR = "Is there a vascular aberrancy demonstrated in this angiogram? Please state yes or no."
Q_EXTRAV = "Is there active arterial extravasation in this angiogram? Please state yes or no."

Q_EMBOL = "What is the name of the dominant artery that has been embolized in this angiogram?"


# -----------------------------
# Option sets
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
# Utils
# -----------------------------
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def short_text(s: str, n: int = 60) -> str:
    s = str(s or "").strip()
    return (s[:n] + "…") if len(s) > n else s


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def looks_like_yesno(question_norm: str) -> bool:
    return question_norm.startswith("is ") or ("please state yes or no" in question_norm)


def select_options_for_question(
    question_raw: str,
    fuzzy_threshold: float,
) -> Tuple[List[str], str, float, str]:
    """
    Always returns an option list (never None), plus:
      matched_question (string),
      match_score (float),
      option_strategy (string)
    """
    qn = _norm_q(question_raw)

    # (1) exact match
    if qn in QUESTION_TO_OPTIONS:
        return QUESTION_TO_OPTIONS[qn], qn, 1.0, "exact_template_match"

    # (2) fuzzy match
    best_key = ""
    best_score = -1.0
    for k in CONFIG_QUESTIONS_NORM:
        sc = similarity(qn, k)
        if sc > best_score:
            best_score = sc
            best_key = k

    if best_score >= fuzzy_threshold and best_key in QUESTION_TO_OPTIONS:
        return QUESTION_TO_OPTIONS[best_key], best_key, float(best_score), "fuzzy_template_match"

    # (3) heuristics
    if ("identify" in qn and "artery" in qn) or any(x in qn for x in ["catheter", "sheath", "embolized", "embolised"]):
        return VESSEL_OPTIONS, "vessel_location_or_embolized", float(best_score), "heuristic_vessel"

    if looks_like_yesno(qn):
        return YESNO_OPTIONS, "yes/no/unclear", float(best_score), "heuristic_yesno"

    # (4) hard fallback
    return YESNO_OPTIONS, "yes/no/unclear", float(best_score), "fallback_yesno"


def ensure_csv_header(out_path: Path, columns: List[str]) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(out_path, index=False)


def append_csv_row(out_path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    ordered = {c: row.get(c) for c in columns}
    pd.DataFrame([ordered]).to_csv(out_path, mode="a", header=False, index=False)


def load_already_done_indices(out_path: Path, index_col: str = "input_row_index") -> Set[int]:
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    try:
        done = pd.read_csv(out_path, usecols=[index_col])
        vals = done[index_col].dropna().astype(int).tolist()
        return set(vals)
    except Exception:
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
    direct = base_path / uid
    if direct.exists() and direct.is_dir():
        return direct

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

    candidate = uid_dir / mosaic_relative_dir / mosaic_name if mosaic_relative_dir else uid_dir / mosaic_name
    if candidate.exists():
        return ResolvedMosaic(uid=uid, seq_dir=uid_dir, mosaic_path=candidate, ok=True, error=None)

    try:
        hits = list(uid_dir.rglob(mosaic_name))
        if hits:
            return ResolvedMosaic(uid=uid, seq_dir=uid_dir, mosaic_path=hits[0], ok=True, error=None)
    except Exception:
        pass

    return ResolvedMosaic(uid=uid, seq_dir=uid_dir, mosaic_path=None, ok=False, error=f"Missing {mosaic_name}")


# -----------------------------
# Fine-tuned CLIP inference (classification over options)
# -----------------------------
@torch.no_grad()
def clip_classify_options(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    image_path: Path,
    question: str,
    options: List[str],
    use_amp: bool,
    prompt_mode: str = "q_plus_opt",
) -> Dict[str, Any]:
    """
    prompt_mode:
      - q_plus_opt:  text = "{question} {option}"
      - opt_only:    text = "{option}"
    """
    image = Image.open(image_path).convert("RGB")

    if prompt_mode == "opt_only":
        texts = [f"{opt}" for opt in options]
    else:
        texts = [f"{question} {opt}" for opt in options]

    inputs = processor(text=texts, images=[image] * len(texts), return_tensors="pt", padding=True)
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        out = model(**inputs, return_dict=True)

        # HF CLIPModel returns logits_per_image shape [batch_images, batch_texts]
        # Here we repeated the image len(texts) times => logits_per_image is [N, N].
        # We want the diagonal: each repeated-image paired with each text prompt.
        logits = out.logits_per_image  # [N, N]
        diag = logits.diag()  # [N]
        probs = F.softmax(diag, dim=0).detach().cpu().numpy()

    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx] * 100.0)

    all_scores = {opt: float(p * 100.0) for opt, p in zip(options, probs)}
    top_indices = np.argsort(probs)[-3:][::-1]
    evidence = [f"{options[i]}: {probs[i]*100.0:.1f}%" for i in top_indices]

    return {
        "answer": options[best_idx],
        "confidence": round(confidence, 2),
        "evidence": evidence,
        "all_scores": all_scores,
        "notes": f"CLIP fine-tuned classification | prompt_mode={prompt_mode}",
    }


# -----------------------------
# Main processing loop
# -----------------------------
def run_rows(
    df: pd.DataFrame,
    base_path: Path,
    uid_col: str,
    question_col: str,
    out_csv: Path,
    columns: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    use_amp: bool,
    delay: float,
    mosaic_name: str,
    mosaic_relative_dir: str,
    limit: Optional[int],
    skip_done: bool,
    fuzzy_threshold: float,
    prompt_mode: str,
) -> None:
    ensure_csv_header(out_csv, columns)
    done_indices = load_already_done_indices(out_csv) if skip_done else set()

    if uid_col not in df.columns:
        raise ValueError(f"Input CSV missing UID column: {uid_col}")
    if question_col not in df.columns:
        raise ValueError(f"Input CSV missing Question column: {question_col}")

    total_rows = len(df) if limit is None else min(len(df), limit)

    with tqdm(total=total_rows, desc="FT-CLIP validating mosaics", unit="row", dynamic_ncols=True) as pbar:
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

            out_row: Dict[str, Any] = {c: row_in.get(c) for c in df.columns}

            out_row["Timestamp"] = utc_timestamp()
            out_row["Model Name"] = str(model.name_or_path)
            out_row["Checkpoint"] = str(model.name_or_path)
            out_row["input_row_index"] = input_row_index
            out_row["uid"] = uid
            out_row["question"] = question

            if not uid or uid.lower() == "nan":
                out_row.update(
                    dict(
                        sequence_dir="",
                        mosaic_path="",
                        matched_question="",
                        match_score=0.0,
                        option_strategy="missing_uid",
                        answer="Not stated",
                        confidence=0,
                        evidence="[]",
                        all_scores="{}",
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
                        matched_question="",
                        match_score=0.0,
                        option_strategy="missing_question",
                        answer="Not stated",
                        confidence=0,
                        evidence="[]",
                        all_scores="{}",
                        notes="Missing Question in this row",
                    )
                )
                append_csv_row(out_csv, out_row, columns)
                pbar.update(1)
                processed += 1
                continue

            pbar.set_postfix_str(f"UID={uid} | status=RESOLVE_MOSAIC")
            resolved = resolve_mosaic_for_uid(base_path, uid, mosaic_name, mosaic_relative_dir)

            if not resolved.ok or not resolved.mosaic_path or not resolved.seq_dir:
                out_row.update(
                    dict(
                        sequence_dir=str(resolved.seq_dir) if resolved.seq_dir else "",
                        mosaic_path="",
                        matched_question="",
                        match_score=0.0,
                        option_strategy="missing_mosaic",
                        answer="Not stated",
                        confidence=0,
                        evidence="[]",
                        all_scores="{}",
                        notes=resolved.error or "Could not resolve mosaic",
                    )
                )
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
                result = clip_classify_options(
                    model=model,
                    processor=processor,
                    device=device,
                    image_path=resolved.mosaic_path,
                    question=question,
                    options=options,
                    use_amp=use_amp,
                    prompt_mode=prompt_mode,
                )

                out_row.update(
                    dict(
                        answer=result["answer"],
                        confidence=result["confidence"],
                        evidence=json.dumps(result.get("evidence", [])),
                        all_scores=json.dumps(result.get("all_scores", {})),
                        notes=result.get("notes", ""),
                    )
                )
                pbar.set_postfix_str(f"UID={uid} | status=DONE")
            except Exception as e:
                out_row.update(
                    dict(
                        answer="Error",
                        confidence=0,
                        evidence="[]",
                        all_scores="{}",
                        notes=str(e)[:200],
                    )
                )
                pbar.set_postfix_str(f"UID={uid} | status=ERROR")

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

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to fine-tuned CLIP checkpoint folder (e.g., ..._CLIP_FT_Output/best or /last).",
    )

    parser.add_argument("--device", type=str, default=None, help="cuda/cpu or None for auto")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--prompt_mode", type=str, default="q_plus_opt", choices=["q_plus_opt", "opt_only"])

    parser.add_argument("--mosaic_name", type=str, default=DEFAULT_MOSAIC_NAME)
    parser.add_argument("--mosaic_relative_dir", type=str, default=DEFAULT_MOSAIC_RELATIVE_DIR)

    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--fuzzy_threshold", type=float, default=DEFAULT_FUZZY_THRESHOLD)

    args = parser.parse_args()

    if not args.in_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {args.in_csv}")
    if not args.base_path.exists():
        raise FileNotFoundError(f"Base path not found: {args.base_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = (device.type == "cuda") and (not args.no_amp)

    # load checkpoint
    processor = CLIPProcessor.from_pretrained(args.checkpoint)
    model = CLIPModel.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    df = pd.read_csv(args.in_csv)

    # output csv
    # Keep output next to your FT outputs by default:
    # <base_path>_CLIP_FT_Output/validation_mosaics_clip_ft_labels.csv
    output_root = Path(f"{args.base_path}_CLIP_FT_Output")
    output_root.mkdir(parents=True, exist_ok=True)
    out_csv = output_root / "validation_mosaics_clip_ft_labels.csv"

    added_cols = [
        "Timestamp",
        "Model Name",
        "Checkpoint",
        "input_row_index",
        "uid",
        "question",
        "sequence_dir",
        "mosaic_path",
        "matched_question",
        "match_score",
        "option_strategy",
        "answer",
        "confidence",
        "evidence",
        "all_scores",
        "notes",
    ]
    columns = list(df.columns) + [c for c in added_cols if c not in df.columns]

    print(f"Input CSV: {args.in_csv}")
    print(f"Rows in input: {len(df)}")
    print(f"Base path: {args.base_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output CSV: {out_csv}")
    print(f"Device: {device} | AMP: {use_amp} | prompt_mode: {args.prompt_mode}")
    print(f"Resume enabled: {not args.no_resume}")
    print(f"Fuzzy threshold: {args.fuzzy_threshold}")

    run_rows(
        df=df,
        base_path=args.base_path,
        uid_col=args.uid_col,
        question_col=args.question_col,
        out_csv=out_csv,
        columns=columns,
        model=model,
        processor=processor,
        device=device,
        use_amp=use_amp,
        delay=args.delay,
        mosaic_name=args.mosaic_name,
        mosaic_relative_dir=args.mosaic_relative_dir,
        limit=args.limit,
        skip_done=(not args.no_resume),
        fuzzy_threshold=args.fuzzy_threshold,
        prompt_mode=args.prompt_mode,
    )

    print("Done ✔ Incremental results preserved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
