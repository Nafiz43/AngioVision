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

# BioMedCLIP / OpenCLIP deps
try:
    import torch
    import open_clip
    import open_clip.factory as ocf
except ImportError:
    print("ERROR: Required packages not found. Install with:")
    print("  pip install -U torch torchvision open_clip_torch transformers")
    sys.exit(1)


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
DEFAULT_MOSAIC_RELATIVE_DIR = ""  # empty => look in UID dir first

DEFAULT_BIOMED_MODEL = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEFAULT_FUZZY_THRESHOLD = 0.80  # a bit more forgiving for your repeated phrasing


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

# Some datasets use abbreviations in GT; we keep options canonical,
# and you can canonicalize GT later when scoring (optional).
# If you want abbreviations as possible outputs too, add them here:
# "SMA", "IMA", "GDA", etc. (but I recommend keeping canonical).


# -----------------------------
# EXACT question templates from your message
# We route based on these, plus heuristics, so you will not get unsupported questions.
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
    # Vessel location (catheter/sheath/sheath tip)
    {"question": Q_CATHTIP, "options": VESSEL_OPTIONS},
    {"question": Q_SHEATH, "options": VESSEL_OPTIONS},
    {"question": Q_SHEATH_TIP, "options": VESSEL_OPTIONS},

    # Yes/No group
    {"question": Q_ABN, "options": YESNO_OPTIONS},
    {"question": Q_ACUTE_ABN, "options": YESNO_OPTIONS},
    {"question": Q_ACUTE_INJURY, "options": YESNO_OPTIONS},
    {"question": Q_VASC_ABERR, "options": YESNO_OPTIONS},
    {"question": Q_EXTRAV, "options": YESNO_OPTIONS},

    # Embolized artery name (multi-class vessel)
    {"question": Q_EMBOL, "options": VESSEL_OPTIONS},
]


def _norm_q(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())


QUESTION_TO_OPTIONS: Dict[str, List[str]] = {_norm_q(d["question"]): list(d["options"]) for d in QUESTIONS_WITH_OPTIONS}
CONFIG_QUESTIONS_NORM = list(QUESTION_TO_OPTIONS.keys())


# -----------------------------
# Helpers
# -----------------------------
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def short_text(s: str, n: int = 60) -> str:
    s = str(s or "").strip()
    return (s[:n] + "…") if len(s) > n else s


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def looks_like_yesno(question_norm: str) -> bool:
    starters = (
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "has ",
        "have ",
        "had ",
        "can ",
        "could ",
        "should ",
        "will ",
        "would ",
    )
    return question_norm.startswith(starters)


def select_options_for_question(
    question_raw: str,
    fuzzy_threshold: float,
) -> Tuple[List[str], str, float, str]:
    """
    Always returns an option list (never None), plus:
      matched_question (string),
      match_score (float),
      option_strategy (string)

    Strategy order:
    1) Exact normalized match to configured question (your exact templates)
    2) Fuzzy match to configured question if above threshold
    3) Heuristic route:
       - catheter/sheath/embolized/vessel/artery keywords -> VESSEL_OPTIONS
       - yes/no question style -> YESNO_OPTIONS
    4) Hard fallback -> YESNO_OPTIONS
    """
    qn = _norm_q(question_raw)

    # (1) exact match
    if qn in QUESTION_TO_OPTIONS:
        return QUESTION_TO_OPTIONS[qn], qn, 1.0, "exact_template_match"

    # (2) fuzzy match to one of your templates
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
    # Route vessel questions (catheter/sheath/embolized)
    vessel_keywords = [
        "catheter",
        "sheath",
        "sheath tip",
        "embolized",
        "embolised",
        "dominant artery",
        "artery in which",
        "which artery",
        "which vessel",
        "catheter location",
        "vessel",
        "artery",
        "aorta",
        "iliac",
        "mesenteric",
        "celiac",
        "coeliac",
        "renal",
        "hepatic",
        "splenic",
        "gastroduodenal",
        "extravasation",  # often co-occurs; but we still prefer yes/no below for explicit yes/no style
    ]
    # If question asks to "identify the artery" or contains catheter/sheath/embolized -> vessel options
    if ("identify" in qn and "artery" in qn) or any(kw in qn for kw in ["catheter", "sheath", "embolized", "embolised"]):
        return VESSEL_OPTIONS, "vessel_location_or_embolized", float(best_score), "heuristic_vessel"

    # Yes/No routing (your yes/no questions always begin with "Is there ...")
    if looks_like_yesno(qn) or "please state yes or no" in qn:
        return YESNO_OPTIONS, "yes/no/unclear", float(best_score), "heuristic_yesno"

    # (4) hard fallback
    return YESNO_OPTIONS, "yes/no/unclear", float(best_score), "fallback_yesno"


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
# BioMedCLIP checkpoint compatibility patch
# -----------------------------
def _patch_openclip_load_checkpoint_for_position_ids() -> None:
    if getattr(ocf, "_position_ids_patch_applied", False):
        return

    def _load_checkpoint_compat(model, checkpoint_path, strict=True, **kwargs):
        try:
            state_dict = ocf.load_state_dict(checkpoint_path, **kwargs)
        except TypeError:
            state_dict = ocf.load_state_dict(checkpoint_path)

        key = "text.transformer.embeddings.position_ids"
        model_sd = model.state_dict()

        if key not in state_dict and key in model_sd:
            state_dict[key] = model_sd[key]

        if key in state_dict and key not in model_sd:
            del state_dict[key]

        ocf.resize_pos_embed(state_dict, model)
        ocf.resize_text_pos_embed(state_dict, model)

        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        return incompatible_keys

    ocf.load_checkpoint = _load_checkpoint_compat
    ocf._position_ids_patch_applied = True


def _to_hf_hub_id(model_arg: str) -> str:
    if model_arg.startswith("hf-hub:"):
        return model_arg
    return f"hf-hub:{model_arg}"


def load_biomedclip_model(model_name: str, device: Optional[str] = None) -> Tuple[Any, Any, Any, torch.device]:
    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    hf_id = _to_hf_hub_id(model_name)

    print(f"Loading BioMedCLIP model: {model_name} on {device_t}")
    print(f"Using open_clip id: {hf_id}")

    _patch_openclip_load_checkpoint_for_position_ids()

    model, _, preprocess_val = open_clip.create_model_and_transforms(hf_id)
    tokenizer = open_clip.get_tokenizer(hf_id)

    model = model.to(device_t)
    model.eval()

    print("✓ BioMedCLIP model loaded successfully")
    return model, preprocess_val, tokenizer, device_t


def biomedclip_classify(
    image_path: Path,
    question: str,
    options: List[str],
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    device: torch.device,
) -> Dict[str, Any]:
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        prompts = [f"{question} {opt}" for opt in options]
        text_tokens = tokenizer(prompts).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity_logits = (100.0 * image_features @ text_features.T)  # [1, n_options]
            probs = torch.softmax(similarity_logits, dim=-1).detach().cpu().numpy()[0]

        best_idx = int(np.argmax(probs))
        answer = options[best_idx]
        confidence = float(probs[best_idx] * 100.0)

        all_scores = {opt: float(p * 100.0) for opt, p in zip(options, probs)}
        top_indices = np.argsort(probs)[-3:][::-1]
        evidence = [f"{options[i]}: {probs[i]*100.0:.1f}%" for i in top_indices]

        return {
            "answer": answer,
            "confidence": round(confidence, 2),
            "evidence": evidence,
            "all_scores": all_scores,
            "notes": "BioMedCLIP zero-shot classification",
        }
    except Exception as e:
        return {
            "answer": "Error",
            "confidence": 0,
            "evidence": [],
            "all_scores": {},
            "notes": f"BioMedCLIP error: {str(e)[:200]}",
        }


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
                pbar.set_postfix_str(f"UID={uid} | status=FAIL(missing Question)")
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
            resolved = resolve_mosaic_for_uid(
                base_path=base_path,
                uid=uid,
                mosaic_name=mosaic_name,
                mosaic_relative_dir=mosaic_relative_dir,
            )

            if not resolved.ok or not resolved.mosaic_path or not resolved.seq_dir:
                pbar.set_postfix_str(f"UID={uid} | status=MISSING_MOSAIC")
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

            # ALWAYS choose an option-set (never unsupported)
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

                out_row.update(
                    dict(
                        answer=result.get("answer"),
                        confidence=result.get("confidence"),
                        evidence=json.dumps(result.get("evidence", [])),
                        all_scores=json.dumps(result.get("all_scores", {})),
                        notes=f"{result.get('notes')} | strategy={strategy} | matched='{matched_q}' | score={score:.3f}",
                    )
                )
                pbar.set_postfix_str(f"UID={uid} | status=DONE")
            except Exception as e:
                pbar.set_postfix_str(f"UID={uid} | status=ERROR")
                out_row.update(
                    dict(
                        answer="Error",
                        confidence=0,
                        evidence="[]",
                        all_scores="{}",
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

    parser.add_argument(
        "--biomed_model",
        type=str,
        default=DEFAULT_BIOMED_MODEL,
        help="BioMedCLIP HF id like microsoft/... (or open_clip id hf-hub:...)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'cpu', or None for auto",
    )

    parser.add_argument("--delay", type=float, default=0.0)

    parser.add_argument("--mosaic_name", type=str, default=DEFAULT_MOSAIC_NAME)
    parser.add_argument("--mosaic_relative_dir", type=str, default=DEFAULT_MOSAIC_RELATIVE_DIR)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true", help="Do NOT skip already-processed rows")

    parser.add_argument(
        "--fuzzy_threshold",
        type=float,
        default=DEFAULT_FUZZY_THRESHOLD,
        help="Fuzzy match threshold (0..1). Lower => more matching; higher => more conservative.",
    )

    # Back-compat alias
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="(alias) BioMedCLIP model id. Prefer --biomed_model.",
    )

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
        "Timestamp",
        "Model Name",
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
    print(f"Output CSV: {out_csv}")
    print(f"BioMedCLIP Model: {args.biomed_model}")
    print(f"UID col: {args.uid_col} | Question col: {args.question_col}")
    print(f"Mosaic: {args.mosaic_name} (relative dir: '{args.mosaic_relative_dir or ''}')")
    print(f"Resume enabled: {not args.no_resume}")
    print(f"Fuzzy threshold: {args.fuzzy_threshold}")
    print(f"Configured templates: {len(QUESTION_TO_OPTIONS)} (tailored to your exact question list)")

    model, preprocess, tokenizer, device = load_biomedclip_model(args.biomed_model, args.device)

    run_validation_rows_biomedclip(
        df=df,
        base_path=args.base_path,
        uid_col=args.uid_col,
        question_col=args.question_col,
        out_csv=out_csv,
        columns=columns,
        biomed_model_name=args.biomed_model,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        delay=args.delay,
        mosaic_name=args.mosaic_name,
        mosaic_relative_dir=args.mosaic_relative_dir,
        limit=args.limit,
        skip_done=(not args.no_resume),
        fuzzy_threshold=args.fuzzy_threshold,
    )

    print("Done ✔ Incremental results preserved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
