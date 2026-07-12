"""Shared helpers: text/answer normalization, validation CSV loading, mosaic
sequence indexing, CSV appending.

Ported from frame-processing/02_extract_labels_from_mosaics.py and
utils/07_mcneimer_test.py so both steps of this pipeline share ONE copy of the
normalization rules (any drift between them would silently corrupt the paired
statistics).
"""

from __future__ import annotations

import base64
import csv
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# Merge keys pairing baseline and fine-tuned predictions row-for-row.
MERGE_COLS = ["AccessionNumber", "SOPInstanceUID", "Question"]

_NO_TERMS = [
    "no", "unclear", "not visible", "cannot say", "can not say", "can't say",
    "cant say", "unable to determine", "cannot determine", "can not determine",
    "not sure", "unknown", "indeterminate", "not identifiable", "not seen",
    "not clear", "not possible to determine",
]


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return re.sub(r"\s+", " ", s)


def normalize_gt_answer(raw_answer: object) -> str:
    """Ground-truth answer -> YES/NO. Uncertain/unclear counts as NO."""
    if pd.isna(raw_answer):
        return "NO"
    ans = normalize_text(raw_answer)

    if ans in {"yes", "y", "true", "1", "present", "positive"}:
        return "YES"
    if ans in set(_NO_TERMS) | {"n", "false", "0", "absent", "negative"}:
        return "NO"
    if "yes" in ans or "present" in ans or "positive" in ans:
        return "YES"
    return "NO"


def normalize_llm_answer(raw_answer: object) -> str:
    """Raw VLM output -> strict YES/NO. Anything not clearly YES is NO."""
    if raw_answer is None:
        return "NO"
    ans = normalize_text(raw_answer)

    if ans in {"yes", "y"}:
        return "YES"
    if ans in set(_NO_TERMS) | {"n"}:
        return "NO"
    if "yes" in ans:
        return "YES"
    return "NO"


def normalize_binary(x) -> int:
    """YES/NO (or common variants) -> 1/0 for the statistics step."""
    ans = normalize_text(x)
    if ans in {"1", "true", "yes", "y", "positive", "pos", "present",
               "abnormal", "disease", "stenosis"}:
        return 1
    try:
        return 1 if float(ans) == 1 else 0
    except (TypeError, ValueError):
        return 0


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_row_csv(csv_path: str, row: Dict, fieldnames: List[str]) -> None:
    ensure_parent_dir(csv_path)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_metadata_csv(metadata_path: str) -> Dict[str, str]:
    """metadata.csv with [Information, Value] columns -> dict (keys lowercased)."""
    out: Dict[str, str] = {}
    try:
        df = pd.read_csv(metadata_path)
        if "Information" not in df.columns or "Value" not in df.columns:
            return out
        for _, row in df.iterrows():
            key = normalize_text(row["Information"])
            out[key] = "" if pd.isna(row["Value"]) else str(row["Value"]).strip()
    except Exception:
        return out
    return out


def build_sequence_index(base_path: str) -> Dict[str, Dict[str, str]]:
    """normalized SOPInstanceUID -> sequence dir info.

    Walks base_path recursively; any dir holding BOTH metadata.csv and
    mosaic.png is a sequence dir. Recursive so it handles both a flat layout
    (mosaics_root/<seq>/) and a nested one (mosaics_root/<split>/<accession>/
    <SOP>/), e.g. the Validation_VDP DSA_Split tree.
    """
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"mosaics_root not found: {base_path}")

    seq_index: Dict[str, Dict[str, str]] = {}
    for seq_dir, _dirs, files in os.walk(base_path):
        if "metadata.csv" not in files or "mosaic.png" not in files:
            continue
        metadata_path = os.path.join(seq_dir, "metadata.csv")
        mosaic_path = os.path.join(seq_dir, "mosaic.png")

        meta = load_metadata_csv(metadata_path)
        sop_uid = meta.get("sopinstanceuid", "").strip()
        if not sop_uid:
            continue

        seq_index[normalize_text(sop_uid)] = {
            "sop_uid": sop_uid,
            "accession_number": meta.get("accessionnumber", "").strip(),
            "sequence_dir": seq_dir,
            "mosaic_path": mosaic_path,
            "metadata_path": metadata_path,
        }
    return seq_index


def detect_column(df: pd.DataFrame, candidates: List[str],
                  required: bool = True) -> Optional[str]:
    norm_map = {normalize_text(c): c for c in df.columns}
    for cand in candidates:
        if normalize_text(cand) in norm_map:
            return norm_map[normalize_text(cand)]
    if required:
        raise ValueError(
            f"Could not find required column. Expected one of: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def load_validation_csv(validation_csv: str) -> pd.DataFrame:
    df = pd.read_csv(validation_csv)

    sop_col = detect_column(df, ["SOPInstanceUID", "SOP UID", "SOP_UID"])
    q_col = detect_column(df, ["Question"])
    a_col = detect_column(df, ["Answer", "GroundTruth", "ground_truth"])
    accession_col = detect_column(
        df, ["AccessionNumber", "Accession Number", "Accession"], required=False
    )

    out = pd.DataFrame({
        "SOPInstanceUID": df[sop_col].astype(str),
        "Question": df[q_col].astype(str),
        "GT_Answer_Raw": df[a_col].astype(str),
    })
    out["AccessionNumber"] = (
        df[accession_col].astype(str) if accession_col is not None else ""
    )

    out["SOP_norm"] = out["SOPInstanceUID"].apply(normalize_text)
    out["Question_norm"] = out["Question"].apply(normalize_text)
    out["GT_Answer"] = out["GT_Answer_Raw"].apply(normalize_gt_answer)

    out = out[(out["SOP_norm"] != "") & (out["Question_norm"] != "")].copy()
    return out


def sanitize_model_tag(model: str) -> str:
    """Ollama tag -> filesystem-safe name: 'qwen3-vl:32b' -> 'qwen3-vl_32b'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip())
