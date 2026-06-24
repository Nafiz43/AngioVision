"""Validation CSV loading, sequence indexing, and metrics computation."""

import os
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from shared.text_utils import normalize_gt_answer, normalize_text


# ---------- Column detection ----------

def detect_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> Optional[str]:
    """Find the first matching column in *df* from a list of *candidates*
    (case-insensitive).
    """
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


# ---------- Validation CSV loading ----------

def load_validation_csv(validation_csv: str) -> pd.DataFrame:
    """Load and normalise a validation CSV into a standard schema."""
    df = pd.read_csv(validation_csv)

    sop_col = detect_column(df, ["SOPInstanceUID", "sopinstanceuid", "SOP UID", "SOP_UID"])
    q_col = detect_column(df, ["Question", "question"])
    a_col = detect_column(df, ["Answer", "answer", "GroundTruth", "ground_truth"])

    accession_col = detect_column(
        df,
        ["AccessionNumber", "accessionnumber", "Accession Number"],
        required=False,
    )

    out = pd.DataFrame({
        "SOPInstanceUID": df[sop_col].astype(str),
        "Question": df[q_col].astype(str),
        "GT_Answer_Raw": df[a_col].astype(str),
    })

    if accession_col is not None:
        out["AccessionNumber"] = df[accession_col].astype(str)
    else:
        out["AccessionNumber"] = ""

    out["SOP_norm"] = out["SOPInstanceUID"].apply(normalize_text)
    out["Question_norm"] = out["Question"].apply(normalize_text)
    out["GT_Answer"] = out["GT_Answer_Raw"].apply(normalize_gt_answer)

    out = out.dropna(subset=["SOPInstanceUID", "Question"])
    out = out[out["SOP_norm"] != ""].copy()
    out = out[out["Question_norm"] != ""].copy()

    return out


# ---------- Metadata loading ----------

def load_metadata_csv(metadata_path: str) -> Dict[str, str]:
    """Load an ``Information,Value`` metadata CSV into a dict."""
    out: Dict[str, str] = {}
    try:
        df = pd.read_csv(metadata_path)
        if "Information" not in df.columns or "Value" not in df.columns:
            return out

        for _, row in df.iterrows():
            key = normalize_text(row["Information"])
            val = "" if pd.isna(row["Value"]) else str(row["Value"]).strip()
            out[key] = val
    except Exception:
        return out

    return out


# ---------- Sequence index building ----------

def build_sequence_index(base_path: str) -> Dict[str, Dict[str, str]]:
    """Build an index mapping normalised SOP UIDs to their sequence paths."""
    seq_index: Dict[str, Dict[str, str]] = {}

    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")

    for item in os.listdir(base_path):
        seq_dir = os.path.join(base_path, item)
        if not os.path.isdir(seq_dir):
            continue

        metadata_path = os.path.join(seq_dir, "metadata.csv")
        mosaic_path = os.path.join(seq_dir, "mosaic.png")

        if not os.path.isfile(metadata_path):
            continue
        if not os.path.isfile(mosaic_path):
            continue

        meta = load_metadata_csv(metadata_path)
        sop_uid = meta.get("sopinstanceuid", "").strip()
        accession = meta.get("accessionnumber", "").strip()

        if not sop_uid:
            continue

        seq_index[normalize_text(sop_uid)] = {
            "sop_uid": sop_uid,
            "accession_number": accession,
            "sequence_dir": seq_dir,
            "mosaic_path": mosaic_path,
            "metadata_path": metadata_path,
        }

    return seq_index


# ---------- Metrics computation ----------

def compute_binary_metrics(all_gt: List[str], all_pred: List[str]) -> Dict:
    """Compute accuracy, precision, recall, F1, and confusion matrix values
    from lists of ``"YES"``/``"NO"`` strings.
    """
    y_true = [1 if x == "YES" else 0 for x in all_gt]
    y_pred = [1 if x == "YES" else 0 for x in all_pred]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "F1 Score": round(f1_score(y_true, y_pred, zero_division=0), 3),
    }


def print_metrics(metrics: Dict) -> None:
    """Pretty-print the metrics dictionary produced by ``compute_binary_metrics``."""
    print("\n=== FINAL METRICS ===")
    for key in ("TP", "TN", "FP", "FN"):
        print(f"{key}: {metrics[key]}")
    for key in ("Accuracy", "Precision", "Recall", "F1 Score"):
        print(f"{key:9s}: {metrics[key]:.3f}")
