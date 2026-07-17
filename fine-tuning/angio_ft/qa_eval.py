"""
angio_ft.qa_eval
─────────────────
Binary (yes/no) QA evaluation of trained checkpoints on validation sequence
directories, followed by scoring via ``calculate_score.py``.

Flow (per checkpoint):
    for each sequence dir:
        read metadata.csv -> (accession, SOPInstanceUID)
        look up the GT questions for that SOP
        encode the sequence once (temporal-aware, matches training)
        for each question: build yes/no hypotheses, pick the higher-similarity
                           option -> YES / NO prediction
    write predictions CSV -> run calculate_score.py -> parse metrics

Multiple checkpoints (a run directory of ``epoch_*.pt`` / ``last.pt``) are
compared and the best-by-ORIGINAL-accuracy prediction CSV is copied to
``--out_csv``.  The model is the shared :class:`angio_ft.models.PooledCLIP`, so
any training checkpoint loads cleanly regardless of arch / temporal settings.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from .common import (
    POOL_CHOICES,
    find_frames_dir,
    get_vit_processor,
    list_images_in_dir,
    normalize_question,
    normalize_str,
    resolve_vit_image_size,
)
from .constants import ARCH_DEFAULT_VIT
from .models import ARCH_CHOICES, PooledCLIP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Optional central settings.py (VALIDATION_CSV / DATA_DIR overrides)
# ─────────────────────────────────────────────────────────────────────────────

SETTINGS_PATH = Path("/data/Deep_Angiography/AngioVision/configs/settings.py")
VALIDATION_CSV_FROM_SETTINGS: Optional[str] = None
DATA_DIR_FROM_SETTINGS: Optional[str] = None

try:
    if SETTINGS_PATH.exists():
        spec = importlib.util.spec_from_file_location("angio_settings", SETTINGS_PATH)
        if spec is None or spec.loader is None:
            print(f"[WARN] Could not build import spec for settings.py: {SETTINGS_PATH}")
        else:
            angio_settings = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(angio_settings)
            VALIDATION_CSV_FROM_SETTINGS = getattr(angio_settings, "VALIDATION_CSV", None)
            DATA_DIR_FROM_SETTINGS = getattr(angio_settings, "DATA_DIR", None)
            if VALIDATION_CSV_FROM_SETTINGS:
                print(f"[INFO] Loaded VALIDATION_CSV from settings.py: {VALIDATION_CSV_FROM_SETTINGS}")
            if DATA_DIR_FROM_SETTINGS:
                print(f"[INFO] Loaded DATA_DIR from settings.py: {DATA_DIR_FROM_SETTINGS}")
    else:
        print(f"[WARN] settings.py not found at: {SETTINGS_PATH}")
except Exception as e:
    print(f"[WARN] Failed to load settings.py ({type(e).__name__}: {e})")


# ─────────────────────────────────────────────────────────────────────────────
# Score-output parsing
# ─────────────────────────────────────────────────────────────────────────────

def _run_subprocess_capture(cmd: List[str]) -> subprocess.CompletedProcess:
    print("\n[INFO] Running subprocess:")
    print("       " + " ".join(cmd))
    cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if cp.stdout:
        print(cp.stdout, end="" if cp.stdout.endswith("\n") else "\n")
    if cp.stderr:
        print(cp.stderr, end="" if cp.stderr.endswith("\n") else "\n", file=sys.stderr)
    return cp


def _parse_int_metric(text: str, key: str) -> Optional[int]:
    m = re.search(rf"\b{re.escape(key)}\b\s*[:=]\s*([0-9]+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_float_metric(text: str, key: str) -> Optional[float]:
    m = re.search(rf"\b{re.escape(key)}\b\s*[:=]\s*([0-9]*\.?[0-9]+)\s*%?", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        val = float(m.group(1))
        if "%" in m.group(0) or val > 1.0:
            val = val / 100.0
        return val
    except Exception:
        return None


def parse_score_output(text: str) -> Dict[str, Optional[float]]:
    prefixes = ["ORIGINAL", "FLIPPED", "ALL_YES", "ALL_NO", "RANDOM"]
    fields = ["TP", "TN", "FP", "FN", "ACCURACY", "PRECISION", "RECALL", "F1"]
    out: Dict[str, Optional[float]] = {}
    for prefix in prefixes:
        for field in fields:
            key = f"{prefix}_{field}"
            if field in {"TP", "TN", "FP", "FN"}:
                out[key] = _parse_int_metric(text, key)
            else:
                out[key] = _parse_float_metric(text, key)
    return out


def format_pair(x: Optional[float], y: Optional[float], decimals: int = 6) -> str:
    def _one(v):
        if v is None:
            return "N/A"
        if isinstance(v, int):
            return str(v)
        return f"{v:.{decimals}f}"
    return f"{_one(x)}/{_one(y)}"


def format_single(x: Optional[float], decimals: int = 6) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Validation lookup + metadata
# ─────────────────────────────────────────────────────────────────────────────

def load_validation_question_lookup(validation_csv: str) -> Dict[str, List[str]]:
    path = Path(validation_csv)
    if not path.exists():
        raise SystemExit(f"[ERROR] Validation CSV does not exist: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"[ERROR] Could not read validation CSV: {path} ({type(e).__name__}: {e})")

    required_cols = {"SOPInstanceUID", "Question"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"[ERROR] Validation CSV missing required columns: {sorted(missing)}")

    lookup: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        sop = normalize_str(row.get("SOPInstanceUID"))
        q = normalize_question(row.get("Question"))
        if not sop or not q:
            continue
        lookup.setdefault(sop, [])
        if q not in lookup[sop]:
            lookup[sop].append(q)

    print(f"[INFO] Loaded validation lookup from: {path}")
    print(f"[INFO] Unique SOPInstanceUIDs in GT: {len(lookup)}")
    print(f"[INFO] Total SOP-question pairs in GT: {sum(len(v) for v in lookup.values())}")
    return lookup


def read_metadata_key_value_csv(seq_dir: Path) -> Tuple[Optional[str], Optional[str], str]:
    meta_path = seq_dir / "metadata.csv"
    if not meta_path.exists():
        return None, None, "missing_metadata_csv"
    try:
        df = pd.read_csv(meta_path)
    except Exception as e:
        return None, None, f"metadata_read_error:{type(e).__name__}"
    if df.empty or len(df.columns) < 2:
        return None, None, "metadata_empty_or_bad_format"

    cols = [c.strip().lower() for c in df.columns]
    try:
        info_idx = cols.index("information")
        val_idx = cols.index("value")
    except ValueError:
        info_idx, val_idx = 0, 1

    info_col = df.columns[info_idx]
    val_col = df.columns[val_idx]

    kv: Dict[str, str] = {}
    for _, row in df.iterrows():
        k = str(row.get(info_col, "")).strip()
        v = str(row.get(val_col, "")).strip()
        if k:
            kv[k] = v

    acc = kv.get("AccessionNumber") or kv.get("Accession Number") or kv.get("Accession")
    sop = kv.get("SOPInstanceUID") or kv.get("SOP Instance UID") or kv.get("SOPInstanceUid")

    # Fallback: case/separator-insensitive lookup. The DICOM extraction
    # pipeline (utils/visual-data-preparation s01) appends lowercase
    # ``accession_number`` / ``sop_instance_uid`` rows, which matter when the
    # DICOM keyword rows are absent (nullish tags are dropped at extraction).
    if not acc or not sop:
        kv_norm = {k.lower().replace(" ", "").replace("_", ""): v for k, v in kv.items()}
        acc = acc or kv_norm.get("accessionnumber") or kv_norm.get("accession")
        sop = sop or kv_norm.get("sopinstanceuid")

    if not acc or acc.lower() == "nan":
        return None, None, f"metadata_missing_accession(keys={list(kv.keys())[:20]})"
    if not sop or sop.lower() == "nan":
        return None, None, f"metadata_missing_sop(keys={list(kv.keys())[:20]})"
    return acc.strip(), sop.strip(), "ok"


def find_validation_sequence_dirs(data_dir: Path) -> List[Path]:
    """Discover sequence directories anywhere under *data_dir* (recursive).

    A sequence dir is any directory owning a ``metadata.csv`` or a ``frames/``
    (or ``Frames/``) subdirectory. This handles both the legacy flat layout
    (sequence dirs directly under data_dir) and nested extraction layouts such
    as ``<AccessionNumber>/<SOPInstanceUID>/`` produced by
    utils/visual-data-preparation ``s01_process_sequences.py``. Dirs matching
    only one of the two signals are still returned so the per-sequence error
    log reports exactly what is missing (metadata vs frames).
    """
    seq_dirs = set()
    for meta in data_dir.rglob("metadata.csv"):
        if meta.is_file():
            seq_dirs.add(meta.parent)
    for name in ("frames", "Frames"):
        for frames_dir in data_dir.rglob(name):
            if frames_dir.is_dir():
                seq_dirs.add(frames_dir.parent)
    return sorted(seq_dirs)


def _clean_vessel_phrase(x: str) -> str:
    x = x.strip().rstrip(".?").strip()
    x = x.replace("it's", "its")
    x = re.sub(r"\s+or one o[fr] its branches$", " or one of its branches", x)
    return x


def make_yes_no_hypotheses_tagged(question: str) -> Tuple[str, str, str]:
    """Return (yes_hypothesis, no_hypothesis, family_tag).

    Hypotheses are phrased as declarative report-style findings (matching the
    text distribution the tower is fine-tuned on), never as questions: with a
    report-tuned tower, "Yes. <question>" / "No. <question>" embed almost
    identically and the decision degenerates to a global polarity bias.
    The family tag groups questions for per-family margin debiasing.
    """
    q = question.strip().rstrip("?").lower()
    q = re.sub(r"\s*please s?t?ate yes or no\W*$", "", q).rstrip("?. ")
    if "variant anatomy" in q or "vascular aberrancy" in q:
        return (
            "Angiography demonstrates variant vascular anatomy.",
            "Angiography demonstrates no variant vascular anatomy.",
            "variant_anatomy",
        )
    if "hemorrhage" in q or "extravasation" in q or "bleeding" in q:
        return (
            "Angiography shows hemorrhage or contrast extravasation.",
            "Angiography shows no hemorrhage and no contrast extravasation.",
            "bleeding",
        )
    if "dissection" in q:
        return (
            "Angiography shows evidence of arterial or venous dissection.",
            "Angiography shows no evidence of arterial or venous dissection.",
            "dissection",
        )
    if "stenosis" in q:
        return (
            "Angiography shows stenosis in a visualized vessel.",
            "Angiography shows no stenosis in any visualized vessel.",
            "stenosis",
        )
    if "stent" in q:
        return (
            "An endovascular stent is visible on angiography.",
            "No endovascular stent is visible on angiography.",
            "stent",
        )
    if "microcatheter" in q:
        return (
            "Contrast is injected through a microcatheter.",
            "Contrast is injected through the base catheter rather than a microcatheter.",
            "microcatheter",
        )
    m = re.search(r"is the (catheter|sheath) tip (?:located )?in (?:an |a |the )?(.+)$", q)
    if m:
        tip, vessel = m.group(1), _clean_vessel_phrase(m.group(2))
        return (
            f"The {tip} tip is positioned in the {vessel}.",
            f"The {tip} tip is not in the {vessel}.",
            f"{tip}_tip",
        )
    m = re.search(r"is the perfused organ .*?the (\w+)$", q)
    if m:
        organ = m.group(1)
        return (
            f"The organ perfused in this angiogram is the {organ}.",
            f"The organ perfused in this angiogram is not the {organ}.",
            "perfused_organ",
        )
    if "opacif" in q:  # also catches typos: opacifed / "In the ... opacified"
        m = re.search(r"(?:is|are|in) (?:the |a |an )?(.+?) opacif", q)
        subj = _clean_vessel_phrase(m.group(1)) if m else "target structure"
        return (
            f"Contrast opacifies the {subj}.",
            f"The {subj} is not opacified.",
            "opacified",
        )
    m = re.search(r"is the (.+?) patent", q)
    if m:
        vessel = _clean_vessel_phrase(m.group(1))
        return (
            f"The {vessel} is patent.",
            f"The {vessel} is occluded.",
            "patency",
        )
    if "embolic material" in q or "coil or plug" in q:
        return (
            "Embolic material such as coils or plugs is present.",
            "No embolic material is present.",
            "embolic_material",
        )
    if "tumor" in q:
        return (
            "A vascular tumor is demonstrated on angiography.",
            "No tumor is demonstrated on angiography.",
            "tumor",
        )
    if "calcif" in q:
        return (
            "Atherosclerotic calcifications are identified along the vessels.",
            "No atherosclerotic calcification is identified.",
            "calcification",
        )
    if "radial or brachial" in q:
        return (
            "The procedure is performed from a radial or brachial approach.",
            "The procedure is performed from a femoral approach.",
            "approach",
        )
    if "femoral approach" in q:
        return (
            "The procedure is performed from a femoral approach.",
            "The procedure is performed from a radial or brachial approach.",
            "approach",
        )
    for pat in (
        r"is there (?:a |an )?(.+?)(?: demonstrated| identified| definitively demonstrated)?(?: (?:in|on|within) (?:this|the) angiogram)?$",
        r"does the angiogram demonstrate (?:a |an )?(.+)$",
        r"is (?:a |an )?(.+?) (?:demonstrated|identified|visible|visualized)(?: .*)?$",
        r"is (?:a |an )?(.+?) (?:demonstrated|identified)? ?(?:in|on) (?:this|the) angiogram$",
    ):
        m = re.search(pat, q)
        if m:
            finding = _clean_vessel_phrase(m.group(1))
            return (
                f"Angiography demonstrates {finding}.",
                f"Angiography demonstrates no {finding}.",
                "demonstrates",
            )
    return (f"Yes. {question}", f"No. {question}", "fallback")


def make_yes_no_hypotheses(question: str) -> Tuple[str, str]:
    yes_h, no_h, _ = make_yes_no_hypotheses_tagged(question)
    return (yes_h, no_h)


# Extra paraphrases per question family, used only with --prompt_ensemble.
# The first hypothesis of each polarity is always the legacy single sentence
# from make_yes_no_hypotheses, so the non-ensemble path is untouched.
_ENSEMBLE_EXTRAS: List[Tuple[Tuple[str, ...], Tuple[List[str], List[str]]]] = [
    (("variant anatomy",), (
        ["There is variant vascular anatomy.",
         "The angiogram shows an anatomic variant of the vessels."],
        ["There is no variant vascular anatomy.",
         "The angiogram shows conventional vascular anatomy without variants."],
    )),
    (("hemorrhage", "extravasation"), (
        ["There is active contrast extravasation.",
         "The angiogram demonstrates active bleeding."],
        ["There is no active contrast extravasation.",
         "The angiogram demonstrates no active bleeding."],
    )),
    (("dissection",), (
        ["There is a vessel dissection.",
         "The angiogram demonstrates an intimal dissection flap."],
        ["There is no vessel dissection.",
         "The angiogram demonstrates no intimal dissection flap."],
    )),
    (("stenosis",), (
        ["There is significant vessel narrowing.",
         "The angiogram demonstrates a stenotic vessel segment."],
        ["There is no significant vessel narrowing.",
         "The angiogram demonstrates widely patent vessels."],
    )),
    (("stent",), (
        ["There is a stent in place.",
         "The angiogram shows a deployed endovascular stent."],
        ["There is no stent in place.",
         "The angiogram shows no deployed endovascular stent."],
    )),
]


def make_yes_no_hypothesis_sets(question: str, ensemble: bool = False) -> Tuple[List[str], List[str]]:
    """Return (yes_hypotheses, no_hypotheses).

    With ensemble=False this is exactly the legacy single pair, so default
    behaviour and numerics are unchanged. With ensemble=True, 2 extra
    paraphrases per polarity are added and the caller averages the (already
    L2-normalised) text embeddings before re-normalising.
    """
    yes_h, no_h = make_yes_no_hypotheses(question)
    if not ensemble:
        return [yes_h], [no_h]
    q = question.strip().rstrip("?").lower()
    for keywords, (yes_extra, no_extra) in _ENSEMBLE_EXTRAS:
        if any(k in q for k in keywords):
            return [yes_h] + yes_extra, [no_h] + no_extra
    base = question.strip().rstrip("?")
    return (
        [yes_h, f"{base}? Yes.", f"The answer is yes: {base.lower()}."],
        [no_h, f"{base}? No.", f"The answer is no: {base.lower()}."],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery / loading
# ─────────────────────────────────────────────────────────────────────────────

def checkpoint_sort_key(p: Path):
    name = p.stem
    m = re.match(r"epoch_(\d+)$", name)
    if m:
        return (0, int(m.group(1)))
    if name == "last":
        return (1, float("inf"))
    return (2, name)


def discover_checkpoints(checkpoint_arg: str) -> List[Path]:
    p = Path(checkpoint_arg)
    if p.is_file():
        return [p]
    if not p.exists():
        raise SystemExit(f"[ERROR] Checkpoint path does not exist: {p}")
    if not p.is_dir():
        raise SystemExit(f"[ERROR] Checkpoint path is neither file nor directory: {p}")

    epoch_ckpts = sorted(
        [x for x in p.iterdir() if x.is_file() and re.fullmatch(r"epoch_\d+\.pt", x.name)],
        key=checkpoint_sort_key,
    )
    if epoch_ckpts:
        return epoch_ckpts

    last_ckpt = p / "last.pt"
    if last_ckpt.exists() and last_ckpt.is_file():
        return [last_ckpt]

    raise SystemExit(f"[ERROR] No checkpoint files found in directory: {p}")


def checkpoint_label(ckpt_path: Path) -> str:
    return ckpt_path.stem


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        # Harmless extras (e.g. a generation decoder, or siglip_bias when
        # evaluating with a clip-arch model) are ignored but reported.
        print(f"[WARN] Unexpected keys ignored (showing up to 20): {unexpected[:20]}")
    if missing:
        raise SystemExit(
            f"[ERROR] Checkpoint {ckpt_path} is missing {len(missing)} model keys "
            f"(showing up to 10): {missing[:10]}\n"
            "        The evaluation architecture does not match the training run - "
            "evaluating a partially-initialised model would produce garbage metrics.\n"
            "        For old checkpoints without an embedded config, pass the matching "
            "--arch / --vit_name / --bert_name / --embed_dim flags explicitly."
        )
    print(f"[INFO] Checkpoint loaded: {ckpt_path}")


def peek_checkpoint_config(ckpt_path: Path, device: torch.device) -> Optional[Dict[str, Any]]:
    """Return the embedded training ``config`` dict if present, else None."""
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print(f"[WARN] Could not peek config from {ckpt_path}: {type(e).__name__}: {e}")
        return None
    if isinstance(ckpt, dict):
        cfg = ckpt.get("config")
        if isinstance(cfg, dict):
            return cfg
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Single-checkpoint evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _lr_probe_cv(
    probe_rows: List[Tuple[str, str, str, "np.ndarray", "np.ndarray"]],
    gt_csv: str,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Grouped-CV logistic-regression probe on buffered embeddings.

    Trains LR on [seq ; question ; seq*question] features against the GT
    answers, with GroupKFold folds grouped by accession so no study leaks
    across train/test. This is the 'templates + LR on embeddings' readout;
    scores are cross-validated because only ~361 labeled rows exist.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gt: Dict[Tuple[str, str], int] = {}
    with open(gt_csv, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            a = str(r.get("Answer", "")).strip().lower()
            if a in ("yes", "no"):
                key = (normalize_str(r.get("SOPInstanceUID", "")), str(r.get("Question", "")).strip())
                gt[key] = 1 if a == "yes" else 0

    X, y, groups = [], [], []
    seen: set = set()
    for acc, sop_norm, q, s, qv in probe_rows:
        key = (sop_norm, q.strip())
        lbl = gt.get(key)
        if lbl is None or key in seen:
            continue
        seen.add(key)
        X.append(np.concatenate([s, qv, s * qv]))
        y.append(lbl)
        groups.append(str(acc))
    if len(set(groups)) < n_splits or len(set(y)) < 2:
        print(f"[LR_PROBE] skipped: {len(y)} matched rows / {len(set(groups))} groups is too few")
        return {}
    X = np.stack(X); y = np.asarray(y); groups = np.asarray(groups)

    accs, f1s = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred))
    import numpy as _np
    maj = float(max(_np.mean(y), 1 - _np.mean(y)))
    out = {
        "LR_PROBE_ACC": float(_np.mean(accs)),
        "LR_PROBE_ACC_STD": float(_np.std(accs)),
        "LR_PROBE_F1": float(_np.mean(f1s)),
        "LR_PROBE_MAJORITY": maj,
        "LR_PROBE_N": float(len(y)),
    }
    print(f"[LR_PROBE] n={len(y)} majority={maj:.3f} "
          f"cv_acc={out['LR_PROBE_ACC']:.3f}±{out['LR_PROBE_ACC_STD']:.3f} "
          f"cv_F1={out['LR_PROBE_F1']:.3f} folds={[round(a, 4) for a in accs]}")
    return out


def _dump_rich_probe_npz(rich_rows, gt_csv, path="/tmp/rich_probe_emb.npz"):
    """Dump approach-A frozen features for offline feature-set sweeping."""
    import numpy as np
    gt: Dict[Tuple[str, str], int] = {}
    with open(gt_csv, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            a = str(r.get("Answer", "")).strip().lower()
            if a in ("yes", "no"):
                gt[(normalize_str(r.get("SOPInstanceUID", "")), str(r.get("Question", "")).strip())] = 1 if a == "yes" else 0
    final, pre, attn, qv, y, groups, ql = [], [], [], [], [], [], []
    seen: set = set()
    for acc, sop_norm, q, f_, p_, a_, qv_ in rich_rows:
        key = (sop_norm, q.strip())
        lbl = gt.get(key)
        if lbl is None or key in seen:
            continue
        seen.add(key)
        final.append(f_); pre.append(p_); attn.append(a_); qv.append(qv_)
        y.append(lbl); groups.append(str(acc)); ql.append(q.strip())
    if not y:
        print("[RICH_PROBE] no labeled rows to dump"); return
    np.savez(path, final=np.stack(final), pre=np.stack(pre), attn=np.stack(attn),
             qv=np.stack(qv), y=np.asarray(y), groups=np.asarray(groups), q=np.asarray(ql))
    print(f"[RICH_PROBE] dumped {len(y)} rows -> {path} "
          f"final={np.stack(final).shape} pre={np.stack(pre).shape} attn={np.stack(attn).shape}")


def _complex_probe_cv(
    probe_rows,
    gt_csv: str,
    n_splits: int = 5,
    seed: int = 42,
    pca_dim: int = 64,
    hidden=(256, 64),
    alpha: float = 3.0,
) -> Dict[str, float]:
    """Grouped-CV NON-LINEAR (MLP) probe on the SAME buffered frozen embeddings
    the linear probe uses. PCA feature reduction fights the low-sample/high-dim
    regime (~361 rows, ~1.5k feats). Scaler+PCA+MLP are fit INSIDE each train
    fold (sklearn Pipeline) so held-out studies never leak into fitting. Complements
    _lr_probe_cv; leaves it untouched."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gt: Dict[Tuple[str, str], int] = {}
    with open(gt_csv, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            a = str(r.get("Answer", "")).strip().lower()
            if a in ("yes", "no"):
                key = (normalize_str(r.get("SOPInstanceUID", "")), str(r.get("Question", "")).strip())
                gt[key] = 1 if a == "yes" else 0

    X, y, groups = [], [], []
    seen: set = set()
    for row in probe_rows:
        acc, sop_norm, q, s, qv = row[0], row[1], row[2], row[3], row[4]
        key = (sop_norm, q.strip())
        lbl = gt.get(key)
        if lbl is None or key in seen:
            continue
        seen.add(key)
        X.append(np.concatenate([s, qv, s * qv]))
        y.append(lbl)
        groups.append(str(acc))
    if len(set(groups)) < n_splits or len(set(y)) < 2:
        print(f"[COMPLEX_PROBE] skipped: {len(y)} rows / {len(set(groups))} groups too few")
        return {}
    X = np.stack(X); y = np.asarray(y); groups = np.asarray(groups)
    n_train = (len(y) * (n_splits - 1)) // n_splits
    k = int(min(pca_dim, X.shape[1], max(2, n_train - 1)))

    accs, f1s = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=k, random_state=seed),
            MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=2000,
                          early_stopping=True, n_iter_no_change=25, random_state=seed),
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred))
    maj = float(max(np.mean(y), 1 - np.mean(y)))
    out = {
        "COMPLEX_PROBE_ACC": float(np.mean(accs)),
        "COMPLEX_PROBE_ACC_STD": float(np.std(accs)),
        "COMPLEX_PROBE_F1": float(np.mean(f1s)),
        "COMPLEX_PROBE_MAJORITY": maj,
        "COMPLEX_PROBE_N": float(len(y)),
        "COMPLEX_PROBE_PCA": float(k),
    }
    print(f"[COMPLEX_PROBE] n={len(y)} pca={k} hidden={hidden} alpha={alpha} majority={maj:.3f} "
          f"cv_acc={out['COMPLEX_PROBE_ACC']:.3f}±{out['COMPLEX_PROBE_ACC_STD']:.3f} "
          f"cv_F1={out['COMPLEX_PROBE_F1']:.3f} folds={[round(a, 4) for a in accs]}")
    return out


def predict_and_score(
    model: nn.Module,
    tokenizer,
    processor,
    device: torch.device,
    data_dir_path: str,
    validation_lookup: Dict[str, List[str]],
    score_gt_csv: str,
    out_csv: Path,
    error_csv: Optional[Path],
    label: str,
    frame_chunk_size: int = 64,
    max_frames: Optional[int] = None,
    sequence_repeat_factor: int = 1,
    vit_image_size: Optional[int] = None,
    prompt_ensemble: bool = False,
    margin_debias: bool = False,
    lr_probe: bool = False,
    complex_probe: bool = False,
    rich_probe: bool = False,
    calculate_score_script: str = "calculate_score.py",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Run yes/no QA prediction with the CURRENT model weights and score it.

    This is the checkpoint-free core of :func:`run_single_checkpoint`, also
    called by the training engine after every epoch (``--epoch_qa_eval``) so
    the live in-memory model is evaluated without a reload round-trip.
    Returns the parsed metrics dict (ORIGINAL/FLIPPED/ALL_YES/ALL_NO/RANDOM x
    ACCURACY/PRECISION/RECALL/F1/TP/TN/FP/FN) plus bookkeeping fields.
    """
    was_training = model.training
    model.eval()

    data_dir = Path(data_dir_path)
    if not data_dir.exists():
        raise SystemExit(f"[ERROR] Validation data dir does not exist: {data_dir}")

    temp_out_csv = Path(out_csv)
    temp_out_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_err_csv = Path(error_csv) if error_csv else None
    if temp_err_csv is not None:
        temp_err_csv.parent.mkdir(parents=True, exist_ok=True)

    error_rows: List[Dict[str, str]] = []
    n_total = 0
    n_ok = 0
    skip_counts: Dict[str, int] = {}

    pred_rows: List[Tuple[str, str, str, float, str]] = []  # acc, sop, q, margin, family
    probe_rows: List[Tuple[str, str, str, Any, Any]] = []  # acc, sop_norm, q, seq_vec, q_vec
    rich_rows: List = []  # acc, sop_norm, q, final, pre, attn, qv (approach-A features)

    if True:
        seq_dirs = find_validation_sequence_dirs(data_dir)
        for seq_dir in tqdm(seq_dirs, desc=f"Sequences [{label}]", dynamic_ncols=True):
            n_total += 1

            acc, sop, meta_status = read_metadata_key_value_csv(seq_dir)
            if meta_status != "ok":
                skip_counts[meta_status] = skip_counts.get(meta_status, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": meta_status, "details": ""})
                continue

            sop_norm = normalize_str(sop)
            relevant_questions = validation_lookup.get(sop_norm, [])
            if not relevant_questions:
                s = "no_gt_questions_for_sop"
                skip_counts[s] = skip_counts.get(s, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": s, "details": f"SOPInstanceUID={sop}"})
                continue

            frames_dir = find_frames_dir(seq_dir)
            if frames_dir is None:
                s = "missing_frames_dir"
                skip_counts[s] = skip_counts.get(s, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": s, "details": ""})
                continue

            frame_paths = list_images_in_dir(frames_dir)
            if not frame_paths:
                s = "no_frame_files"
                skip_counts[s] = skip_counts.get(s, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": s, "details": str(frames_dir)})
                continue

            img_emb, emb_status = model.encode_sequence_from_frames(
                processor=processor,
                frame_paths=frame_paths,
                device=device,
                frame_chunk_size=frame_chunk_size,
                max_frames=max_frames,
                sequence_repeat_factor=sequence_repeat_factor,
                vit_image_size=vit_image_size,
            )
            if emb_status != "ok" or img_emb is None:
                skip_counts[emb_status] = skip_counts.get(emb_status, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": emb_status, "details": ""})
                continue

            rich = None
            if rich_probe:
                rich, _ = model.encode_sequence_rich(
                    processor=processor, frame_paths=frame_paths, device=device,
                    frame_chunk_size=frame_chunk_size, max_frames=max_frames,
                    vit_image_size=vit_image_size)

            for q in relevant_questions:
                yes_hs, no_hs = make_yes_no_hypothesis_sets(q, ensemble=prompt_ensemble)
                _, _, family = make_yes_no_hypotheses_tagged(q)
                txt_emb = model.encode_text(tokenizer, yes_hs + no_hs, device=device)
                yes_emb = F.normalize(txt_emb[: len(yes_hs)].mean(dim=0, keepdim=True), dim=-1)
                no_emb = F.normalize(txt_emb[len(yes_hs):].mean(dim=0, keepdim=True), dim=-1)
                sim_yes = (img_emb @ yes_emb.t()).item()
                sim_no = (img_emb @ no_emb.t()).item()
                pred_rows.append((acc, sop, q, sim_yes - sim_no, family))
                if lr_probe or complex_probe or rich_probe:
                    q_emb = model.encode_text(tokenizer, [q], device=device)
                    probe_rows.append((acc, sop_norm, q,
                                       img_emb.squeeze(0).float().cpu().numpy(),
                                       q_emb.squeeze(0).float().cpu().numpy()))
                    if rich_probe and rich is not None:
                        _qn = F.normalize(q_emb, dim=-1).squeeze(0)
                        _fp = torch.as_tensor(rich["frame_proj"], device=device)
                        _w = torch.softmax((_fp @ _qn) / 0.1, dim=0)
                        _sattn = F.normalize((_w.unsqueeze(1) * _fp).sum(0), dim=-1).float().cpu().numpy()
                        rich_rows.append((acc, sop_norm, q, rich["final"], rich["pre"], _sattn,
                                          q_emb.squeeze(0).float().cpu().numpy()))

            n_ok += 1

    # ── decision rule ────────────────────────────────────────────────────
    # Raw: margin > 0. Debiased: subtract the per-family MEDIAN margin first,
    # removing the global polarity bias a drifted text tower imposes on every
    # sequence, while keeping the (informative) ranking across sequences.
    # Uses no ground-truth labels, so there is no leakage.
    thresholds: Dict[str, float] = {}
    if margin_debias and pred_rows:
        by_family: Dict[str, List[float]] = {}
        for _, _, _, margin, family in pred_rows:
            by_family.setdefault(family, []).append(margin)
        import statistics
        thresholds = {fam: statistics.median(v) for fam, v in by_family.items()}
        print("[INFO] Margin debias ON; per-family median thresholds:")
        for fam, thr in sorted(thresholds.items()):
            print(f"    {fam}: {thr:+.6f} (n={len(by_family[fam])})")

    with open(temp_out_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["AccessionNumber", "SOPInstanceUID", "Question", "Answer"])
        for acc, sop, q, margin, family in pred_rows:
            pred = "YES" if margin > thresholds.get(family, 0.0) else "NO"
            writer.writerow([acc, sop, q, pred])

    print(f"[INFO] Wrote predictions to: {temp_out_csv}")
    print("\n[SUMMARY]")
    print(f"  Evaluated:               {label}")
    print(f"  Data dir:                {data_dir}")
    print(f"  Scoring GT CSV:          {score_gt_csv}")
    print(f"  Total sequence dirs:     {n_total}")
    print(f"  Successfully predicted:  {n_ok}")
    print(f"  Sequence repeat factor:  {sequence_repeat_factor}")
    if skip_counts:
        print("  Skipped counts:")
        for k, v in sorted(skip_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {k}: {v}")

    if temp_err_csv is not None:
        pd.DataFrame(error_rows, columns=["seq_dir", "status", "details"]).to_csv(temp_err_csv, index=False)
        print(f"[INFO] Wrote error log to: {temp_err_csv}")

    if not temp_out_csv.exists():
        raise SystemExit(f"[ERROR] Prediction CSV not found, cannot run calculate_score.py: {temp_out_csv}")

    score_cmd = [
        sys.executable, calculate_score_script,
        "--pred_path", str(temp_out_csv),
        "--gt_path", str(score_gt_csv),
        "--random_seed", str(random_seed),
    ]
    cp = _run_subprocess_capture(score_cmd)

    combined_output = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
    metrics = parse_score_output(combined_output)

    print("\n[CHECKPOINT METRICS]")
    print(f"  Evaluated:  {label}")
    print(f"  Accuracy (orig/flip): {format_pair(metrics.get('ORIGINAL_ACCURACY'), metrics.get('FLIPPED_ACCURACY'))}")
    print(f"  F1-score (orig/flip): {format_pair(metrics.get('ORIGINAL_F1'), metrics.get('FLIPPED_F1'))}")

    if lr_probe and probe_rows:
        metrics.update(_lr_probe_cv(probe_rows, score_gt_csv, seed=random_seed))

    if rich_probe and rich_rows:
        _dump_rich_probe_npz(rich_rows, score_gt_csv)

    if complex_probe and probe_rows:
        metrics.update(_complex_probe_cv(probe_rows, score_gt_csv, seed=random_seed))

    if was_training:
        model.train()

    result: Dict[str, Any] = {
        "label": label,
        "pred_csv": temp_out_csv,
        "error_csv": temp_err_csv,
        "n_total": n_total,
        "n_ok": n_ok,
    }
    result.update(metrics)
    return result


def run_single_checkpoint(
    model: nn.Module,
    tokenizer,
    processor,
    ckpt_path: Path,
    device: torch.device,
    args,
    validation_lookup: Dict[str, List[str]],
    validation_csv_path: str,
    data_dir_path: str,
    vit_image_size: Optional[int],
) -> Dict[str, Any]:
    model.eval()
    load_checkpoint(model, ckpt_path, device=device)

    final_out_csv = Path(args.out_csv)
    final_out_csv.parent.mkdir(parents=True, exist_ok=True)

    final_error_csv = Path(args.error_csv) if args.error_csv else None
    if final_error_csv is not None:
        final_error_csv.parent.mkdir(parents=True, exist_ok=True)

    label = checkpoint_label(ckpt_path)
    temp_out_csv = final_out_csv.parent / f"{final_out_csv.stem}__{label}{final_out_csv.suffix or '.csv'}"
    temp_err_csv = None
    if final_error_csv is not None:
        temp_err_csv = final_error_csv.parent / f"{final_error_csv.stem}__{label}{final_error_csv.suffix or '.csv'}"

    result = predict_and_score(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        device=device,
        data_dir_path=data_dir_path,
        validation_lookup=validation_lookup,
        score_gt_csv=validation_csv_path,
        out_csv=temp_out_csv,
        error_csv=temp_err_csv,
        label=label,
        frame_chunk_size=args.frame_chunk_size,
        max_frames=(args.max_frames if args.max_frames and args.max_frames > 0 else None),
        sequence_repeat_factor=args.sequence_repeat_factor,
        vit_image_size=vit_image_size,
        prompt_ensemble=getattr(args, "prompt_ensemble", False),
        margin_debias=getattr(args, "margin_debias", False),
        lr_probe=getattr(args, "lr_probe", False),
        complex_probe=getattr(args, "complex_probe", False),
        rich_probe=getattr(args, "rich_probe", False),
        calculate_score_script=args.calculate_score_script,
        random_seed=args.random_seed,
    )
    result["checkpoint"] = ckpt_path
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Multi-checkpoint driver
# ─────────────────────────────────────────────────────────────────────────────

def run(args) -> None:
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] arch   = {args.arch}")

    checkpoints = discover_checkpoints(args.checkpoint)
    print(f"[INFO] Found {len(checkpoints)} checkpoint(s) to evaluate:")
    for ck in checkpoints:
        print(f"  - {ck}")

    # If the checkpoint embeds its training config (new-format checkpoints), use
    # it to rebuild the exact architecture. This prevents silent train/val
    # mismatch - e.g. evaluating a temporal-OFF checkpoint with the default
    # temporal-ON flags, which would corrupt results.
    embedded = peek_checkpoint_config(checkpoints[0], device)
    if embedded:
        print("[INFO] Checkpoint carries an embedded training config; applying it to the model architecture.")
        args.arch = embedded.get("arch", args.arch)
        args.vit_name = embedded.get("vit_name", args.vit_name)
        args.bert_name = embedded.get("bert_name", args.bert_name)
        args.embed_dim = embedded.get("embed_dim", args.embed_dim)
        args.temporal_mode = embedded.get("temporal_mode", args.temporal_mode)
        if "temporal_on_frames" in embedded:
            args.disable_frame_temporal = not bool(embedded["temporal_on_frames"])
        if "temporal_on_sequences" in embedded:
            args.enable_sequence_temporal = bool(embedded["temporal_on_sequences"])
        args.frame_temporal_scale = embedded.get("frame_temporal_scale", args.frame_temporal_scale)
        args.sequence_temporal_scale = embedded.get("sequence_temporal_scale", args.sequence_temporal_scale)
        if embedded.get("frame_pooling"):
            args.frame_pooling = embedded["frame_pooling"]
        if embedded.get("sequence_pooling"):
            args.sequence_pooling = embedded["sequence_pooling"]
        if embedded.get("vit_image_size") is not None:
            args.vit_image_size = embedded["vit_image_size"]
        print(f"[INFO]   -> arch={args.arch} temporal_mode={args.temporal_mode} "
              f"frame_temporal={not args.disable_frame_temporal} seq_temporal={args.enable_sequence_temporal} "
              f"vit={args.vit_name} bert={args.bert_name}")
    else:
        print("[INFO] No embedded config found in checkpoint; using CLI flags for the model architecture.")

    if args.vit_name is None:
        args.vit_name = ARCH_DEFAULT_VIT.get(args.arch, "microsoft/rad-dino")
        print(f"[INFO] --vit_name not given; defaulting to {args.vit_name} for --arch {args.arch}.")

    processor = get_vit_processor(args.vit_name)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    vit_image_size = resolve_vit_image_size(args.vit_name, args.vit_image_size)

    frame_pooling = args.frame_pooling if args.frame_pooling else args.pooling
    sequence_pooling = args.sequence_pooling if args.sequence_pooling else args.pooling

    validation_csv_path = VALIDATION_CSV_FROM_SETTINGS or args.validation_csv
    data_dir_path = DATA_DIR_FROM_SETTINGS or args.data_dir
    selection_csv_path = getattr(args, "selection_csv", "") or None
    print(f"[INFO] Using validation CSV: {validation_csv_path}")
    print(f"[INFO] Using data directory: {data_dir_path}")

    validation_lookup = load_validation_question_lookup(validation_csv_path)

    if selection_csv_path:
        print(f"[INFO] Selection mode: best checkpoint chosen on dev split {selection_csv_path}; "
              f"final metrics reported once on held-out {validation_csv_path}.")
        # Predictions must cover BOTH splits, so merge the dev questions in.
        dev_lookup = load_validation_question_lookup(selection_csv_path)
        for sop, questions in dev_lookup.items():
            bucket = validation_lookup.setdefault(sop, [])
            for q in questions:
                if q not in bucket:
                    bucket.append(q)

    # GT used for per-checkpoint scoring (selection). Without --selection_csv
    # this is the plain single-CSV behaviour, unchanged.
    score_csv_path = selection_csv_path or validation_csv_path

    model = PooledCLIP(
        vit_name=args.vit_name,
        text_model_name=args.bert_name,
        embed_dim=args.embed_dim,
        arch=args.arch,
        frame_pooling=frame_pooling,
        sequence_pooling=sequence_pooling,
        temporal_mode=args.temporal_mode,
        temporal_on_frames=(not args.disable_frame_temporal),
        temporal_on_sequences=args.enable_sequence_temporal,
        frame_temporal_scale=args.frame_temporal_scale,
        sequence_temporal_scale=args.sequence_temporal_scale,
    ).to(device)
    model.eval()

    results: List[Dict[str, Any]] = []
    for ckpt_path in checkpoints:
        print("\n" + "=" * 100)
        print(f"[INFO] Evaluating checkpoint: {ckpt_path}")
        print("=" * 100)
        results.append(run_single_checkpoint(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            ckpt_path=ckpt_path,
            device=device,
            args=args,
            validation_lookup=validation_lookup,
            validation_csv_path=score_csv_path,
            data_dir_path=data_dir_path,
            vit_image_size=vit_image_size,
        ))

    if not results:
        raise SystemExit("[ERROR] No checkpoints were evaluated.")

    valid_results = [r for r in results if r.get("ORIGINAL_ACCURACY") is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x["ORIGINAL_ACCURACY"])
    else:
        best = results[-1]
        print("[WARN] No ORIGINAL accuracy could be parsed; falling back to last checkpoint as best.")

    final_out_csv = Path(args.out_csv)
    final_out_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["pred_csv"], final_out_csv)

    if args.error_csv and best["error_csv"] is not None and Path(best["error_csv"]).exists():
        final_error_csv = Path(args.error_csv)
        final_error_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["error_csv"], final_error_csv)

    print("\n[FINAL CHECKPOINT COMPARISON]")
    print(f"{'Checkpoint':<22}{'Accuracy(O/F)':>22}{'F1(O/F)':>22}")
    for r in results:
        print(
            f"{r['checkpoint'].name:<22}"
            f"{format_pair(r.get('ORIGINAL_ACCURACY'), r.get('FLIPPED_ACCURACY')):>22}"
            f"{format_pair(r.get('ORIGINAL_F1'), r.get('FLIPPED_F1')):>22}"
        )

    selection_note = " (dev split)" if selection_csv_path else ""
    print("\n[BEST CHECKPOINT]")
    print(f"  Checkpoint: {best['checkpoint']}")
    print(f"  Accuracy (orig/flip){selection_note}: "
          f"{format_pair(best.get('ORIGINAL_ACCURACY'), best.get('FLIPPED_ACCURACY'))}")
    print(f"  F1-score (orig/flip){selection_note}: "
          f"{format_pair(best.get('ORIGINAL_F1'), best.get('FLIPPED_F1'))}")
    print(f"  Best prediction CSV copied to: {final_out_csv}")

    if selection_csv_path:
        # Score the selected checkpoint ONCE against the held-out test split.
        test_cmd = [
            sys.executable, args.calculate_score_script,
            "--pred_path", str(best["pred_csv"]),
            "--gt_path", str(validation_csv_path),
            "--random_seed", str(args.random_seed),
        ]
        cp = _run_subprocess_capture(test_cmd)
        test_metrics = parse_score_output(
            (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
        )
        print("\n[HELD-OUT TEST METRICS]")
        print(f"  Selected on dev : {selection_csv_path}")
        print(f"  Reported on test: {validation_csv_path}")
        print(f"  Test Accuracy (orig/flip): "
              f"{format_pair(test_metrics.get('ORIGINAL_ACCURACY'), test_metrics.get('FLIPPED_ACCURACY'))}")
        print(f"  Test F1-score (orig/flip): "
              f"{format_pair(test_metrics.get('ORIGINAL_F1'), test_metrics.get('FLIPPED_F1'))}")
