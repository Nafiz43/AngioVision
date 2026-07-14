"""Step 06 — comprehensive per-question evaluation suite.

Turns the flat *_predictions.csv baselines (steps 01/02/03/05) plus the
fine-tuned frozen-probe per-question table into ONE comprehensive, per-question
leaderboard. Nothing here is hardcoded to a specific model or checkpoint — every
`*_predictions.csv` in cfg.baselines_dir is discovered and scored.

Produces, in the per-run dir (and copied to cfg.eval_out_dir if set):

  model_summary.csv              one row per model: overall macro-F1, F1(yes),
                                 F1(no), FLIPPED macro-F1, TP/TN/FP/FN, accuracy,
                                 n. Honors the "always show F1 + flipped-F1 +
                                 TP/TN/FP/FN" rule.
  per_question_comprehensive.csv one row per (checkpoint, question): the
                                 fine-tuned probe columns (BASELINE / A1..A2+A3),
                                 joined with per-question macro-F1 for every
                                 zero-shot / VLM model AND the three constant
                                 controls (ALL-YES / ALL-NO / RANDOM).

Both are rendered to collapsible-column HTML by bmk.html_report.

Constant controls are computed per question straight from the ground truth:
ALL-YES = predict 1 everywhere, ALL-NO = predict 0, RANDOM = seeded coin flip.
macro-F1 matches the fine-tuned probe table so columns are directly comparable.
"""

from __future__ import annotations

import glob
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .common import normalize_binary, normalize_text
from . import html_report


# ── question <-> label plumbing ──────────────────────────────────────────────

_Q_SUFFIX = "please state yes or no."


def canon_question(q: object) -> str:
    """Join key: lowercase, collapse space, drop the boilerplate suffix.

    The fine-tuned probe table stores questions with the suffix stripped; the
    prediction CSVs keep it. Canonicalize both to the same key.
    """
    s = normalize_text(q)
    if s.endswith(_Q_SUFFIX):
        s = s[: -len(_Q_SUFFIX)].strip()
    return s


def model_label(model_name: str, filename: str) -> str:
    """Short, stable display label for a discovered prediction source."""
    raw = (model_name or "").strip() or os.path.basename(filename)
    s = raw.lower()
    known = [
        ("nova", "Nova"), ("gemma4", "Gemma4"), ("gemma3", "Gemma3"),
        ("medsiglip", "MedSigLIP-zs"), ("siglip2", "SigLIP2-zs"),
        ("siglip", "SigLIP-zs"), ("xclip", "X-CLIP-zs"),
        ("clip-vit", "CLIP-zs"), ("openai/clip", "CLIP-zs"),
        ("llava", "LLaVA"), ("qwen", "Qwen-VL"),
    ]
    for needle, label in known:
        if needle in s:
            return label
    # Fall back to a cleaned raw tag.
    return raw.replace("zeroshot_", "").replace("bedrock_", "")


# ── per-question F1 ──────────────────────────────────────────────────────────

def per_question_f1(y_true: np.ndarray, y_pred: np.ndarray, q: np.ndarray,
                    average: str = "macro") -> dict:
    """{canon_question: F1} over each question's own rows (macro or weighted)."""
    out = {}
    for qu in pd.unique(q):
        m = q == qu
        out[qu] = float(f1_score(y_true[m], y_pred[m], average=average,
                                 labels=[0, 1], zero_division=0))
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def load_prediction_source(path: str):
    """A discovered *_predictions.csv -> (label, y_true, y_pred, canon_q arrays)."""
    df = pd.read_csv(path)
    need = {"Question", "GroundTruth", "Predicted"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} missing {need - set(df.columns)}")
    name = df["Model Name"].iloc[0] if "Model Name" in df.columns else ""
    label = model_label(str(name), path)
    y_true = df["GroundTruth"].apply(normalize_binary).to_numpy()
    y_pred = df["Predicted"].apply(normalize_binary).to_numpy()
    q = df["Question"].apply(canon_question).to_numpy()
    return label, y_true, y_pred, q


def constant_controls(y_true: np.ndarray, q: np.ndarray, seed: int,
                      average: str = "macro") -> dict:
    """{control_label: {canon_question: F1}} for ALL-YES/ALL-NO/RANDOM."""
    rng = np.random.default_rng(seed)
    randp = rng.integers(0, 2, size=len(y_true))
    return {
        "ALL-YES": per_question_f1(y_true, np.ones_like(y_true), q, average),
        "ALL-NO": per_question_f1(y_true, np.zeros_like(y_true), q, average),
        "RANDOM": per_question_f1(y_true, randp, q, average),
    }


# ── the two output tables ────────────────────────────────────────────────────

def build_model_summary(sources: dict, controls_true, controls_q, seed,
                        average: str = "macro") -> pd.DataFrame:
    """One row per model/control: overall F1 + flipped-F1 + confusion counts."""
    rows = []
    f1_col = f"F1_{average}"

    def summarize(label, yt, yp):
        f1_agg = f1_score(yt, yp, average=average, labels=[0, 1], zero_division=0)
        f1_yes = f1_score(yt, yp, pos_label=1, zero_division=0)
        f1_no = f1_score(yt, yp, pos_label=0, zero_division=0)
        f1_flip = f1_score(yt, 1 - yp, average=average, labels=[0, 1], zero_division=0)
        acc = float((yt == yp).mean())
        r = {"model": label, "n": int(len(yt)), "accuracy": round(acc, 3),
             f1_col: round(float(f1_agg), 3), "F1_yes": round(float(f1_yes), 3),
             "F1_no": round(float(f1_no), 3), "F1_flipped": round(float(f1_flip), 3)}
        r.update(confusion(yt, yp))
        return r

    for label, (yt, yp, _q) in sources.items():
        rows.append(summarize(label, yt, yp))
    rng = np.random.default_rng(seed)
    controls = {
        "ALL-YES": np.ones_like(controls_true),
        "ALL-NO": np.zeros_like(controls_true),
        "RANDOM": rng.integers(0, 2, size=len(controls_true)),
    }
    for label, yp in controls.items():
        rows.append(summarize(label, controls_true, yp))

    df = pd.DataFrame(rows).sort_values(f1_col, ascending=False).reset_index(drop=True)
    return df


def build_comprehensive(probe_df: pd.DataFrame, model_q_f1: dict,
                        control_q_f1: dict) -> pd.DataFrame:
    """Join per-question model + control F1 onto the fine-tuned probe rows."""
    df = probe_df.copy()
    df["_q"] = df["question"].apply(canon_question)
    # Constant controls first (they anchor how hard each question is), then models.
    for label, qmap in {**control_q_f1, **model_q_f1}.items():
        df[label] = df["_q"].map(lambda k, m=qmap: m.get(k, np.nan)).round(3)
    return df.drop(columns=["_q"])


# ── entry point ──────────────────────────────────────────────────────────────

def run(cfg, run_dir) -> dict:
    # macro (default) or weighted per-question F1. Weighted writes a parallel
    # *_weighted.{csv,html} set so the macro outputs are never overwritten.
    average = getattr(cfg, "f1_average", "macro")
    if average not in ("macro", "weighted"):
        raise ValueError(f"f1_average must be 'macro' or 'weighted', got {average!r}")
    suffix = "" if average == "macro" else "_weighted"
    tag = f" ({average}-F1)" if average != "macro" else ""

    pred_files = sorted(glob.glob(os.path.join(cfg.baselines_dir, "*_predictions.csv")))
    if not pred_files:
        raise FileNotFoundError(
            f"No *_predictions.csv in {cfg.baselines_dir} — run steps 01/02/03/05 first."
        )

    sources = {}          # label -> (y_true, y_pred, canon_q)
    model_q_f1 = {}       # label -> {canon_question: F1}
    for path in pred_files:
        try:
            label, yt, yp, q = load_prediction_source(path)
        except Exception as e:
            print(f"[06] skip {os.path.basename(path)}: {e}")
            continue
        sources[label] = (yt, yp, q)
        model_q_f1[label] = per_question_f1(yt, yp, q, average)
        print(f"[06] {label:16s} n={len(yt)} from {os.path.basename(path)}")

    if not sources:
        raise RuntimeError("No usable prediction sources discovered.")

    print(f"[06] per-question F1 averaging: {average}")

    # Reference GT/question space for the constant controls: the source with the
    # most rows (all baselines share the 361-row validation set).
    ref_label = max(sources, key=lambda k: len(sources[k][0]))
    ref_yt, _ref_yp, ref_q = sources[ref_label]
    control_q_f1 = constant_controls(ref_yt, ref_q, cfg.random_baseline_seed, average)

    summary = build_model_summary(sources, ref_yt, ref_q,
                                  cfg.random_baseline_seed, average)
    summary_csv = os.path.join(str(run_dir), f"model_summary{suffix}.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\n[06] model summary -> {summary_csv}")
    print(summary.to_string(index=False))

    outputs = {"model_summary_csv": summary_csv, "models": list(sources)}

    # Comprehensive per-question table needs the fine-tuned probe per-question CSV.
    # For weighted F1 use the weighted probe twin (same OOF preds, weighted-F1).
    probe_csv = getattr(cfg, "probe_perq_csv_weighted", "") if average == "weighted" \
        else getattr(cfg, "probe_perq_csv", "")
    if probe_csv and os.path.exists(probe_csv):
        probe_df = pd.read_csv(probe_csv)
        comp = build_comprehensive(probe_df, model_q_f1, control_q_f1)
        comp_csv = os.path.join(str(run_dir), f"per_question_comprehensive{suffix}.csv")
        comp.to_csv(comp_csv, index=False)
        print(f"[06] comprehensive per-question -> {comp_csv} ({len(comp)} rows)")
        outputs["comprehensive_csv"] = comp_csv
        # Freeze checkpoint/group/question/n (cols 0-3); filter chips on
        # checkpoint (0) and question group (1); weighted-average footer by n
        # (col 3); question-distribution chart over (group, question, n).
        html_report.render(comp_csv, comp_csv.replace(".csv", ".html"),
                           title="AngioVision — Comprehensive Per-Question Leaderboard" + tag,
                           heat_from=5, freeze_cols=[0, 1, 2, 3], filter_cols=[0, 1],
                           weight_col=3, qchart_cols=[1, 2, 3])
        outputs["comprehensive_html"] = comp_csv.replace(".csv", ".html")
    else:
        print(f"[06] probe per-question CSV not found ({probe_csv!r}); "
              "comprehensive table skipped — model_summary still written.")

    html_report.render(summary_csv, summary_csv.replace(".csv", ".html"),
                       title="AngioVision — Model Summary Leaderboard" + tag, heat_from=2)
    outputs["model_summary_html"] = summary_csv.replace(".csv", ".html")

    # Mirror to a stable dir if configured (so the latest suite output is easy to find).
    out_dir = getattr(cfg, "eval_out_dir", "")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for key in ("model_summary_csv", "comprehensive_csv",
                    "model_summary_html", "comprehensive_html"):
            p = outputs.get(key)
            if p and os.path.exists(p):
                shutil.copyfile(p, os.path.join(out_dir, os.path.basename(p)))
        print(f"[06] mirrored outputs -> {out_dir}")

    return outputs
