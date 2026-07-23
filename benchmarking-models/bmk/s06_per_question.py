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

Test data spans multiple institutions (validation CSV `Institution` column).
Every score column in both tables gets two extra twins, `<col>_UCD` and
`<col>_NONUCD`, computed by re-slicing the SAME predictions/ground-truth by
institution (no retraining) — the pooled/original column is unchanged and
still spans all institutions. bmk.html_report renders an All/UCD/Non-UCD
toggle that shows only the matching column set.
"""

from __future__ import annotations

import csv
import glob
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .common import load_institution_map, normalize_binary, normalize_text
from . import html_report

# Institution buckets every split column is computed for (pooled/original
# column stays unsuffixed and covers ALL of these).
_INST_BUCKETS = ("UCD", "Non-UCD")
_INST_TAG = {"UCD": "UCD", "Non-UCD": "NONUCD"}  # bucket -> CSV column suffix


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


def _conf_tip(c: dict) -> str:
    """Human-readable tooltip string from a confusion dict."""
    n = c["TP"] + c["TN"] + c["FP"] + c["FN"]
    return f"TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']} (n={n})"


def per_question_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                           q: np.ndarray) -> dict:
    """{canon_question: 'TP=.. TN=.. FP=.. FN=.. (n=..)'} over each question's rows."""
    out = {}
    for qu in pd.unique(q):
        m = q == qu
        out[qu] = _conf_tip(confusion(y_true[m], y_pred[m]))
    return out


def institution_of(sop_norm: np.ndarray, inst_map: dict) -> np.ndarray:
    """normalized-SOP array -> institution bucket array ('UCD'/'Non-UCD'/'')."""
    return np.array([inst_map.get(s, "") for s in sop_norm])


def per_question_f1_by_institution(y_true: np.ndarray, y_pred: np.ndarray,
                                   q: np.ndarray, sop_norm: np.ndarray,
                                   inst_map: dict, average: str = "macro"):
    """{bucket: {canon_question: F1}}, {bucket: {canon_question: n}} for UCD/Non-UCD.

    Re-slices the SAME y_true/y_pred already computed — no retraining, just a
    row mask per institution. NaN (blank) where a question has zero rows for
    that bucket rather than a misleading zero_division F1.
    """
    bucket = institution_of(sop_norm, inst_map)
    f1_out, n_out = {}, {}
    for b in _INST_BUCKETS:
        bmask = bucket == b
        f1_map, n_map = {}, {}
        for qu in pd.unique(q):
            m = (q == qu) & bmask
            n = int(m.sum())
            n_map[qu] = n
            f1_map[qu] = (float(f1_score(y_true[m], y_pred[m], average=average,
                                         labels=[0, 1], zero_division=0))
                         if n > 0 else float("nan"))
        f1_out[b] = f1_map
        n_out[b] = n_map
    return f1_out, n_out


def per_question_confusion_by_institution(y_true: np.ndarray, y_pred: np.ndarray,
                                          q: np.ndarray, sop_norm: np.ndarray,
                                          inst_map: dict):
    """{bucket: {canon_question: 'TP=.. ..'}} for UCD/Non-UCD."""
    bucket = institution_of(sop_norm, inst_map)
    out = {}
    for b in _INST_BUCKETS:
        bmask = bucket == b
        conf_map = {}
        for qu in pd.unique(q):
            m = (q == qu) & bmask
            conf_map[qu] = _conf_tip(confusion(y_true[m], y_pred[m])) if m.sum() else ""
        out[b] = conf_map
    return out


def load_prediction_source(path: str):
    """A discovered *_predictions.csv -> (label, y_true, y_pred, canon_q, sop_norm)."""
    df = pd.read_csv(path)
    need = {"Question", "GroundTruth", "Predicted"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} missing {need - set(df.columns)}")
    name = df["Model Name"].iloc[0] if "Model Name" in df.columns else ""
    label = model_label(str(name), path)
    y_true = df["GroundTruth"].apply(normalize_binary).to_numpy()
    y_pred = df["Predicted"].apply(normalize_binary).to_numpy()
    q = df["Question"].apply(canon_question).to_numpy()
    sop_col = "SOPInstanceUID" if "SOPInstanceUID" in df.columns else None
    sop_norm = (df[sop_col].apply(normalize_text).to_numpy() if sop_col
               else np.array([""] * len(df)))
    return label, y_true, y_pred, q, sop_norm


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


def constant_controls_by_institution(y_true: np.ndarray, q: np.ndarray,
                                     sop_norm: np.ndarray, inst_map: dict,
                                     seed: int, average: str = "macro"):
    """{control_label: {bucket: {canon_question: F1}}} for ALL-YES/ALL-NO/RANDOM."""
    rng = np.random.default_rng(seed)
    randp = rng.integers(0, 2, size=len(y_true))
    preds = {"ALL-YES": np.ones_like(y_true), "ALL-NO": np.zeros_like(y_true),
             "RANDOM": randp}
    out = {}
    for label, yp in preds.items():
        f1_map, _n_map = per_question_f1_by_institution(
            y_true, yp, q, sop_norm, inst_map, average)
        out[label] = f1_map
    return out


# ── the two output tables ────────────────────────────────────────────────────

def build_model_summary(sources: dict, controls_true, controls_q, controls_sop,
                        inst_map: dict, seed, average: str = "macro") -> pd.DataFrame:
    """One row per model/control: overall F1 + flipped-F1 + confusion counts,
    plus _UCD/_NONUCD twins of every metric (same predictions, institution-masked)."""
    rows = []
    f1_col = f"F1_{average}"
    metric_cols = ("accuracy", f1_col, "F1_yes", "F1_no", "F1_flipped",
                   "TP", "TN", "FP", "FN")

    def one(yt, yp):
        f1_agg = f1_score(yt, yp, average=average, labels=[0, 1], zero_division=0)
        f1_yes = f1_score(yt, yp, pos_label=1, zero_division=0)
        f1_no = f1_score(yt, yp, pos_label=0, zero_division=0)
        f1_flip = f1_score(yt, 1 - yp, average=average, labels=[0, 1], zero_division=0)
        acc = float((yt == yp).mean())
        r = {"accuracy": round(acc, 3), f1_col: round(float(f1_agg), 3),
             "F1_yes": round(float(f1_yes), 3), "F1_no": round(float(f1_no), 3),
             "F1_flipped": round(float(f1_flip), 3)}
        r.update(confusion(yt, yp))
        return r

    def summarize(label, yt, yp, sop_norm):
        r = {"model": label, "n": int(len(yt))}
        r.update(one(yt, yp))
        bucket = institution_of(sop_norm, inst_map)
        for b in _INST_BUCKETS:
            tag = _INST_TAG[b]
            m = bucket == b
            n = int(m.sum())
            r[f"n_{tag}"] = n
            if n == 0:
                for k in metric_cols:
                    r[f"{k}_{tag}"] = ""
                continue
            for k, v in one(yt[m], yp[m]).items():
                r[f"{k}_{tag}"] = v
        return r

    for label, (yt, yp, _q, sop_norm) in sources.items():
        rows.append(summarize(label, yt, yp, sop_norm))
    rng = np.random.default_rng(seed)
    controls = {
        "ALL-YES": np.ones_like(controls_true),
        "ALL-NO": np.zeros_like(controls_true),
        "RANDOM": rng.integers(0, 2, size=len(controls_true)),
    }
    for label, yp in controls.items():
        rows.append(summarize(label, controls_true, yp, controls_sop))

    df = pd.DataFrame(rows).sort_values(f1_col, ascending=False).reset_index(drop=True)
    return df


def build_comprehensive(probe_df: pd.DataFrame, model_q_f1: dict, control_q_f1: dict,
                        model_q_f1_inst: dict, control_q_f1_inst: dict) -> pd.DataFrame:
    """Join per-question model + control F1 (pooled + _UCD/_NONUCD twins) onto
    the fine-tuned probe rows (whose own F1_* columns bring their own twins,
    added upstream by fine-tuning/probing/perq_leaderboard.py)."""
    df = probe_df.copy()
    df["_q"] = df["question"].apply(canon_question)
    # Constant controls first (they anchor how hard each question is), then models.
    for label, qmap in {**control_q_f1, **model_q_f1}.items():
        df[label] = df["_q"].map(lambda k, m=qmap: m.get(k, np.nan)).round(3)
    for label, by_bucket in {**control_q_f1_inst, **model_q_f1_inst}.items():
        for b in _INST_BUCKETS:
            tag = _INST_TAG[b]
            qmap = by_bucket[b]
            df[f"{label}_{tag}"] = df["_q"].map(lambda k, m=qmap: m.get(k, np.nan)).round(3)
    return df.drop(columns=["_q"])


# probe F1 column  ->  confusion-sidecar column
_PROBE_F1_TO_CONF = {
    "F1_BASELINE": "BASELINE", "F1_A1_pre": "A1_pre", "F1_A2_attn": "A2_attn",
    "F1_A3_xmodal": "A3_xmodal", "F1_A2+A3": "A2+A3",
}


def _load_conf_sidecar(path):
    """{(checkpoint, canon_question): {col: 'TP=.. ..'}} from a builder sidecar."""
    out = {}
    if path and os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                out[(r["checkpoint"], canon_question(r["question"]))] = r
    return out


_TAG_TO_BUCKET = {"UCD": "UCD", "NONUCD": "Non-UCD"}


def _write_titles_matrix(comp, comp_csv, cfg, model_q_conf, control_q_conf,
                         model_q_conf_inst=None, control_q_conf_inst=None):
    """Parallel CSV (same header/rows) whose cells are TP/TN/FP/FN tooltips.

    Probe columns are read from the builder confusion sidecars; model/control
    columns come from the confusion computed in run(). _UCD/_NONUCD columns
    (model/control only — probe institution-split tooltips aren't built by
    the sidecar yet) look up the matching institution-masked confusion dict.
    Key columns (checkpoint, question) keep their real values so the renderer
    can align rows after sort/filter; every other cell is a tooltip or blank.
    """
    model_q_conf_inst = model_q_conf_inst or {}
    control_q_conf_inst = control_q_conf_inst or {}
    macro_probe = getattr(cfg, "probe_perq_csv", "") or ""
    feats_conf = _load_conf_sidecar(macro_probe.replace(".csv", "_conf.csv"))
    noprobe_conf = _load_conf_sidecar(macro_probe.replace(".csv", "_noprobe_conf.csv"))
    cols = list(comp.columns)
    titles = pd.DataFrame("", index=comp.index, columns=cols)
    titles["checkpoint"] = comp["checkpoint"]
    titles["question"] = comp["question"]
    for i in comp.index:
        ck = str(comp.at[i, "checkpoint"])
        cq = canon_question(comp.at[i, "question"])
        for col in cols:
            base, tag = col, None
            for t in ("_UCD", "_NONUCD"):
                if col.endswith(t):
                    base, tag = col[: -len(t)], t[1:]
                    break
            if col in _PROBE_F1_TO_CONF:
                tip = feats_conf.get((ck, cq), {}).get(_PROBE_F1_TO_CONF[col], "")
            elif col == "no_probe":
                tip = noprobe_conf.get((ck, cq), {}).get("no_probe", "")
            elif col in control_q_conf:
                tip = control_q_conf[col].get(cq, "")
            elif col in model_q_conf:
                tip = model_q_conf[col].get(cq, "")
            elif tag and base in control_q_conf_inst:
                tip = control_q_conf_inst[base][_TAG_TO_BUCKET[tag]].get(cq, "")
            elif tag and base in model_q_conf_inst:
                tip = model_q_conf_inst[base][_TAG_TO_BUCKET[tag]].get(cq, "")
            else:
                continue
            titles.at[i, col] = tip
    titles_csv = comp_csv.replace(".csv", "_titles.csv")
    titles.to_csv(titles_csv, index=False)
    print(f"[06] cell-tooltip matrix -> {titles_csv}")
    return titles_csv


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

    inst_map = load_institution_map(cfg.validation_csv)
    print(f"[06] institution map: {len(inst_map)} SOPs "
          f"({'UCD/non-UCD split available' if inst_map else 'no Institution column found — split skipped'})")

    sources = {}          # label -> (y_true, y_pred, canon_q, sop_norm)
    model_q_f1 = {}       # label -> {canon_question: F1}
    model_q_conf = {}     # label -> {canon_question: 'TP=.. TN=.. ..'} for cell tooltips
    model_q_f1_inst = {}  # label -> {bucket: {canon_question: F1}}
    model_q_conf_inst = {}  # label -> {bucket: {canon_question: 'TP=.. ..'}}
    for path in pred_files:
        try:
            label, yt, yp, q, sop_norm = load_prediction_source(path)
        except Exception as e:
            print(f"[06] skip {os.path.basename(path)}: {e}")
            continue
        sources[label] = (yt, yp, q, sop_norm)
        model_q_f1[label] = per_question_f1(yt, yp, q, average)
        model_q_conf[label] = per_question_confusion(yt, yp, q)
        f1_inst, _n_inst = per_question_f1_by_institution(yt, yp, q, sop_norm, inst_map, average)
        model_q_f1_inst[label] = f1_inst
        model_q_conf_inst[label] = per_question_confusion_by_institution(
            yt, yp, q, sop_norm, inst_map)
        print(f"[06] {label:16s} n={len(yt)} from {os.path.basename(path)}")

    if not sources:
        raise RuntimeError("No usable prediction sources discovered.")

    print(f"[06] per-question F1 averaging: {average}")

    # Reference GT/question space for the constant controls: the source with the
    # most rows (all baselines share the 361-row validation set).
    ref_label = max(sources, key=lambda k: len(sources[k][0]))
    ref_yt, _ref_yp, ref_q, ref_sop = sources[ref_label]
    control_q_f1 = constant_controls(ref_yt, ref_q, cfg.random_baseline_seed, average)
    control_q_f1_inst = constant_controls_by_institution(
        ref_yt, ref_q, ref_sop, inst_map, cfg.random_baseline_seed, average)
    _crng = np.random.default_rng(cfg.random_baseline_seed)
    control_q_conf = {
        "ALL-YES": per_question_confusion(ref_yt, np.ones_like(ref_yt), ref_q),
        "ALL-NO": per_question_confusion(ref_yt, np.zeros_like(ref_yt), ref_q),
        "RANDOM": per_question_confusion(ref_yt, _crng.integers(0, 2, size=len(ref_yt)), ref_q),
    }
    _crng2 = np.random.default_rng(cfg.random_baseline_seed)
    control_q_conf_inst = {
        "ALL-YES": per_question_confusion_by_institution(
            ref_yt, np.ones_like(ref_yt), ref_q, ref_sop, inst_map),
        "ALL-NO": per_question_confusion_by_institution(
            ref_yt, np.zeros_like(ref_yt), ref_q, ref_sop, inst_map),
        "RANDOM": per_question_confusion_by_institution(
            ref_yt, _crng2.integers(0, 2, size=len(ref_yt)), ref_q, ref_sop, inst_map),
    }

    summary = build_model_summary(sources, ref_yt, ref_q, ref_sop, inst_map,
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
        comp = build_comprehensive(probe_df, model_q_f1, control_q_f1,
                                   model_q_f1_inst, control_q_f1_inst)
        comp_csv = os.path.join(str(run_dir), f"per_question_comprehensive{suffix}.csv")
        comp.to_csv(comp_csv, index=False)
        print(f"[06] comprehensive per-question -> {comp_csv} ({len(comp)} rows)")
        outputs["comprehensive_csv"] = comp_csv
        # Per-cell tooltips: TP/TN/FP/FN behind every F1. Probe columns come from
        # the builder's confusion sidecars; model/control columns are computed
        # here from their predictions. Counts are averaging-independent, so this
        # matrix is shared by the macro and weighted HTML.
        titles_csv = _write_titles_matrix(comp, comp_csv, cfg, model_q_conf,
                                           control_q_conf, model_q_conf_inst,
                                           control_q_conf_inst)
        # Freeze checkpoint/group/question/n (cols 0-3); filter chips on
        # checkpoint (0) and question group (1); weighted-average footer by n
        # (col 3); question-distribution chart over (group, question, n).
        html_report.render(comp_csv, comp_csv.replace(".csv", ".html"),
                           title="AngioVision — Comprehensive Per-Question Leaderboard" + tag,
                           heat_from=5, freeze_cols=[0, 1, 2, 3], filter_cols=[0, 1],
                           weight_col=3, qchart_cols=[1, 2, 3],
                           titles_csv=titles_csv, title_key_cols=[0, 2])
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
