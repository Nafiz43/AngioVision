"""
Temporal vs. Non-Temporal Comparison Analysis
AngioVision Systematic Review

Reads the JSONL output from stage2_temporal_ablation.py and produces:
  1. temporal_comparison_paired.csv          — one row per (article × metric) pair
  2. temporal_comparison_summary.csv         — one row per article (primary metric only)
  3. temporal_comparison_stats.csv           — per-metric statistical summary
  4. temporal_comparison_plots.pdf           — multi-panel figure suite
  5. temporal_comparison_report.txt          — plain-text comparison table
  6. temporal_comparison_table.tex           — LaTeX per-metric statistics table
  7. temporal_comparison_table_standalone.tex — full compilable LaTeX document (preview)

Required LaTeX packages (listed at top of generated .tex files):
    booktabs, xcolor, colortbl, array, caption, makecell

Usage:
    python3 temporal_comparison_analysis.py
    python3 temporal_comparison_analysis.py --input results/temporal_ablation.jsonl
    python3 temporal_comparison_analysis.py --input results/temporal_ablation.jsonl --outdir results/
    python3 temporal_comparison_analysis.py --table-only   # LaTeX table only, skip PDF plots
"""

# ── Auto-install ──────────────────────────────────────────────────────────────
import subprocess, sys

def _ensure(pkg, imp=None):
    try:
        __import__(imp or pkg)
    except ImportError:
        print(f"[bootstrap] Installing '{pkg}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("pandas")
_ensure("matplotlib")
_ensure("seaborn")
_ensure("scipy")
_ensure("tabulate")
_ensure("numpy")

# ── Imports ───────────────────────────────────────────────────────────────────
import re, json, argparse, warnings
from pathlib import Path

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy   import stats
from tabulate import tabulate

warnings.filterwarnings("ignore")

DEFAULT_INPUT  = "results/temporal_ablation.jsonl"
DEFAULT_OUTDIR = "results/"

# ── Metric normalisation ──────────────────────────────────────────────────────
# Maps common variants → canonical name + polarity (higher = better → +1, lower = better → -1)
# NOTE: F1 Score is kept separate from Dice so both appear as distinct rows in the HTML table.

METRIC_MAP = {
    # Dice / DSC  (Sørensen–Dice only — F1 now has its own entry below)
    r"\b(dice|dsc|sorensen[-_]?dice|dice[-_]?coefficient)\b":  ("Dice",        +1),
    # F1 Score
    r"\b(f1[-_ ]?score?|f[-_]?measure|f1)\b":                  ("F1 Score",    +1),
    # IoU / Jaccard
    r"\b(iou|jaccard|intersection.over.union)\b":              ("IoU",         +1),
    # Accuracy
    r"\b(acc(uracy)?)\b":                                       ("Accuracy",    +1),
    # Sensitivity / Recall / TPR
    r"\b(sens(itivity)?|recall|tpr|true.positive.rate)\b":     ("Sensitivity", +1),
    # Specificity
    r"\b(spec(ificity)?|tnr)\b":                               ("Specificity", +1),
    # Precision / PPV
    r"\b(prec(ision)?|ppv)\b":                                 ("Precision",   +1),
    # AUC / AUROC
    r"\b(au[rc]|auroc|auc[-_ ]?roc|area.under)\b":            ("AUC",         +1),
    # PSNR
    r"\b(psnr|peak.signal)\b":                                 ("PSNR",        +1),
    # SSIM
    r"\b(ssim|structural.similarity)\b":                       ("SSIM",        +1),
    # MSE / RMSE
    r"\b(mse|mean.squared.error)\b":                           ("MSE",         -1),
    r"\b(rmse|root.mean.squared)\b":                           ("RMSE",        -1),
    # MAE
    r"\b(mae|mean.absolute.error)\b":                          ("MAE",         -1),
    # Hausdorff
    r"\b(hd95|hausdorff.95|95.hausdorff)\b":                  ("HD95",        -1),
    r"\b(hd|hausdorff)\b":                                     ("HD",          -1),
    # FPS / latency
    r"\b(fps|frames.per.second)\b":                            ("FPS",         +1),
    r"\b(inference.time|latency)\b":                           ("Latency",     -1),
    # mAP
    r"\b(map|mean.average.precision)\b":                       ("mAP",         +1),
    # Detection
    r"\b(clot.detection|lesion.detection)\b":                  ("Detection",   +1),
}

# ── Metric section layout (drives HTML table structure) ───────────────────────
# Add / remove sections or metrics here to customise the table.
# Any metric found in the data but absent from all sections appears in "Other metrics".

METRIC_SECTIONS = [
    {
        "label":     "Discrimination",
        "header_bg": "#EAF3DE",   # green-50
        "header_fg": "#3B6D11",   # green-600
        "bar_color": "#1D9E75",   # green-400
        "metrics":   ["AUC"],
    },
    {
        "label":     "Detection metrics",
        "header_bg": "#E6F1FB",   # blue-50
        "header_fg": "#0C447C",   # blue-800
        "bar_color": "#378ADD",   # blue-400
        "metrics":   ["Sensitivity", "Specificity", "Precision"],
    },
    {
        "label":     "Classification metrics",
        "header_bg": "#FAEEDA",   # amber-50
        "header_fg": "#633806",   # amber-900
        "bar_color": "#BA7517",   # amber-400
        "metrics":   ["Accuracy", "F1 Score"],
    },
    {
        "label":     "Segmentation metrics",
        "header_bg": "#FBEAF0",   # pink-50
        "header_fg": "#72243E",   # pink-800
        "bar_color": "#D4537E",   # pink-400
        "metrics":   ["Dice", "IoU"],
    },
    {
        "label":     "Image quality metrics",
        "header_bg": "#EEEDFE",   # purple-50
        "header_fg": "#3C3489",   # purple-800
        "bar_color": "#7F77DD",   # purple-400
        "metrics":   ["PSNR", "SSIM"],
    },
]


# ── Value parser ──────────────────────────────────────────────────────────────

def normalise_metric(raw: str):
    if not raw:
        return ("Unknown", +1)
    s = raw.strip().lower()
    for pattern, (canon, pol) in METRIC_MAP.items():
        if re.search(pattern, s, re.IGNORECASE):
            return (canon, pol)
    return (raw.strip().title(), +1)


def parse_value(raw: str):
    if raw is None:
        return None
    s = str(raw).strip().replace("%", "")
    s = re.split(r"[±(]", s)[0].strip()
    try:
        return float(s)
    except ValueError:
        return None


def pct_normalise(value, metric):
    pct_metrics = {"Dice","IoU","Accuracy","Sensitivity","Specificity",
                   "Precision","AUC","SSIM","Detection","F1 Score"}
    if metric in pct_metrics and value is not None and value > 1.0:
        return value / 100.0
    return value


# ── Data loading ──────────────────────────────────────────────────────────────

def load_confirmed(jsonl_path: str) -> list:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("confirmed_ablation") is True:
                records.append(rec)
    return records


def build_paired_df(records: list) -> pd.DataFrame:
    rows = []
    for i, rec in enumerate(records):
        title = rec.get("title") or f"Article_{i+1}"
        year  = rec.get("year")
        task  = rec.get("task") or "Unknown"
        pid   = f"A{i+1:02d}"

        comp = rec.get("comparison_table") or []
        if comp:
            metric_data = {}
            for row in comp:
                raw_m = row.get("metric", "")
                canon, pol = normalise_metric(raw_m)
                val = parse_value(row.get("value"))
                if val is None:
                    continue
                val = pct_normalise(val, canon)
                if canon not in metric_data:
                    metric_data[canon] = {"polarity": pol, "with": [], "without": []}
                if row.get("has_temporal") is True:
                    metric_data[canon]["with"].append(val)
                elif row.get("has_temporal") is False:
                    metric_data[canon]["without"].append(val)
            for metric, d in metric_data.items():
                if not d["with"] or not d["without"]:
                    continue
                w   = float(np.mean(d["with"]))
                wo  = float(np.mean(d["without"]))
                pol = d["polarity"]
                rows.append(_make_row(pid, title, year, task, metric, pol, w, wo, w - wo))
            if rows:
                continue

        with_results = (rec.get("with_temporal") or {}).get("results") or []
        wo_results   = (rec.get("without_temporal") or {}).get("results") or []
        with_map  = _results_to_map(with_results)
        wo_map    = _results_to_map(wo_results)
        for raw_m in set(with_map) & set(wo_map):
            canon, pol = normalise_metric(raw_m)
            w  = with_map[raw_m]
            wo = wo_map[raw_m]
            if w is None or wo is None:
                continue
            rows.append(_make_row(pid, title, year, task, canon, pol, w, wo, w - wo))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["direction"] = df.apply(
        lambda r: "improvement" if r["delta"] * r["polarity"] > 0
                  else ("degradation" if r["delta"] * r["polarity"] < 0 else "neutral"),
        axis=1,
    )
    return df


def _results_to_map(results):
    out = {}
    for r in results:
        m = (r.get("metric") or "").strip().lower()
        v = parse_value(r.get("value"))
        canon, _ = normalise_metric(m)
        out[m] = pct_normalise(v, canon) if v is not None else None
    return out


def _make_row(pid, title, year, task, metric, polarity, w, wo, delta):
    return {
        "article_id":    pid,
        "title":         title,
        "year":          year,
        "task":          task,
        "metric":        metric,
        "polarity":      polarity,
        "with_value":    round(w,     4),
        "without_value": round(wo,    4),
        "delta":         round(delta, 4),
        "delta_norm":    round(delta * polarity, 4),
    }


# ── Statistical tests ─────────────────────────────────────────────────────────

def run_stats(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for metric, grp in df.groupby("metric"):
        with_vals = grp["with_value"].dropna().values
        wo_vals   = grp["without_value"].dropna().values
        n = min(len(with_vals), len(wo_vals))
        if n < 2:
            continue
        w, wo = with_vals[:n], wo_vals[:n]
        pol = int(grp["polarity"].iloc[0])
        try:
            _, p_ttest  = stats.ttest_rel(w, wo)
        except Exception:
            p_ttest = np.nan
        try:
            _, p_wilcox = stats.wilcoxon(w - wo)
        except Exception:
            p_wilcox = np.nan

        results.append({
            "metric":          metric,
            "polarity":        "+ higher better" if pol > 0 else "- lower better",
            "n_pairs":         n,
            "mean_with":       round(float(np.mean(w)),  4),
            "mean_without":    round(float(np.mean(wo)), 4),
            "mean_delta":      round(float(np.mean(w - wo)), 4),
            "pct_improvement": round(100 * float(np.mean((w - wo) * pol > 0)), 1),
            "p_ttest":         round(p_ttest,  4) if not np.isnan(p_ttest)  else "N/A",
            "p_wilcoxon":      round(p_wilcox, 4) if not np.isnan(p_wilcox) else "N/A",
            "significant_005": (
                "Yes" if (not np.isnan(p_wilcox) and p_wilcox < 0.05) else
                "Yes" if (not np.isnan(p_ttest)  and p_ttest  < 0.05) else "No"
            ),
        })
    return pd.DataFrame(results)


# ── LaTeX table generator ─────────────────────────────────────────────────────

# LaTeX color names for each section (defined once, reused per section row)
_SECTION_LATEX_COLORS = {
    "Discrimination":       ("secGreenBg",  "secGreenFg",  "EAF3DE", "3B6D11"),
    "Detection metrics":    ("secBlueBg",   "secBlueFg",   "E6F1FB", "0C447C"),
    "Classification metrics":("secAmberBg", "secAmberFg",  "FAEEDA", "633806"),
    "Segmentation metrics": ("secPinkBg",   "secPinkFg",   "FBEAF0", "72243E"),
    "Image quality metrics":("secPurpleBg", "secPurpleFg", "EEEDFE", "3C3489"),
    "Other metrics":        ("secGrayBg",   "secGrayFg",   "F1EFE8", "444441"),
}
_DEFAULT_LATEX_COLORS = ("secGrayBg", "secGrayFg", "F1EFE8", "444441")


def _lx_esc(text: str) -> str:
    """Escape special LaTeX characters in a plain-text string."""
    if not text:
        return ""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("^",  r"\^{}"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def generate_latex_table(df_stats: pd.DataFrame, df: pd.DataFrame,
                         n_articles: int, out_path: str):
    """
    Write two LaTeX files:
      <out_path>                     — embeddable table fragment (paste into paper)
      <out_path>_standalone.tex      — full compilable document (for preview/review)

    Required packages (printed at top of both files):
        booktabs, xcolor, colortbl, array, caption, makecell
    """

    # ── Build stats lookup ────────────────────────────────────────────────
    stats_by_metric: dict = {}
    if not df_stats.empty:
        for _, row in df_stats.iterrows():
            stats_by_metric[row["metric"]] = row.to_dict()

    # Single-pair metrics (n < 2, so absent from df_stats — show with limited stats)
    if not df.empty:
        for metric, grp in df.groupby("metric"):
            if metric not in stats_by_metric:
                pol = int(grp["polarity"].iloc[0])
                w   = grp["with_value"].values
                wo  = grp["without_value"].values
                stats_by_metric[metric] = {
                    "metric":          metric,
                    "n_pairs":         len(grp),
                    "mean_with":       round(float(np.mean(w)),  4),
                    "mean_without":    round(float(np.mean(wo)), 4),
                    "mean_delta":      round(float(np.mean(w - wo)), 4),
                    "pct_improvement": round(100 * float(np.mean((w - wo) * pol > 0)), 1),
                    "p_wilcoxon":      "N/A",
                    "significant_005": "N/A",
                }

    # Overflow metrics not in any defined section → append "Other metrics"
    all_section_metrics = {m for s in METRIC_SECTIONS for m in s["metrics"]}
    overflow = sorted(m for m in stats_by_metric if m not in all_section_metrics)
    effective_sections = list(METRIC_SECTIONS)
    if overflow:
        effective_sections.append({
            "label":   "Other metrics",
            "metrics": overflow,
        })

    # ── Cell formatters ───────────────────────────────────────────────────

    NA = r"\textcolor{naGray}{---}"

    def _n_cell(sd):
        if sd is None:
            return NA
        return str(int(sd["n_pairs"]))

    def _val_cell(sd, key):
        if sd is None:
            return NA
        v = sd.get(key)
        if v is None:
            return NA
        return f"{float(v):.4f}"

    def _delta_cell(sd):
        if sd is None:
            return NA
        d = sd.get("mean_delta")
        if d is None:
            return NA
        d = float(d)
        if d > 0:
            return rf"\textcolor{{deltaGreen}}{{$+{d:.4f}$\,$\uparrow$}}"
        elif d < 0:
            return rf"\textcolor{{deltaRed}}{{${d:.4f}$\,$\downarrow$}}"
        else:
            return rf"\textcolor{{naGray}}{{${d:.4f}$\,$\rightarrow$}}"

    def _pct_cell(sd):
        if sd is None:
            return NA
        pct = sd.get("pct_improvement")
        if pct is None:
            return NA
        return rf"{float(pct):.1f}\%"

    def _p_cell(sd):
        if sd is None:
            return NA
        p = sd.get("p_wilcoxon", "N/A")
        if p == "N/A":
            return r"\textcolor{naGray}{\textit{N/A}}"
        return f"{float(p):.4f}"

    def _sig_cell(sd):
        if sd is None:
            return r"\textcolor{naGray}{\textit{N/A}}"
        s = sd.get("significant_005", "N/A")
        if s == "Yes":
            return r"\textcolor{sigGreen}{\textbf{Yes}}"
        if s == "No":
            return r"\textcolor{naGray}{No}"
        return r"\textcolor{naGray}{\textit{N/A}}"

    # ── Build table body ──────────────────────────────────────────────────

    body_lines = []
    for section in effective_sections:
        lbl  = section["label"]
        bg_name, fg_name, _, _ = _SECTION_LATEX_COLORS.get(lbl, _DEFAULT_LATEX_COLORS)
        # Section header row (spans all 8 columns)
        body_lines.append(
            rf"  \rowcolor{{{bg_name}}}"
            rf"\multicolumn{{8}}{{l}}{{"
            rf"\textcolor{{{fg_name}}}{{\small\textit{{{_lx_esc(lbl)}}}}}"
            rf"}} \\"
        )
        for metric in section["metrics"]:
            sd = stats_by_metric.get(metric)
            cols = [
                rf"\textbf{{{_lx_esc(metric)}}}",
                _n_cell(sd),
                _val_cell(sd, "mean_with"),
                _val_cell(sd, "mean_without"),
                _delta_cell(sd),
                _pct_cell(sd),
                _p_cell(sd),
                _sig_cell(sd),
            ]
            body_lines.append("  " + " & ".join(cols) + r" \\")
        body_lines.append(r"  \addlinespace[2pt]")

    # Remove trailing \addlinespace
    while body_lines and body_lines[-1].strip() == r"\addlinespace[2pt]":
        body_lines.pop()

    body = "\n".join(body_lines)

    # ── Collect all color definitions needed ─────────────────────────────

    used_sections = {s["label"] for s in effective_sections}
    color_defs = [
        r"% ── Section header colours ─────────────────────────────────────────────",
    ]
    seen_colors: set = set()
    for lbl in list(_SECTION_LATEX_COLORS.keys()) + ["Other metrics"]:
        if lbl not in used_sections:
            continue
        bg_name, fg_name, bg_hex, fg_hex = _SECTION_LATEX_COLORS.get(lbl, _DEFAULT_LATEX_COLORS)
        for cname, chex in [(bg_name, bg_hex), (fg_name, fg_hex)]:
            if cname not in seen_colors:
                color_defs.append(rf"\definecolor{{{cname}}}{{HTML}}{{{chex}}}")
                seen_colors.add(cname)

    color_defs += [
        r"% ── Delta & significance colours ───────────────────────────────────────",
        r"\definecolor{deltaGreen}{HTML}{059669}",
        r"\definecolor{deltaRed}{HTML}{DC2626}",
        r"\definecolor{sigGreen}{HTML}{0F6E56}",
        r"\definecolor{naGray}{HTML}{9CA3AF}",
    ]
    color_block = "\n".join(color_defs)

    # ── Table fragment (embeddable) ───────────────────────────────────────

    fragment = rf"""% ════════════════════════════════════════════════════════════════════
% Per-Metric Statistics: Temporal vs.\ Non-Temporal Comparison
% AngioVision Systematic Review
%
% Required packages in your preamble:
%   \usepackage{{booktabs}}
%   \usepackage[table,dvipsnames]{{xcolor}}
%   \usepackage{{colortbl}}
%   \usepackage{{array}}
%   \usepackage{{caption}}
%   \usepackage{{makecell}}
% ════════════════════════════════════════════════════════════════════

{color_block}

\begin{{table}}[htbp]
  \centering
  \caption{{Per-metric statistics: temporal vs.\ non-temporal comparison.
    Confirmed ablation studies---AngioVision SLR ($N = {n_articles}$).}}
  \label{{tab:temporal_ablation_stats}}
  \small
  \setlength{{\tabcolsep}}{{6pt}}
  \renewcommand{{\arraystretch}}{{1.3}}
  \begin{{tabular}}{{%
    >{{}}l                   % Metric
    c                         % N
    r                         % Mean with
    r                         % Mean w/o
    r                         % Mean delta
    r                         % % improved
    r                         % p (Wilcoxon)
    c                         % Sig?
  }}
  \toprule
  \makecell[l]{{\textbf{{Metric}}}}
    & \textbf{{N}}
    & \makecell[r]{{\textbf{{Mean with}}\\\textbf{{temporal}}}}
    & \makecell[r]{{\textbf{{Mean w/o}}\\\textbf{{temporal}}}}
    & \makecell[r]{{\textbf{{Mean}}\\\textbf{{$\Delta$}}}}
    & \makecell[r]{{\textbf{{\%}}\\\textbf{{improved}}}}
    & \makecell[r]{{\textbf{{$p$}}\\\textbf{{(Wilcoxon)}}}}
    & \textbf{{Sig?}} \\
  \midrule
{body}
  \bottomrule
  \end{{tabular}}
  \vspace{{4pt}}
  \begin{{minipage}}{{\linewidth}}
    \footnotesize
    \textit{{Note:}}
    $N$ = number of article pairs contributing to each metric.
    $\Delta$ = mean with temporal $-$ mean without temporal
    (positive $\uparrow$ = temporal component improves performance).
    Wilcoxon signed-rank test; significance threshold $p < 0.05$.
    Metrics marked \textcolor{{naGray}}{{---}} were not reported in
    any confirmed ablation study.
  \end{{minipage}}
\end{{table}}
"""

    Path(out_path).write_text(fragment, encoding="utf-8")
    print(f"LaTeX table  → {out_path}")

    # ── Standalone compilable document ────────────────────────────────────

    standalone_path = str(out_path).replace(".tex", "_standalone.tex")
    standalone = rf"""\documentclass{{article}}
\usepackage[margin=2cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage[table,dvipsnames]{{xcolor}}
\usepackage{{colortbl}}
\usepackage{{array}}
\usepackage{{caption}}
\usepackage{{makecell}}
\usepackage{{microtype}}

\begin{{document}}
\pagestyle{{empty}}

\input{{{Path(out_path).name}}}

\end{{document}}
"""
    Path(standalone_path).write_text(standalone, encoding="utf-8")
    print(f"Standalone   → {standalone_path}")


# ── Plotting helpers ──────────────────────────────────────────────────────────

PALETTE = {
    "with":        "#2563EB",
    "without":     "#DC2626",
    "improvement": "#16A34A",
    "degradation": "#DC2626",
    "neutral":     "#6B7280",
}

def _style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "figure.facecolor":  "white",
        "axes.facecolor":    "#F9FAFB",
    })


def plot_paired_bars(df_primary, ax):
    df_s = df_primary.sort_values("delta_norm", ascending=True).reset_index(drop=True)
    n, w = len(df_s), 0.35
    idx  = np.arange(n)
    ax.barh(idx + w/2, df_s["with_value"],    w, color=PALETTE["with"],    alpha=0.85, label="With temporal")
    ax.barh(idx - w/2, df_s["without_value"], w, color=PALETTE["without"], alpha=0.85, label="Without temporal")
    for bar in ax.patches:
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.3f}", va="center", ha="left", fontsize=7)
    labels = [f"{r.article_id}: {r.title[:30]}…\n({r.metric})" for _, r in df_s.iterrows()]
    ax.set_yticks(idx); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Metric value")
    ax.set_title("Performance: with vs. without temporal component", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)


def plot_delta_distribution(df, ax):
    deltas = df["delta_norm"].dropna()
    if deltas.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center"); return
    ax.bar(range(len(deltas)), sorted(deltas),
           color=[PALETTE["improvement"] if v >= 0 else PALETTE["degradation"]
                  for v in sorted(deltas)],
           alpha=0.8, width=0.6)
    ax.axhline(0,            color="black", linewidth=1.2, linestyle="--")
    ax.axhline(deltas.mean(), color="navy",  linewidth=1.5, linestyle=":",
               label=f"Mean Δ = {deltas.mean():+.4f}")
    ax.set_xlabel("Article (sorted by Δ)")
    ax.set_ylabel("Normalised Δ (positive = temporal helps)")
    ax.set_title("Performance delta distribution", fontweight="bold")
    ax.legend(fontsize=8)


def plot_forest(df, ax):
    df_s = df.sort_values("delta_norm").reset_index(drop=True)
    for i, row in df_s.iterrows():
        c = (PALETTE["improvement"] if row["delta_norm"] > 0
             else PALETTE["degradation"] if row["delta_norm"] < 0
             else PALETTE["neutral"])
        ax.plot(row["delta_norm"], i, "o", color=c, markersize=6, zorder=3)
        ax.hlines(i, 0, row["delta_norm"], color=c, alpha=0.5, linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", zorder=2)
    ax.set_yticks(np.arange(len(df_s)))
    ax.set_yticklabels([f"{r.article_id} – {r.metric}" for _, r in df_s.iterrows()], fontsize=7)
    ax.set_xlabel("Normalised Δ  (positive = temporal helps)")
    ax.set_title("Forest plot: temporal vs. non-temporal", fontweight="bold")
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color=PALETTE["improvement"], label="Improvement"),
        mpatches.Patch(color=PALETTE["degradation"], label="Degradation"),
        mpatches.Patch(color=PALETTE["neutral"],      label="Neutral"),
    ], fontsize=8, loc="lower right")


def plot_task_breakdown(df, ax):
    tasks = df["task"].value_counts().index.tolist()
    data  = [df[df["task"] == t]["delta_norm"].dropna().values for t in tasks]
    data, tasks = zip(*[(d, t) for d, t in zip(data, tasks) if len(d) > 0]) if data else ([], [])
    if not data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center"); return
    bp = ax.boxplot(data, vert=False, patch_artist=True, medianprops=dict(color="black", linewidth=2))
    for patch, d in zip(bp["boxes"], data):
        patch.set_facecolor(PALETTE["improvement"] if np.median(d) > 0 else PALETTE["degradation"])
        patch.set_alpha(0.6)
    for j, (d, y) in enumerate(zip(data, range(1, len(data)+1))):
        ax.scatter(d, np.full_like(d, y) + np.random.normal(0, 0.08, len(d)),
                   color="black", s=18, alpha=0.7, zorder=3)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(range(1, len(tasks)+1))
    ax.set_yticklabels([t[:35] for t in tasks], fontsize=8)
    ax.set_xlabel("Normalised Δ"); ax.set_title("Task-wise performance delta", fontweight="bold")


def plot_metric_scatter(df, ax):
    metrics = df["metric"].unique()
    cmap    = plt.cm.get_cmap("tab10", len(metrics))
    mc      = {m: cmap(i) for i, m in enumerate(metrics)}
    for _, row in df.iterrows():
        ax.scatter(row["without_value"], row["with_value"],
                   color=mc[row["metric"]], s=60, alpha=0.75,
                   edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(row["article_id"], (row["without_value"], row["with_value"]),
                    fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    all_v = pd.concat([df["with_value"], df["without_value"]]).dropna()
    if not all_v.empty:
        lo, hi = all_v.min() * 0.95, all_v.max() * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.5, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=mc[m], label=m) for m in metrics],
              fontsize=7, loc="lower right")
    ax.set_xlabel("Without temporal"); ax.set_ylabel("With temporal")
    ax.set_title("With vs. without scatter\n(above diagonal = temporal helps)", fontweight="bold")


def plot_win_pie(df, ax):
    counts = df["direction"].value_counts()
    ax.pie(counts.values, labels=counts.index,
           colors=[PALETTE.get(l, "#9CA3AF") for l in counts.index],
           autopct="%1.0f%%", startangle=140,
           wedgeprops=dict(edgecolor="white", linewidth=1.5),
           textprops=dict(fontsize=9))
    ax.set_title("Outcome distribution\n(all metrics, all articles)", fontweight="bold")


# ── Text report ───────────────────────────────────────────────────────────────

def write_text_report(df, df_stats, records, out_path):
    SEP, sep = "=" * 80, "─" * 80
    lines = [SEP, "  ANGIOVISION SLR — TEMPORAL vs. NON-TEMPORAL COMPARISON REPORT", SEP, ""]
    n_pairs = len(df)
    n_imp = int((df["direction"] == "improvement").sum())
    n_deg = int((df["direction"] == "degradation").sum())
    n_neu = int((df["direction"] == "neutral").sum())
    lines += ["OVERVIEW", sep,
              tabulate([
                  ["Confirmed ablation articles",  df["article_id"].nunique()],
                  ["Total metric pairs",            n_pairs],
                  ["Improvements",  f"{n_imp} ({100*n_imp/n_pairs:.1f}%)"],
                  ["Degradations",  f"{n_deg} ({100*n_deg/n_pairs:.1f}%)"],
                  ["Neutral",       f"{n_neu} ({100*n_neu/n_pairs:.1f}%)"],
                  ["Mean norm. Δ",  f"{df['delta_norm'].mean():+.4f}"],
              ], tablefmt="simple_outline"), ""]
    lines += ["PER-ARTICLE RESULTS", sep]
    tbl = []
    for _, r in df.sort_values(["article_id", "metric"]).iterrows():
        arrow = "↑" if r["direction"] == "improvement" else ("↓" if r["direction"] == "degradation" else "→")
        tbl.append([r["article_id"], r["title"][:40], r.get("year",""), r["task"][:30],
                    r["metric"], f"{r['with_value']:.4f}", f"{r['without_value']:.4f}",
                    f"{r['delta_norm']:+.4f} {arrow}"])
    lines += [tabulate(tbl, headers=["ID","Title","Year","Task","Metric",
                                     "With","W/o","Δ(norm)"], tablefmt="simple_outline"), ""]
    if not df_stats.empty:
        lines += ["STATISTICAL SUMMARY", sep,
                  tabulate(df_stats.values.tolist(), headers=df_stats.columns.tolist(),
                           tablefmt="simple_outline"), ""]
    lines += ["AUTHORS' CONCLUSIONS", sep]
    for i, rec in enumerate(records):
        c = rec.get("conclusion_on_temporal")
        if c:
            lines.append(f"A{i+1:02d}. {(rec.get('title') or '')[:60]}")
            words, cur = c.split(), ""
            for w in words:
                if len(cur) + len(w) + 1 > 74:
                    lines.append(f"     {cur}"); cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                lines.append(f"     {cur}")
            lines.append("")
    lines.append(SEP)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Text report  → {out_path}")


# ── Console summary ───────────────────────────────────────────────────────────

def print_console_summary(df, df_stats):
    SEP, sep = "=" * 65, "─" * 65
    n = len(df)
    print(f"\n{SEP}\n  TEMPORAL vs. NON-TEMPORAL — SUMMARY\n{SEP}")
    print(tabulate([
        ["Articles with confirmed ablation", df["article_id"].nunique()],
        ["Metric pair comparisons",          n],
        ["Temporal helps", f"{(df.direction=='improvement').sum()} ({100*(df.direction=='improvement').mean():.1f}%)"],
        ["Temporal hurts", f"{(df.direction=='degradation').sum()} ({100*(df.direction=='degradation').mean():.1f}%)"],
        ["Neutral",        f"{(df.direction=='neutral').sum()} ({100*(df.direction=='neutral').mean():.1f}%)"],
        ["Mean Δ (norm)",  f"{df.delta_norm.mean():+.4f}"],
        ["Median Δ (norm)",f"{df.delta_norm.median():+.4f}"],
    ], tablefmt="simple_outline"))
    if not df_stats.empty:
        print(f"\n{sep}\n  PER-METRIC STATISTICS")
        print(tabulate(df_stats[["metric","n_pairs","mean_with","mean_without",
                                  "mean_delta","pct_improvement",
                                  "p_wilcoxon","significant_005"]].values.tolist(),
                       headers=["Metric","N","Mean With","Mean W/o",
                                 "Mean Δ","% Improved","p (Wilcoxon)","Sig?"],
                       tablefmt="simple_outline"))
    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Temporal vs. Non-Temporal comparison — plots, stats, and HTML table"
    )
    parser.add_argument("--input",      default=DEFAULT_INPUT,
                        help=f"JSONL from stage2_temporal_ablation.py (default: {DEFAULT_INPUT})")
    parser.add_argument("--outdir",     default=DEFAULT_OUTDIR,
                        help=f"Output directory (default: {DEFAULT_OUTDIR})")
    parser.add_argument("--table-only", action="store_true",
                        help="Generate only the HTML table, skip PDF plots")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    stem = Path(args.outdir) / "temporal_comparison"

    # ── Load ────────────────────────────────────────────────────────────
    print(f"Loading confirmed ablation records from: {args.input}")
    records = load_confirmed(args.input)
    if not records:
        print("No confirmed ablation records found. Nothing to do.")
        return
    print(f"  → {len(records)} confirmed ablation article(s) loaded.")

    # ── Build paired DataFrame ──────────────────────────────────────────
    df = build_paired_df(records)
    if df.empty:
        print("Could not extract any numeric paired values. Check your JSONL data.")
        return
    print(f"  → {len(df)} (article × metric) pairs extracted.")

    df_primary = (df.sort_values("metric")
                    .groupby("article_id", as_index=False)
                    .first())

    # ── Statistical tests ────────────────────────────────────────────────
    df_stats = run_stats(df)

    # ── CSVs ────────────────────────────────────────────────────────────
    df.to_csv(f"{stem}_paired.csv", index=False)
    print(f"Paired CSV   → {stem}_paired.csv")
    df_primary.to_csv(f"{stem}_summary.csv", index=False)
    print(f"Summary CSV  → {stem}_summary.csv")
    if not df_stats.empty:
        df_stats.to_csv(f"{stem}_stats.csv", index=False)
        print(f"Stats CSV    → {stem}_stats.csv")

    # ── LaTeX table ──────────────────────────────────────────────────────
    generate_latex_table(df_stats, df, len(records), f"{stem}_table.tex")

    if args.table_only:
        print_console_summary(df, df_stats)
        return

    # ── Text report ──────────────────────────────────────────────────────
    write_text_report(df, df_stats, records, f"{stem}_report.txt")

    # ── Plots ────────────────────────────────────────────────────────────
    _style()
    pdf_path = f"{stem}_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(df_primary) * 0.6 + 2)))
        plot_paired_bars(df_primary, ax1)
        plot_delta_distribution(df, ax2)
        fig.suptitle("AngioVision SLR — Temporal Ablation Analysis",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.4 + 2)))
        plot_forest(df, ax)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig = plt.figure(figsize=(18, 7))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
        plot_task_breakdown(df, fig.add_subplot(gs[0]))
        plot_metric_scatter(df, fig.add_subplot(gs[1]))
        plot_win_pie(df,        fig.add_subplot(gs[2]))
        fig.suptitle("Task-wise, Scatter, and Outcome Distribution",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        d = pdf.infodict()
        d["Title"] = "AngioVision SLR — Temporal Ablation Comparison"

    print(f"PDF plots    → {pdf_path}")
    print_console_summary(df, df_stats)


if __name__ == "__main__":
    main()