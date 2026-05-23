"""
CLAIM 2024 Adherence Forest Plot
AngioVision Systematic Review (PRISMA)

Reads the JSONL output of claim2024_extraction.py and produces a
publication-quality grouped forest plot (PDF + PNG).

12 thematic clusters, Wilson 95% CIs, section colour-coding.

NA responses are excluded from each cluster's denominator so the
proportion is: Yes / (Yes + No).

Usage:
    python3 claim2024_forest_plot.py
    python3 claim2024_forest_plot.py --input results/claim2024_results.jsonl
    python3 claim2024_forest_plot.py --input results/claim2024_results.jsonl --out_dir figures/
"""

# ── Auto-install dependencies ─────────────────────────────────────────────────
import subprocess
import sys

def _ensure(package, import_name=None):
    name = import_name or package
    try:
        __import__(name)
    except ImportError:
        print(f"[bootstrap] Installing '{package}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

_ensure("matplotlib")
_ensure("numpy")
_ensure("pandas")

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Cluster metadata (must stay in sync with claim2024_extraction.py) ─────────
CLAIM2024_CLUSTERS = [
    {"cluster_id": "C01", "cluster_label": "AI identification & abstract completeness", "section": "Title/Abstract",   "items_covered": "1–2"},
    {"cluster_id": "C02", "cluster_label": "Background & study objectives",             "section": "Introduction",     "items_covered": "3–5"},
    {"cluster_id": "C03", "cluster_label": "Study design & data sources",               "section": "Methods",          "items_covered": "6–9"},
    {"cluster_id": "C04", "cluster_label": "Participants & image acquisition",          "section": "Methods",          "items_covered": "10–13"},
    {"cluster_id": "C05", "cluster_label": "Ground truth & annotation",                 "section": "Methods",          "items_covered": "14–16"},
    {"cluster_id": "C06", "cluster_label": "Data partitioning & class handling",        "section": "Methods",          "items_covered": "17–21"},
    {"cluster_id": "C07", "cluster_label": "Model architecture & training",             "section": "Methods",          "items_covered": "22–26"},
    {"cluster_id": "C08", "cluster_label": "Evaluation design & fairness",              "section": "Methods",          "items_covered": "27–31"},
    {"cluster_id": "C09", "cluster_label": "Performance & external validation",         "section": "Results",          "items_covered": "32–36"},
    {"cluster_id": "C10", "cluster_label": "Demographic & subgroup reporting",          "section": "Results",          "items_covered": "37–39"},
    {"cluster_id": "C11", "cluster_label": "Limitations, implications & future work",   "section": "Discussion",       "items_covered": "40–42"},
    {"cluster_id": "C12", "cluster_label": "Reproducibility & funding disclosure",      "section": "Other Information","items_covered": "43–44"},
]

SECTION_COLORS = {
    "Title/Abstract":   "#4C72B0",
    "Introduction":     "#55A868",
    "Methods":          "#C44E52",
    "Results":          "#DD8452",
    "Discussion":       "#8172B2",
    "Other Information":"#937860",
}

DEFAULT_INPUT  = "results/claim2024_results.jsonl"
DEFAULT_OUT_DIR = "figures"


# ── Wilson confidence interval ────────────────────────────────────────────────
def wilson_ci(p: float, n: int, z: float = 1.96):
    """Return (lower, upper) Wilson score 95% CI. Returns (p, p) when n == 0."""
    if n == 0:
        return p, p
    denom  = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ── Load JSONL ────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


# ── Compute cluster-level adherence proportions ───────────────────────────────
def compute_proportions(records: list[dict]) -> list[dict]:
    """
    For each cluster compute:
      - n_yes, n_no, n_na, n_total (=yes+no, NA excluded)
      - proportion = n_yes / n_total
      - Wilson 95% CI (lower, upper)
    """
    rows = []
    for c in CLAIM2024_CLUSTERS:
        cid  = c["cluster_id"]
        yes  = no = na = 0
        for rec in records:
            if "_error" in rec:
                continue
            adh = rec.get("claim2024_adherence", {})
            val = None
            if isinstance(adh, dict):
                cell = adh.get(cid, {})
                if isinstance(cell, dict):
                    val = cell.get("adherence")
                elif isinstance(cell, str):
                    val = cell
            if val == "Yes":
                yes += 1
            elif val == "No":
                no += 1
            elif val == "NA":
                na += 1

        n_total = yes + no          # denominator excludes NA
        prop    = yes / n_total if n_total > 0 else 0.0
        lo, hi  = wilson_ci(prop, n_total)

        rows.append({
            "cluster_id":    cid,
            "cluster_label": c["cluster_label"],
            "section":       c["section"],
            "items_covered": c["items_covered"],
            "n_yes":         yes,
            "n_no":          no,
            "n_na":          na,
            "n_total":       n_total,
            "proportion":    prop,
            "ci_lower":      lo,
            "ci_upper":      hi,
        })
    return rows


# ── Forest plot ───────────────────────────────────────────────────────────────
def draw_forest_plot(prop_rows: list[dict], n_studies: int, out_dir: str):
    """
    Draw a publication-quality horizontal forest plot.

    Layout (bottom-up order so most-important clusters appear at top):
    - Y axis: cluster labels + items badge
    - X axis: proportion 0–1
    - Diamond markers with Wilson CI whiskers
    - Section-coloured markers; dashed reference lines at 0.50 and 0.75
    - Alternating row shading
    """

    # Reverse order so C01 appears at top of the figure
    rows = list(reversed(prop_rows))
    n    = len(rows)

    labels    = [r["cluster_label"] for r in rows]
    sections  = [r["section"]       for r in rows]
    items_cov = [r["items_covered"] for r in rows]
    props     = [r["proportion"]    for r in rows]
    ci_lo     = [r["proportion"] - r["ci_lower"]  for r in rows]   # error below
    ci_hi     = [r["ci_upper"]  - r["proportion"] for r in rows]   # error above
    colors    = [SECTION_COLORS[s] for s in sections]
    n_totals  = [r["n_total"] for r in rows]

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig_h = max(6, n * 0.62 + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    y = list(range(n))

    # Alternating row shading
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.45, i + 0.45, color="#f5f5f5", zorder=0)

    # Reference lines
    ax.axvline(0.50, color="#999999", linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)
    ax.axvline(0.75, color="#bbbbbb", linestyle=":",  linewidth=0.9, alpha=0.5, zorder=1)

    # Plot CI whiskers + diamond markers
    for i in range(n):
        p  = props[i]
        lo = ci_lo[i]
        hi = ci_hi[i]
        c  = colors[i]

        # Whisker line
        ax.plot([p - lo, p + hi], [y[i], y[i]],
                color=c, linewidth=1.6, solid_capstyle="round", zorder=3)

        # Cap ticks
        for xpos in [p - lo, p + hi]:
            ax.plot([xpos, xpos], [y[i] - 0.18, y[i] + 0.18],
                    color=c, linewidth=1.4, zorder=3)

        # Diamond marker
        dx, dy = 0.013, 0.32
        diamond = plt.Polygon(
            [[p, y[i] + dy], [p + dx, y[i]], [p, y[i] - dy], [p - dx, y[i]]],
            closed=True, facecolor=c, edgecolor="white", linewidth=0.8, zorder=4,
        )
        ax.add_patch(diamond)

        # Percentage annotation (right of CI)
        pct_x = p + hi + 0.015
        ax.text(pct_x, y[i], f"{p*100:.0f}%",
                va="center", fontsize=8, color="#444444", zorder=5)

        # n annotation (left margin, small)
        ax.text(-0.005, y[i], f"n={n_totals[i]}",
                va="center", ha="right", fontsize=7, color="#888888", zorder=5)

    # Y-axis: cluster labels
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    # Items-covered badges — drawn as text just inside the left margin
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    ax2.set_yticklabels(
        [f"Items {r}" for r in items_cov],
        fontsize=7.5, color="#777777",
    )
    ax2.yaxis.set_tick_params(length=0)

    # Axes formatting
    ax.set_xlim(-0.01, 1.16)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(
        "Proportion of Studies Fulfilling CLAIM 2024 Cluster  (95% Wilson CI, NA excluded)",
        fontsize=10, labelpad=8,
    )
    ax.set_title(
        f"CLAIM 2024 Adherence by Thematic Cluster\n",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax2.spines[["top", "left", "bottom"]].set_visible(False)

    # Reference line annotations
    ax.text(0.50, -0.65, "50%", ha="center", fontsize=7.5, color="#999999")
    ax.text(0.75, -0.65, "75%", ha="center", fontsize=7.5, color="#bbbbbb")

    # Section colour legend
    seen   = []
    handles = []
    for c in CLAIM2024_CLUSTERS:
        if c["section"] not in seen:
            seen.append(c["section"])
            handles.append(
                mpatches.Patch(color=SECTION_COLORS[c["section"]], label=c["section"])
            )
    ax.legend(
        handles=handles,
        title="CLAIM 2024 Section",
        title_fontsize=8.5,
        fontsize=8,
        loc="lower right",
        framealpha=0.92,
        edgecolor="#cccccc",
    )

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "forest_plot_claim2024_grouped.pdf")
    png_path = os.path.join(out_dir, "forest_plot_claim2024_grouped.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    return pdf_path, png_path


# ── Summary table to CSV ──────────────────────────────────────────────────────
def save_summary_csv(prop_rows: list[dict], out_dir: str):
    df = pd.DataFrame(prop_rows)
    df["proportion_pct"] = (df["proportion"] * 100).round(1)
    df["ci_lower_pct"]   = (df["ci_lower"]   * 100).round(1)
    df["ci_upper_pct"]   = (df["ci_upper"]   * 100).round(1)
    csv_path = os.path.join(out_dir, "claim2024_cluster_proportions.csv")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Cluster proportions CSV: {csv_path}")
    return csv_path


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate CLAIM 2024 grouped forest plot from extraction JSONL"
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help=f"JSONL output from claim2024_extraction.py (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--out_dir", default=DEFAULT_OUT_DIR,
        help=f"Directory for output figures and CSV (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found.")
        sys.exit(1)

    records = load_jsonl(args.input)
    valid   = [r for r in records if "_error" not in r]
    print(f"Loaded {len(records)} records ({len(valid)} without extraction errors).")

    if not valid:
        print("No valid records found. Aborting.")
        sys.exit(1)

    prop_rows = compute_proportions(records)   # uses all records; errors auto-excluded inside

    # Print console table
    print("\nCLAIM 2024 Cluster Adherence Rates")
    print(f"{'─'*75}")
    print(f"{'ID':<5} {'Cluster':<44} {'Yes':>4} {'No':>4} {'NA':>4} {'N':>4}  {'% Yes':>6}")
    print(f"{'─'*75}")
    for r in prop_rows:
        print(
            f"{r['cluster_id']:<5} {r['cluster_label'][:43]:<44} "
            f"{r['n_yes']:>4} {r['n_no']:>4} {r['n_na']:>4} {r['n_total']:>4}  "
            f"{r['proportion']*100:>5.1f}%"
        )
    print(f"{'─'*75}")

    save_summary_csv(prop_rows, args.out_dir)
    draw_forest_plot(prop_rows, n_studies=len(valid), out_dir=args.out_dir)


if __name__ == "__main__":
    main()