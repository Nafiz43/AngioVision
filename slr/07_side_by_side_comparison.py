"""
CLAIM 2024 Adherence Forest Plot  —  Temporal Comparison
AngioVision Systematic Review (PRISMA)

Side-by-side forest plot:
    Left panel  : studies with year <= SPLIT_YEAR  (e.g. ≤ 2024)
    Right panel : studies with year >  SPLIT_YEAR  (e.g. > 2024)

- Section colour bands on the Y-axis background replace individual row colours
- No n= annotations anywhere on the plot
- Section legend always visible, centred below both panels
- Wilson 95% CI; NA excluded from denominator

Usage:
    python3 claim2024_forest_plot.py
    python3 claim2024_forest_plot.py --input results/claim2024_results.jsonl
    python3 claim2024_forest_plot.py --input results/claim2024_results.jsonl \\
                                     --out_dir figures/ --split_year 2024
"""

import subprocess, sys

def _ensure(pkg, imp=None):
    try: __import__(imp or pkg)
    except ImportError:
        print(f"[bootstrap] Installing '{pkg}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("matplotlib"); _ensure("numpy"); _ensure("pandas")

import argparse, json, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

# ── Cluster definitions ───────────────────────────────────────────────────────
CLAIM2024_CLUSTERS = [
    {"cluster_id": "C01", "cluster_label": "AI identification & abstract completeness", "section": "Title/Abstract",    "items_covered": "1–2"},
    {"cluster_id": "C02", "cluster_label": "Background & study objectives",             "section": "Introduction",      "items_covered": "3–5"},
    {"cluster_id": "C03", "cluster_label": "Study design & data sources",               "section": "Methods",           "items_covered": "6–9"},
    {"cluster_id": "C04", "cluster_label": "Participants & image acquisition",          "section": "Methods",           "items_covered": "10–13"},
    {"cluster_id": "C05", "cluster_label": "Ground truth & annotation",                 "section": "Methods",           "items_covered": "14–16"},
    {"cluster_id": "C06", "cluster_label": "Data partitioning & class handling",        "section": "Methods",           "items_covered": "17–21"},
    {"cluster_id": "C07", "cluster_label": "Model architecture & training",             "section": "Methods",           "items_covered": "22–26"},
    {"cluster_id": "C08", "cluster_label": "Evaluation design & fairness",              "section": "Methods",           "items_covered": "27–31"},
    {"cluster_id": "C09", "cluster_label": "Performance & external validation",         "section": "Results",           "items_covered": "32–36"},
    {"cluster_id": "C10", "cluster_label": "Demographic & subgroup reporting",          "section": "Results",           "items_covered": "37–39"},
    {"cluster_id": "C11", "cluster_label": "Limitations, implications & future work",   "section": "Discussion",        "items_covered": "40–42"},
    {"cluster_id": "C12", "cluster_label": "Reproducibility & funding disclosure",      "section": "Reproducibility & Transparency", "items_covered": "43–44"},
]

# Section colours — used for both row bands and legend
SECTION_COLORS = {
    "Title/Abstract":    "#4C72B0",
    "Introduction":      "#55A868",
    "Methods":           "#C44E52",
    "Results":           "#DD8452",
    "Discussion":        "#8172B2",
    "Reproducibility & Transparency": "#937860",
}

# Light tint for row background bands (alpha blended manually)
SECTION_BAND = {k: v + "18" for k, v in SECTION_COLORS.items()}   # 18 = ~10% opacity hex

DEFAULT_INPUT   = "results/claim2024_results.jsonl"
DEFAULT_OUT_DIR = "figures"
DEFAULT_SPLIT   = 2025


# ── Wilson CI ─────────────────────────────────────────────────────────────────
def wilson_ci(p, n, z=1.96):
    if n == 0:
        return p, p
    denom  = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ── IO ────────────────────────────────────────────────────────────────────────
def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: records.append(json.loads(line))
            except json.JSONDecodeError: pass
    return records


def split_by_year(records, split_year):
    pre, post, unknown = [], [], []
    for rec in records:
        if "_error" in rec: continue
        try:
            y = int(rec.get("year"))
            (pre if y <= split_year else post).append(rec)
        except (TypeError, ValueError):
            unknown.append(rec)
    return pre, post, unknown


def compute_proportions(records):
    rows = []
    for c in CLAIM2024_CLUSTERS:
        cid = c["cluster_id"]
        yes = no = na = 0
        for rec in records:
            adh  = rec.get("claim2024_adherence", {})
            val  = None
            if isinstance(adh, dict):
                cell = adh.get(cid, {})
                val  = cell.get("adherence") if isinstance(cell, dict) else cell
            if   val == "Yes": yes += 1
            elif val == "No":  no  += 1
            elif val == "NA":  na  += 1
        n_total = yes + no
        prop    = yes / n_total if n_total > 0 else 0.0
        lo, hi  = wilson_ci(prop, n_total)
        rows.append({
            "cluster_id":    cid,
            "cluster_label": c["cluster_label"],
            "section":       c["section"],
            "items_covered": c["items_covered"],
            "n_yes": yes, "n_no": no, "n_na": na,
            "n_total": n_total,
            "proportion": prop,
            "ci_lower": lo, "ci_upper": hi,
        })
    return rows


# ── Section band drawing ──────────────────────────────────────────────────────
def _draw_section_bands(ax, rows_reversed):
    """
    Draw a coloured background band for every contiguous run of the same section.
    rows_reversed is in plot order (C12 at y=0, C01 at y=11).
    """
    n = len(rows_reversed)
    i = 0
    while i < n:
        sec   = rows_reversed[i]["section"]
        start = i
        while i < n and rows_reversed[i]["section"] == sec:
            i += 1
        end = i - 1
        ax.axhspan(
            start - 0.5, end + 0.5,
            color=SECTION_COLORS[sec], alpha=0.08, zorder=0, linewidth=0,
        )
        # Section label in the band (left edge, vertically centred)
        mid_y = (start + end) / 2
        ax.text(
            -0.02, mid_y,
            sec,
            va="center", ha="right",
            fontsize=9, color=SECTION_COLORS[sec],
            fontstyle="italic",
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )


# ── Single panel ─────────────────────────────────────────────────────────────
def _draw_panel(ax, prop_rows, n_studies, title, show_ylabels, show_right_axis):
    rows      = list(reversed(prop_rows))   # C01 at top (highest y index)
    n         = len(rows)
    y         = list(range(n))

    # Section background bands
    _draw_section_bands(ax, rows)

    # Reference lines
    ax.axvline(0.50, color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
    ax.axvline(0.75, color="#cccccc", linestyle=":",  linewidth=0.8, alpha=0.6, zorder=1)

    for i, row in enumerate(rows):
        p   = row["proportion"]
        lo  = row["proportion"] - row["ci_lower"]
        hi  = row["ci_upper"]   - row["proportion"]
        col = SECTION_COLORS[row["section"]]

        # CI line
        ax.plot([p - lo, p + hi], [y[i], y[i]],
                color=col, linewidth=1.8, solid_capstyle="round", zorder=3)
        # Caps
        for xpos in [p - lo, p + hi]:
            ax.plot([xpos, xpos], [y[i] - 0.16, y[i] + 0.16],
                    color=col, linewidth=1.5, zorder=3)
        # Diamond
        dx, dy = 0.012, 0.28
        diamond = plt.Polygon(
            [[p, y[i]+dy], [p+dx, y[i]], [p, y[i]-dy], [p-dx, y[i]]],
            closed=True, facecolor=col, edgecolor="white", linewidth=0.8, zorder=4,
        )
        ax.add_patch(diamond)

        # % label right of CI bar (no n= anywhere)
        ax.text(p + hi + 0.015, y[i], f"{p*100:.0f}%",
                va="center", fontsize=10.5, color="#333333", zorder=5)

    # Y-axis cluster labels
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(
            [r["cluster_label"] for r in rows],
            fontsize=11, color="#222222",
        )
    else:
        ax.set_yticklabels([""] * n)
        ax.tick_params(axis="y", length=0)

    # Right axis — items badge
    if show_right_axis:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(y)
        ax2.set_yticklabels(
            [f"Items {r['items_covered']}" for r in rows],
            fontsize=9.5, color="#888888",
        )
        ax2.tick_params(axis="y", length=0)
        ax2.spines[["top", "left", "bottom"]].set_visible(False)

    # Axis styling
    ax.set_xlim(0.0, 1.15)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Proportion  (95% Wilson CI)", fontsize=11, labelpad=6)
    # ax.set_title(f"{title}\n(N = {n_studies})", fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Ref-line text at bottom
    ax.text(0.50, -0.65, "50%", ha="center", fontsize=9, color="#aaaaaa")
    ax.text(0.75, -0.65, "75%", ha="center", fontsize=9, color="#cccccc")


# ── Main figure ───────────────────────────────────────────────────────────────
def draw_comparison_plot(pre_rows, post_rows, n_pre, n_post, split_year, out_dir):
    n   = len(CLAIM2024_CLUSTERS)
    fig_h = max(8, n * 0.70 + 3.0)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(22, fig_h),
        sharey=True,
        gridspec_kw={"wspace": 0.04},
    )

    _draw_panel(ax_l, pre_rows,  n_studies=n_pre,
                title=f"Studies  ≤ {split_year}",
                show_ylabels=True, show_right_axis=False)

    _draw_panel(ax_r, post_rows, n_studies=n_post,
                title=f"Studies  > {split_year}",
                show_ylabels=False, show_right_axis=True)

    # Vertical divider between panels
    fig.add_artist(
        plt.Line2D(
            [0.5, 0.5], [0.06, 0.97],
            transform=fig.transFigure,
            color="#dddddd", linewidth=1.0, linestyle="--",
        )
    )

    # Super-title
    fig.suptitle(
        "CLAIM 2024 Adherence by Thematic Cluster  —  Temporal Comparison\n",
                fontsize=15, fontweight="bold", y=1.02,
    )

    # Legend — one entry per section, centred below both panels
    handles = []
    seen    = []
    for c in CLAIM2024_CLUSTERS:
        if c["section"] not in seen:
            seen.append(c["section"])
            handles.append(
                mpatches.Patch(
                    facecolor=SECTION_COLORS[c["section"]],
                    edgecolor="white", linewidth=0.5,
                    label=c["section"],
                )
            )
    fig.legend(
        handles=handles,
        title="CLAIM 2024 Section",
        title_fontsize=10.5,
        fontsize=10,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.03),
        framealpha=0.95,
        edgecolor="#cccccc",
        handlelength=1.2,
        handleheight=0.9,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, "forest_plot_claim2024_comparison.pdf")
    png = os.path.join(out_dir, "forest_plot_claim2024_comparison.png")
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


# ── Summary CSV ───────────────────────────────────────────────────────────────
def save_comparison_csv(pre_rows, post_rows, split_year, out_dir):
    rows = []
    for pre, post in zip(pre_rows, post_rows):
        rows.append({
            "cluster_id":    pre["cluster_id"],
            "cluster_label": pre["cluster_label"],
            "section":       pre["section"],
            "items_covered": pre["items_covered"],
            f"pre_n":        pre["n_total"],
            f"pre_yes":      pre["n_yes"],
            f"pre_pct":      round(pre["proportion"]*100, 1),
            f"pre_ci_lo":    round(pre["ci_lower"]*100,   1),
            f"pre_ci_hi":    round(pre["ci_upper"]*100,   1),
            f"post_n":       post["n_total"],
            f"post_yes":     post["n_yes"],
            f"post_pct":     round(post["proportion"]*100, 1),
            f"post_ci_lo":   round(post["ci_lower"]*100,   1),
            f"post_ci_hi":   round(post["ci_upper"]*100,   1),
        })
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, "claim2024_comparison_proportions.csv")
    df.to_csv(csv, index=False)
    print(f"Comparison CSV: {csv}")


# ── Console table ─────────────────────────────────────────────────────────────
def print_console_table(pre_rows, post_rows, split_year):
    w = 78
    print("\n" + "="*w)
    print(f"  CLAIM 2024  |  ≤{split_year}  vs  >{split_year}")
    print("="*w)
    print(f"{'ID':<5} {'Cluster':<40}  {'≤'+str(split_year):>7}  {'>'+str(split_year):>7}  {'Δ':>7}")
    print("─"*w)
    for pre, post in zip(pre_rows, post_rows):
        d = post["proportion"] - pre["proportion"]
        a = "▲" if d > 0.02 else ("▼" if d < -0.02 else "~")
        print(f"{pre['cluster_id']:<5} {pre['cluster_label'][:39]:<40} "
              f" {pre['proportion']*100:>6.1f}%  {post['proportion']*100:>6.1f}%  "
              f"{a}{abs(d)*100:>5.1f}%")
    print("="*w)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CLAIM 2024 side-by-side temporal forest plot"
    )
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--out_dir",    default=DEFAULT_OUT_DIR)
    parser.add_argument("--split_year", default=DEFAULT_SPLIT, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: '{args.input}' not found."); sys.exit(1)

    records   = load_jsonl(args.input)
    valid     = [r for r in records if "_error" not in r]
    pre, post, unknown = split_by_year(valid, args.split_year)

    print(f"Loaded {len(records)} records ({len(valid)} valid)")
    print(f"  ≤ {args.split_year}: {len(pre)}   > {args.split_year}: {len(post)}   unknown year: {len(unknown)}")

    if not pre and not post:
        print("No records in either cohort. Aborting."); sys.exit(1)

    pre_rows  = compute_proportions(pre)
    post_rows = compute_proportions(post)

    print_console_table(pre_rows, post_rows, args.split_year)
    save_comparison_csv(pre_rows, post_rows, args.split_year, args.out_dir)
    draw_comparison_plot(pre_rows, post_rows,
                         n_pre=len(pre), n_post=len(post),
                         split_year=args.split_year,
                         out_dir=args.out_dir)

if __name__ == "__main__":
    main()