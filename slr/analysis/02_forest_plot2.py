import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================
OUT_PDF = Path("forest_plot_claim2024.pdf")

# =========================
# CLAIM2024 CRITERIA
# Mongan et al. (2024) - Checklist for AI in Medical Imaging
# 14 items across 5 sections
# =========================

CLAIM2024_ITEMS = {
    # SECTION 1: Title / Abstract
    "Identifies study as AI/ML-based [Title]":               ("Title/Abstract",    0.82),
    "Structured abstract with AI components [Abstract]":      ("Title/Abstract",    0.61),

    # SECTION 2: Introduction
    "Scientific and clinical background stated":              ("Introduction",      0.78),
    "Study objectives clearly defined":                       ("Introduction",      0.91),

    # SECTION 3: Methods
    "Study design described (prospective/retrospective)":     ("Methods",           0.74),
    "Data sources and acquisition described":                 ("Methods",           0.88),
    "Eligibility criteria specified":                         ("Methods",           0.67),
    "Ground truth / reference standard defined":              ("Methods",           0.72),
    "Model architecture described in detail":                 ("Methods",           0.69),
    "Training / validation / test split reported":            ("Methods",           0.58),
    "Hyperparameters and training procedure stated":          ("Methods",           0.44),
    "Performance metrics pre-specified":                      ("Methods",           0.63),

    # SECTION 4: Results
    "Participant / image flow diagram or table":              ("Results",           0.39),
    "Model performance with confidence intervals":            ("Results",           0.51),

    # SECTION 5: Discussion
    "Study limitations discussed":                            ("Discussion",        0.83),
    "Clinical implications addressed":                        ("Discussion",        0.71),
    "Potential biases acknowledged":                          ("Discussion",        0.55),
    "Comparison with prior work":                             ("Discussion",        0.76),
}

# =========================
# EXAMPLE DATA
# Simulated N=100 studies; proportions from CLAIM2024_ITEMS above
# Wilson 95% CI
# =========================
N = 100

def wilson_ci(p, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    phat = p
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / denom
    margin = (z * math.sqrt((phat*(1-phat)/n) + (z**2/(4*n**2)))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

labels, sections, means, lowers, uppers = [], [], [], [], []

for label, (section, prop) in CLAIM2024_ITEMS.items():
    lo, hi = wilson_ci(prop, N)
    labels.append(label)
    sections.append(section)
    means.append(prop)
    lowers.append(prop - lo)
    uppers.append(hi - prop)

# Reverse for bottom-up display
labels   = labels[::-1]
sections = sections[::-1]
means    = means[::-1]
lowers   = lowers[::-1]
uppers   = uppers[::-1]

# =========================
# COLOR PALETTE PER SECTION
# =========================
SECTION_COLORS = {
    "Title/Abstract": "#4C72B0",
    "Introduction":   "#55A868",
    "Methods":        "#C44E52",
    "Results":        "#DD8452",
    "Discussion":     "#8172B2",
}

point_colors = [SECTION_COLORS[s] for s in sections]

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(13, 8))

y = list(range(len(labels)))

for i in range(len(labels)):
    ax.errorbar(
        means[i], y[i],
        xerr=[[lowers[i]], [uppers[i]]],
        fmt='D',
        color=point_colors[i],
        markersize=8,
        capsize=5,
        elinewidth=1.6,
        markeredgewidth=1.2,
        markeredgecolor="white",
        linestyle='none',
        zorder=3,
    )
    # Percentage annotation
    ax.text(
        means[i] + uppers[i] + 0.012,
        y[i],
        f"{means[i]*100:.0f}%",
        va='center',
        fontsize=8.5,
        color="#333333",
    )

# Reference lines
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)
ax.axvline(0.75, color="#aaaaaa", linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)

# Alternating row shading
for i in range(len(labels)):
    if i % 2 == 0:
        ax.axhspan(i - 0.45, i + 0.45, color="#f5f5f5", zorder=0)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Proportion of Studies Fulfilling CLAIM2024 Item  (95% Wilson CI)", fontsize=10)
ax.set_title(
    "Forest Plot: CLAIM2024 Adherence in Systematic Literature Review\n",
        fontsize=12, fontweight="bold", pad=12
)
ax.set_xlim(0, 1.12)
ax.set_ylim(-0.7, len(labels) - 0.3)
ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)

# Legend
legend_handles = [
    mpatches.Patch(color=col, label=sec)
    for sec, col in SECTION_COLORS.items()
]
ax.legend(
    handles=legend_handles,
    title="CLAIM2024 Section",
    title_fontsize=9,
    fontsize=8.5,
    loc="lower right",
    framealpha=0.9,
    edgecolor="#cccccc",
)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight", dpi=200)
plt.savefig("forest_plot_claim2024.png", bbox_inches="tight", dpi=200)
print(f"Saved: {OUT_PDF}")
print("Saved: forest_plot_claim2024.png")