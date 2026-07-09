"""
Task Co-occurrence Dendrogram
------------------------------
Each included study is assigned exactly one `primary_task` label (Section 3.4,
Figure task_distribution). Reviewers repeatedly noted that this forced
single-label assignment obscures how much the ten primary task categories
actually overlap in practice: many studies report a `secondary_tasks` list
alongside the primary one (e.g. a vessel-segmentation paper that also performs
temporal sequence modelling).

Rather than have the authors subjectively merge "close" categories, this
script lets the co-occurrence structure speak for itself: it builds a
task x paper incidence matrix (primary_task OR secondary_tasks, restricted to
the ten canonical categories shown in Figure task_distribution), converts
co-occurrence into a Tanimoto distance between every pair of tasks, and runs
average-linkage (UPGMA) hierarchical clustering over that distance matrix.
The resulting dendrogram is included as-is; readers can cut the tree at
whatever height suits their own definition of "overlapping" categories.

Note: for binary/set-membership data (a task is either associated with a
paper or not, as here), the Tanimoto coefficient reduces algebraically to
the Jaccard index -- |A ∩ B| / |A ∪ B| -- so this is a naming/citation
choice, not a change to the underlying computation or the resulting numbers.

Method references:
  - Rogers, D. J., & Tanimoto, T. T. (1960). "A computer program for
    classifying plants." Science, 132(3434), 1115-1118.     -> similarity measure
  - Sokal, R. R., & Michener, C. D. (1958). "A statistical method for
    evaluating systematic relationships." University of Kansas Science
    Bulletin, 38, 1409-1438.                                -> UPGMA / average linkage
  - van Eck, N. J., & Waltman, L. (2010). "Software survey: VOSviewer, a
    computer program for bibliometric mapping." Scientometrics, 84(2),
    523-538.                                                 -> co-occurrence-based
                                                                  mapping in reviews
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# === GLOBAL FONT SETTINGS (consistent with 01_topical_analysis.py) ===
plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   17,
    "axes.labelsize":   15,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
})

# === CONFIGURATION ===
SLR_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = SLR_ROOT / "results" / "stage2_results.filtered.jsonl"
OUTPUT_DIR = SLR_ROOT / "analysis-results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PDF = OUTPUT_DIR / "task_dendrogram.pdf"

# The ten primary task categories shown in Figure task_distribution (i.e. the
# top-10 `primary_task` values, excluding "other"/"unknown"). Restricting the
# co-occurrence analysis to this fixed set keeps the dendrogram directly
# comparable to that figure.
CANONICAL_TASKS = [
    "Vessel Segmentation",
    "Object Detection & Localisation",
    "Catheter & Device Tracking",
    "Disease Classification & Grading",
    "Outcome Prediction",
    "Contrast Flow Analysis",
    "Image Enhancement & Denoising",
    "Temporal Sequence Modelling",
    "Background Subtraction",
    "Motion Correction & Registration",
]

# Free-text secondary_tasks entries that are unambiguous synonyms of a
# canonical category (schema drift, not a distinct task). Secondary_tasks
# entries not covered here or in CANONICAL_TASKS are left out of the
# co-occurrence analysis since they do not belong to the fixed ten-category
# scheme used throughout the rest of the paper.
SECONDARY_TASK_ALIASES = {
    "Motion Correction": "Motion Correction & Registration",
}


def load_paper_task_sets(path: Path):
    """Return a list of sets, one per paper, each containing every canonical
    task (primary + secondary) that paper touches."""
    paper_sets = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            task = rec.get("task", {}) or {}
            tasks_here = set()

            primary = task.get("primary_task")
            if primary:
                primary = primary.strip()
                if primary in CANONICAL_TASKS:
                    tasks_here.add(primary)

            for sec in (task.get("secondary_tasks") or []):
                sec = sec.strip()
                sec = SECONDARY_TASK_ALIASES.get(sec, sec)
                if sec in CANONICAL_TASKS:
                    tasks_here.add(sec)

            if tasks_here:
                paper_sets.append(tasks_here)

    return paper_sets


def build_tanimoto_distance_matrix(paper_sets, tasks):
    """Pairwise Tanimoto distance between tasks, based on the set of papers
    each task appears in (as primary or secondary). For binary/set-membership
    data this is algebraically identical to the Jaccard index."""
    task_papers = {
        t: {i for i, s in enumerate(paper_sets) if t in s}
        for t in tasks
    }

    n = len(tasks)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = task_papers[tasks[i]], task_papers[tasks[j]]
            union = a | b
            tanimoto_sim = len(a & b) / len(union) if union else 0.0
            d = 1.0 - tanimoto_sim
            dist[i, j] = d
            dist[j, i] = d

    return dist



# Validated categorical hues (blue, aqua -- slots 1-2 of the design system's
# fixed 8-hue theme, see dataviz skill references/palette.md), used in a
# fixed order for the two sub-threshold branch clusters. The muted ink gray
# recedes for the higher, less informative merges near the top of the tree.
CLUSTER_COLORS = ["#2a78d6", "#1baf7a"]
ABOVE_THRESHOLD_COLOR = "#52514e"
TEXT_PRIMARY = "#0b0b0b"


def main():
    paper_sets = load_paper_task_sets(INPUT_PATH)
    print(f"Loaded {len(paper_sets)} papers with at least one canonical task.")

    dist_matrix = build_tanimoto_distance_matrix(paper_sets, CANONICAL_TASKS)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")  # UPGMA

    merge_heights = Z[:, 2]
    x_floor = merge_heights.min() - 0.02  # zoom to where the tree actually branches
    # All merges in this analysis fall in a narrow ~0.90-1.00 band (co-occurrence
    # is rare across task categories), so a 0-1 axis would render every branch as
    # a nearly flat comb. Zooming to the observed range is what makes the actual
    # cluster structure -- which pairs are relatively more overlapping -- legible.
    color_threshold = 0.975  # splits out the two lowest, most defensible clusters

    from scipy.cluster.hierarchy import set_link_color_palette
    set_link_color_palette(CLUSTER_COLORS)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    dendrogram(
        Z,
        labels=CANONICAL_TASKS,
        orientation="right",       # leaves on the left, dissimilarity increases left-to-right
        leaf_font_size=13,
        color_threshold=color_threshold,
        above_threshold_color=ABOVE_THRESHOLD_COLOR,
        ax=ax,
    )

    ax.set_xlabel("Tanimoto distance  (1 − co-occurrence overlap)", color=TEXT_PRIMARY)
    ax.set_ylabel("")
    ax.set_xlim(x_floor, 1.005)
    ax.set_title("Task Category Co-occurrence (UPGMA clustering, Tanimoto distance)",
                 fontsize=15, color=TEXT_PRIMARY, pad=12)
    ax.axvline(1.0, color="#c9c8c2", linewidth=1, linestyle="--", zorder=0)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    plt.setp(ax.get_yticklabels(), color=TEXT_PRIMARY)
    plt.setp(ax.get_xticklabels(), color=TEXT_PRIMARY)

    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved dendrogram to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
