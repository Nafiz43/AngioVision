"""
Consolidated Analysis: Topical Trends & Venue Dominance
Merges and refactors topical_analysis.py and venue_dominance.py
Preserves 100% original functionality, styling, and output paths.
"""

# %% IMPORTS & CONFIGURATION
import json
import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
INPUT_PATH = Path("/data/Deep_Angiography/AngioVision/slr/results/stage2_results.jsonl")
OUTPUT_DIR = Path("/data/Deep_Angiography/AngioVision/slr/analysis-results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Venue normalization mapping (preserved from original)
VENUE_NORMALIZATION = {
    "Proc SPIE Int Soc Opt Eng": "SPIE",
    "International Journal of Computer Assisted Radiology and Surgery": "IJCARS",
    "MICCAI 2010, Part III, LNCS 6363": "MICCAI",
}

# Figure registry (preserved structure & paths)
FIGURES = {
    "temporal_trends": {
        "file": "temporal_trends.pdf",
        "desc": "Publication volume over time (from 2000 onwards)"
    },
    "modality_distribution": {
        "file": "modality_distribution.pdf",
        "desc": "Distribution of imaging modalities (top 10)"
    },
    "task_distribution": {
        "file": "task_distribution.pdf",
        "desc": "Distribution of primary clinical/research tasks (top 10)"
    },
    "anatomy_distribution": {
        "file": "anatomy_distribution.pdf",
        "desc": "Distribution of anatomical regions (top 10)"
    },
    "venue_stacked": {
        "file": "venue_analysis_stacked.pdf",
        "desc": "Top venues split into journal vs conference counts"
    },
}


# %% HELPER FUNCTIONS
def normalize_venue(v: str) -> str:
    """Normalize venue name using predefined mapping."""
    if not v:
        return v
    return VENUE_NORMALIZATION.get(v.strip(), v.strip())


def load_and_process_data(input_path: Path) -> Tuple[pd.DataFrame, Counter, Counter, Dict[str, str]]:
    """Single-pass loader that builds topical DF and venue counters simultaneously."""
    records = []
    raw_venue_counts = Counter()
    norm_venue_counts = Counter()
    venue_types = {}

    with input_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            si = d.get("study_identity", {})
            im = d.get("imaging", {})
            task = d.get("task", {})

            # Extract topical fields
            records.append({
                "year": si.get("year"),
                "modality": im.get("modality"),
                "primary_task": task.get("primary_task"),
                "anatomy": im.get("anatomy"),
            })

            # Extract venue fields
            raw_venue = si.get("journal_or_venue")
            if raw_venue:
                r_stripped = raw_venue.strip()
                raw_venue_counts[r_stripped] += 1

                n_venue = normalize_venue(r_stripped)
                norm_venue_counts[n_venue] += 1

                v_type = si.get("publication_type")
                if v_type:
                    venue_types[n_venue] = v_type.lower().strip()

    df = pd.DataFrame(records)
    return df, raw_venue_counts, norm_venue_counts, venue_types


# %% TOPICAL ANALYSIS FUNCTIONS
def plot_distribution(series: pd.Series, title: str, filename: Path, top_n: int = 10) -> None:
    """Helper to plot categorical distributions."""
    series = series.dropna()
    series = series[~series.str.lower().isin(["unknown", "other"])]
    counts = series.value_counts().head(top_n).sort_values()

    plt.figure(figsize=(8, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(counts)))
    counts.plot(kind="barh", color=colors)
    plt.xlabel("Count")
    plt.title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def run_topical_analysis(df: pd.DataFrame, norm_venue_counts: Counter, venue_types: Dict[str, str]) -> None:
    """Generate all topical distribution & venue comparison plots."""
    # 1. Temporal Trends
    df_year = df.dropna(subset=["year"]).copy()
    df_year["year"] = df_year["year"].astype(int)
    df_year = df_year[df_year["year"] >= 2000]

    year_range = pd.Series(range(2000, 2026))
    year_counts = df_year["year"].value_counts().reindex(year_range, fill_value=0)

    plt.figure(figsize=(9, 5))
    year_counts.sort_index().plot(kind="bar", color="skyblue")
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Number of Papers", fontsize=12)
    plt.title("Temporal Trends of Publications (2000–Present)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / FIGURES["temporal_trends"]["file"], dpi=300)
    plt.close()

    # 2. Categorical Distributions
    plot_distribution(df["modality"], "Imaging Modalities", OUTPUT_DIR / FIGURES["modality_distribution"]["file"])
    plot_distribution(df["primary_task"], "Primary Tasks", OUTPUT_DIR / FIGURES["task_distribution"]["file"])
    plot_distribution(df["anatomy"], "Anatomical Regions", OUTPUT_DIR / FIGURES["anatomy_distribution"]["file"])

    # 3. Venue Journal vs Conference Stacked Bar
    journal_counts = Counter({v: c for v, c in norm_venue_counts.items() if venue_types.get(v) == "journal"})
    conference_counts = Counter({v: c for v, c in norm_venue_counts.items() if venue_types.get(v) == "conference"})

    top_journals = journal_counts.most_common(5)
    top_conferences = conference_counts.most_common(5)
    labels = [v for v, _ in top_journals + top_conferences]

    journal_values = [journal_counts.get(v, 0) for v in labels]
    conference_values = [conference_counts.get(v, 0) for v in labels]

    plt.figure(figsize=(12, 6))
    x = range(len(labels))
    plt.bar(x, journal_values, label="Journal", color="#4C72B0")
    plt.bar(x, conference_values, bottom=journal_values, label="Conference", color="#DD8452")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Number of Papers", fontsize=12)
    plt.title("Top Venues (Journals vs Conferences)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / FIGURES["venue_stacked"]["file"], dpi=300)
    plt.close()


# %% VENUE DOMINANCE FUNCTIONS
def compute_dominance_metrics(venue_counts: Counter) -> Dict:
    """Calculate top-k shares, HHI, and singleton statistics."""
    total_papers = sum(venue_counts.values())
    sorted_counts = sorted(venue_counts.values(), reverse=True)
    proportions = np.array(sorted_counts) / total_papers
    cumulative = np.cumsum(proportions)
    ranks = np.arange(1, len(sorted_counts) + 1)

    def top_k_share(k):
        return sum(sorted_counts[:k]) / total_papers if sorted_counts else 0.0

    singletons = sum(1 for c in sorted_counts if c == 1)

    return {
        "total_papers": total_papers,
        "unique_venues": len(sorted_counts),
        "top1": top_k_share(1),
        "top3": top_k_share(3),
        "top5": top_k_share(5),
        "top10": top_k_share(10),
        "hhi": float(np.sum(proportions ** 2)),
        "singletons": singletons,
        "singleton_ratio": singletons / len(sorted_counts) if sorted_counts else 0.0,
        "proportions": proportions,
        "cumulative": cumulative,
        "ranks": ranks,
        "sorted_counts": sorted_counts,
    }


def run_venue_dominance(raw_venue_counts: Counter) -> None:
    """Generate dominance metrics, text report, and plots."""
    metrics = compute_dominance_metrics(raw_venue_counts)

    # Save stats
    txt_path = OUTPUT_DIR / "venue_dominance_stats.txt"
    with txt_path.open("w") as f:
        f.write("=== VENUE DOMINANCE ANALYSIS ===\n\n")
        f.write(f"Total papers: {metrics['total_papers']}\n")
        f.write(f"Total unique venues: {metrics['unique_venues']}\n\n")
        f.write("--- Dominance (Top-k Share) ----\n")
        f.write(f"Top 1 venue:  {metrics['top1']:.3f}\n")
        f.write(f"Top 3 venues: {metrics['top3']:.3f}\n")
        f.write(f"Top 5 venues: {metrics['top5']:.3f}\n")
        f.write(f"Top 10 venues:{metrics['top10']:.3f}\n\n")
        f.write("--- Concentration ----\n")
        f.write(f"HHI: {metrics['hhi']:.4f}\n\n")
        f.write("--- Long Tail ----\n")
        f.write(f"Singleton venues (1 paper): {metrics['singletons']}\n")
        f.write(f"Singleton ratio: {metrics['singleton_ratio']:.3f}\n")

    print(txt_path.read_text())

    # Zipf Plot
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    plt.plot(
        metrics["ranks"], metrics["sorted_counts"],
        color='#D90429', linewidth=2.5, marker='o',
        markersize=9, markerfacecolor='#FFFFFF',
        markeredgecolor='#D90429', markeredgewidth=1.8, zorder=3
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Venue Rank (log scale)", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Papers (log scale)", fontsize=12, fontweight='bold')
    plt.title("Zipf Plot of Publication Venues", fontsize=14, fontweight='bold', pad=10)

    ax.grid(True, which="major", linestyle="-", linewidth=0.8, color='#E5E7EB')
    plt.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color='#F3F4F6')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "venue_zipf_plot.pdf", dpi=300, facecolor='#FAFAFA')
    plt.close(fig)

    # Cumulative Dominance Curve
    fig2, ax2 = plt.subplots(figsize=(7, 5), facecolor='#FAFAFA')
    ax2.set_facecolor('#FAFAFA')

    plt.plot(
        metrics["ranks"], metrics["cumulative"],
        color='#023E8A', linewidth=2.5, marker='s',
        markersize=9, markerfacecolor='#FFFFFF',
        markeredgecolor='#023E8A', markeredgewidth=1.8, zorder=3
    )
    plt.xlabel("Number of Venues (Top-k)", fontsize=12, fontweight='bold')
    plt.ylabel("Cumulative Share of Papers", fontsize=12, fontweight='bold')
    plt.title("Venue Dominance Curve", fontsize=14, fontweight='bold', pad=10)

    ax2.set_ylim(0, 1)
    ax2.grid(True, which="major", linestyle="-", linewidth=0.8, color='#E5E7EB')
    plt.minorticks_on()
    ax2.grid(True, which="minor", linestyle=":", linewidth=0.5, color='#F3F4F6')

    plt.axhline(0.5, linestyle="--", color='#FF9F1C', linewidth=2, alpha=0.8)
    plt.axhline(0.8, linestyle="--", color='#2EC4B6', linewidth=2, alpha=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "venue_cumulative_curve.pdf", dpi=300, facecolor='#FAFAFA')
    plt.close(fig2)


# %% MAIN EXECUTION
if __name__ == "__main__":
    print("Loading & processing data...")
    df, raw_venues, norm_venues, v_types = load_and_process_data(INPUT_PATH)

    print("Running Topical Analysis...")
    run_topical_analysis(df, norm_venues, v_types)

    print("Running Venue Dominance Analysis...")
    run_venue_dominance(raw_venues)

    # Final Summary
    print("\n================ FIGURE SUMMARY ================")
    for key, meta in FIGURES.items():
        path = OUTPUT_DIR / meta["file"]
        print(f"- {meta['file']}")
        print(f"  Path       : {path}")
        print(f"  Description: {meta['desc']}\n")

    dom_files = ["venue_zipf_plot.pdf", "venue_cumulative_curve.pdf", "venue_dominance_stats.txt"]
    for f_name in dom_files:
        p = OUTPUT_DIR / f_name
        print(f"- {p}")
        print(f"  Path       : {p}\n")

    print("=" * 65)
    print("GENERATED OUTPUTS — FULL PATHS")
    print("=" * 65)

    out_dir = OUTPUT_DIR.resolve()
    print(f"\nBase Directory:\n{out_dir}\n")

    files_output = {
        "Topical Analysis": [
            "temporal_trends.pdf",
            "modality_distribution.pdf",
            "task_distribution.pdf",
            "anatomy_distribution.pdf",
        ],
        "Venue & Dominance Analysis": [
            "venue_analysis_stacked.pdf",
            "venue_zipf_plot.pdf",
            "venue_cumulative_curve.pdf",
        ],
        "Statistics & Reports": [
            "venue_dominance_stats.txt",
        ],
    }

    for category, files in files_output.items():
        print(f"{category}:")
        for f_name in files:
            full_path = out_dir / f_name
            print(f"  {f_name:<40} -> {full_path}")
        print()

    print("=" * 65)
    print("All analysis complete. Outputs are ready to use.")
    print("=" * 65 + "\n")