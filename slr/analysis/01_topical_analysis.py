"""
Consolidated Analysis: Topical Trends & Venue Dominance
Merges and refactors topical_analysis.py and venue_dominance.py
Preserves 100% original functionality, styling, and output paths.

Rev: anatomy grouping (Cardiac / Neuro / Other Thoracic / Abdominopelvic / Peripheral / Multi-region)
     modality grouping (X-ray + Fluoroscopy merged)
     venue stacked bar: improved readability with narrower bars and larger fonts
     venue normalization: short-form abbreviations for all 10 top venues
     venue dominance curve: caption with yellow/green line definitions
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

# === GLOBAL FONT SETTINGS ===
plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   17,
    "axes.labelsize":   15,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  13,
})

# === CONFIGURATION ===
INPUT_PATH = Path("/data/Deep_Angiography/AngioVision/slr/results/stage2_results.jsonl")
OUTPUT_DIR = Path("/data/Deep_Angiography/AngioVision/slr/analysis-results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Venue normalization: raw JSONL string → short display label ───────────────
# Keys are the exact strings that appear in stage2_results.jsonl.
# Multiple raw variants map to the same short form to handle
# different database export styles (PubMed, IEEE Xplore, Scopus, etc.).
VENUE_NORMALIZATION = {
    # ── Journals ────────────────────────────────────────────────────────────
    # IJCARS
    "International Journal of Computer Assisted Radiology and Surgery": "IJCARS",
    "Int J Comput Assist Radiol Surg":                                  "IJCARS",
    "Int. J. Comput. Assist. Radiol. Surg.":                            "IJCARS",

    # Medical Physics
    "Medical Physics":                                                  "Med. Phys.",
    "Med Phys":                                                         "Med. Phys.",
    "Med. Phys.":                                                       "Med. Phys.",

    # Computers in Biology and Medicine
    "Computers in Biology and Medicine":                                "CBM",
    "Comput Biol Med":                                                  "CBM",
    "Comput. Biol. Med.":                                               "CBM",

    # Computerized Medical Imaging and Graphics
    "Computerized Medical Imaging and Graphics":                        "CMIG",
    "Comput Med Imaging Graph":                                         "CMIG",
    "Comput. Med. Imaging Graph.":                                      "CMIG",

    # Diagnostics
    "Diagnostics":                                                      "Diagnostics",
    "Diagnostics (Basel)":                                              "Diagnostics",

    # ── Conferences ─────────────────────────────────────────────────────────
    # SPIE
    "Proc SPIE Int Soc Opt Eng":                                        "SPIE",
    "Proceedings of SPIE":                                              "SPIE",
    "Proc. SPIE":                                                       "SPIE",
    "SPIE Medical Imaging":                                             "SPIE",

    # EMBC
    "Annual International Conference of the IEEE Engineering in Medicine and Biology Society": "EMBC",
    "Conf Proc IEEE Eng Med Biol Soc":                                  "EMBC",
    "IEEE EMBC":                                                        "EMBC",
    "Annu Int Conf IEEE Eng Med Biol Soc":                              "EMBC",

    # BIBM
    "IEEE International Conference on Bioinformatics and Biomedicine":  "BIBM",
    "IEEE BIBM":                                                        "BIBM",
    "Proc IEEE Int Conf Bioinformatics Biomedicine":                    "BIBM",

    # ISBI
    "IEEE International Symposium on Biomedical Imaging":               "ISBI",
    "IEEE ISBI":                                                        "ISBI",
    "Proc IEEE Int Symp Biomed Imaging":                                "ISBI",
    "IEEE International Symposium on Biomedical Imaging (ISBI)":        "ISBI",

    # ISMR
    "International Symposium on Medical Robotics":                      "ISMR",
    "International Symposium on Medical Robotics (ISMR)":               "ISMR",
    "Int Symp Med Robot":                                               "ISMR",

    # ── Legacy entry (kept for backward compatibility) ───────────────────────
    "MICCAI 2010, Part III, LNCS 6363":                                 "MICCAI",
}

# ── Anatomy grouping ──────────────────────────────────────────────────────────
ANATOMY_NORMALIZATION_MAP: Dict[str, str] = {
    # Cardiac
    "cardiac":          "Cardiac",
    "coronary":         "Cardiac",
    "heart":            "Cardiac",
    "myocardial":       "Cardiac",
    "aortic":           "Cardiac",
    "aorta":            "Cardiac",

    # Neuro (Cerebral / Spinal)
    "cerebral":         "Neuro (Cerebral/Spinal)",
    "spinal":           "Neuro (Cerebral/Spinal)",
    "brain":            "Neuro (Cerebral/Spinal)",
    "intracranial":     "Neuro (Cerebral/Spinal)",
    "neurovascular":    "Neuro (Cerebral/Spinal)",
    "cranial":          "Neuro (Cerebral/Spinal)",
    "neuro":            "Neuro (Cerebral/Spinal)",

    # Other Thoracic
    "lung":             "Other Thoracic",
    "pulmonary":        "Other Thoracic",
    "intercostal":      "Other Thoracic",
    "thoracic":         "Other Thoracic",
    "chest":            "Other Thoracic",
    "bronchial":        "Other Thoracic",
    "tracheal":         "Other Thoracic",

    # Abdominopelvic
    "abdominal":        "Abdominopelvic",
    "abdomen":          "Abdominopelvic",
    "pelvic":           "Abdominopelvic",
    "pelvis":           "Abdominopelvic",
    "renal":            "Abdominopelvic",
    "hepatic":          "Abdominopelvic",
    "mesenteric":       "Abdominopelvic",
    "splenic":          "Abdominopelvic",
    "visceral":         "Abdominopelvic",
    "portal":           "Abdominopelvic",
    "celiac":           "Abdominopelvic",

    # Peripheral
    "peripheral":       "Peripheral",
    "femoral":          "Peripheral",
    "iliac":            "Peripheral",
    "lower extremity":  "Peripheral",
    "upper extremity":  "Peripheral",
    "limb":             "Peripheral",
    "tibial":           "Peripheral",
    "popliteal":        "Peripheral",
    "brachial":         "Peripheral",
    "subclavian":       "Peripheral",
    "carotid":          "Peripheral",

    # Multi-region
    "multi":            "Multi-region",
    "multi-region":     "Multi-region",
    "multi region":     "Multi-region",
    "multiple":         "Multi-region",
    "multiple regions": "Multi-region",
    "whole body":       "Multi-region",
    "whole-body":       "Multi-region",
    "systemic":         "Multi-region",
    "generalized":      "Multi-region",
    "generalised":      "Multi-region",
}

# ── Modality grouping ─────────────────────────────────────────────────────────
MODALITY_NORMALIZATION_MAP: Dict[str, str] = {
    "x-ray":            "X-ray / Fluoroscopy",
    "xray":             "X-ray / Fluoroscopy",
    "x ray":            "X-ray / Fluoroscopy",
    "radiograph":       "X-ray / Fluoroscopy",
    "radiography":      "X-ray / Fluoroscopy",
    "fluoroscopy":      "X-ray / Fluoroscopy",
    "fluoroscopic":     "X-ray / Fluoroscopy",
    "fluoroscope":      "X-ray / Fluoroscopy",
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
        "desc": "Distribution of anatomical regions (grouped: Cardiac / Neuro / Other Thoracic / Abdominopelvic / Peripheral / Multi-region)"
    },
    "venue_stacked": {
        "file": "venue_analysis_stacked.pdf",
        "desc": "Top venues (abbreviated) split into journal vs conference counts"
    },
}


# %% HELPER FUNCTIONS
def normalize_venue(v: str) -> str:
    """Normalize venue name using predefined mapping."""
    if not v:
        return v
    return VENUE_NORMALIZATION.get(v.strip(), v.strip())


def normalize_anatomy(val: str) -> str:
    """Map raw anatomy string to one of the six canonical groups."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return val
    key = str(val).lower().strip()
    return ANATOMY_NORMALIZATION_MAP.get(key, val.strip())


def normalize_modality(val: str) -> str:
    """Merge X-ray and fluoroscopy variants into a single label."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return val
    key = str(val).lower().strip()
    return MODALITY_NORMALIZATION_MAP.get(key, val.strip())


def load_and_process_data(input_path: Path) -> Tuple[pd.DataFrame, Counter, Counter, Dict[str, str]]:
    """Single-pass loader that builds topical DF and venue counters simultaneously."""
    records = []
    raw_venue_counts  = Counter()
    norm_venue_counts = Counter()
    venue_types = {}

    with input_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            d    = json.loads(line)
            si   = d.get("study_identity", {})
            im   = d.get("imaging", {})
            task = d.get("task", {})

            records.append({
                "year":         si.get("year"),
                "modality":     im.get("modality"),
                "primary_task": task.get("primary_task"),
                "anatomy":      im.get("anatomy"),
            })

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
    plt.xlabel("Count", fontsize=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def run_topical_analysis(df: pd.DataFrame, norm_venue_counts: Counter, venue_types: Dict[str, str]) -> None:
    """Generate all topical distribution & venue comparison plots."""

    # Apply normalisations before any plotting
    modality_series = df["modality"].map(
        lambda x: normalize_modality(x) if pd.notna(x) else x
    )
    anatomy_series = df["anatomy"].map(
        lambda x: normalize_anatomy(x) if pd.notna(x) else x
    )

    # 1. Temporal Trends
    df_year = df.dropna(subset=["year"]).copy()
    df_year["year"] = df_year["year"].astype(int)
    df_year = df_year[df_year["year"] >= 2000]

    year_range  = pd.Series(range(2000, 2026))
    year_counts = df_year["year"].value_counts().reindex(year_range, fill_value=0)

    plt.figure(figsize=(9, 5))
    year_counts.sort_index().plot(kind="bar", color="skyblue")
    plt.xlabel("Year", fontsize=15)
    plt.ylabel("Number of Papers", fontsize=15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / FIGURES["temporal_trends"]["file"], dpi=300)
    plt.close()

    # 2. Categorical Distributions
    plot_distribution(modality_series,    "Imaging Modalities", OUTPUT_DIR / FIGURES["modality_distribution"]["file"])
    plot_distribution(df["primary_task"], "Primary Tasks",      OUTPUT_DIR / FIGURES["task_distribution"]["file"])
    plot_distribution(anatomy_series,     "Anatomical Regions", OUTPUT_DIR / FIGURES["anatomy_distribution"]["file"])

    # 3. Venue Journal vs Conference Stacked Bar (IMPROVED READABILITY)
    journal_counts    = Counter({v: c for v, c in norm_venue_counts.items() if venue_types.get(v) == "journal"})
    conference_counts = Counter({v: c for v, c in norm_venue_counts.items() if venue_types.get(v) == "conference"})

    top_journals    = journal_counts.most_common(5)
    top_conferences = conference_counts.most_common(5)
    labels = [v for v, _ in top_journals + top_conferences]

    journal_values    = [journal_counts.get(v, 0)    for v in labels]
    conference_values = [conference_counts.get(v, 0) for v in labels]

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(labels))
    width = 0.5  # Narrower bar width for improved readability
    
    ax.bar(x, journal_values, width,
           label="Journal", color="#4C72B0")
    ax.bar(x, conference_values, width, bottom=journal_values,
           label="Conference", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=14,  # Increased from 12
    )
    ax.set_ylabel("Number of Papers", fontsize=16, fontweight="bold")
    ax.set_xlabel("Venue", fontsize=16, fontweight="bold")
    ax.legend(fontsize=14, loc="upper right")
    ax.tick_params(axis="y", labelsize=14)

    fig.subplots_adjust(bottom=0.25)
    fig.savefig(
        OUTPUT_DIR / FIGURES["venue_stacked"]["file"],
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# %% VENUE DOMINANCE FUNCTIONS
def compute_dominance_metrics(venue_counts: Counter) -> Dict:
    """Calculate top-k shares, HHI, and singleton statistics."""
    total_papers  = sum(venue_counts.values())
    sorted_counts = sorted(venue_counts.values(), reverse=True)
    proportions   = np.array(sorted_counts) / total_papers
    cumulative    = np.cumsum(proportions)
    ranks         = np.arange(1, len(sorted_counts) + 1)

    def top_k_share(k):
        return sum(sorted_counts[:k]) / total_papers if sorted_counts else 0.0

    singletons = sum(1 for c in sorted_counts if c == 1)

    return {
        "total_papers":    total_papers,
        "unique_venues":   len(sorted_counts),
        "top1":            top_k_share(1),
        "top3":            top_k_share(3),
        "top5":            top_k_share(5),
        "top10":           top_k_share(10),
        "hhi":             float(np.sum(proportions ** 2)),
        "singletons":      singletons,
        "singleton_ratio": singletons / len(sorted_counts) if sorted_counts else 0.0,
        "proportions":     proportions,
        "cumulative":      cumulative,
        "ranks":           ranks,
        "sorted_counts":   sorted_counts,
    }


def run_venue_dominance(raw_venue_counts: Counter) -> None:
    """Generate dominance metrics, text report, and plots."""
    metrics = compute_dominance_metrics(raw_venue_counts)

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
    plt.xlabel("Venue Rank (log scale)",        fontsize=15, fontweight='bold')
    plt.ylabel("Number of Papers (log scale)",  fontsize=15, fontweight='bold')
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, color='#E5E7EB')
    plt.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color='#F3F4F6')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "venue_zipf_plot.pdf", dpi=300, facecolor='#FAFAFA')
    plt.close(fig)

    # Cumulative Dominance Curve with defined captions
    fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor='#FAFAFA')
    ax2.set_facecolor('#FAFAFA')
    
    # Main cumulative curve
    plt.plot(
        metrics["ranks"], metrics["cumulative"],
        color='#023E8A', linewidth=2.5, marker='s',
        markersize=9, markerfacecolor='#FFFFFF',
        markeredgecolor='#023E8A', markeredgewidth=1.8, zorder=3,
        label="Cumulative Share"
    )
    
    # Reference lines with explicit labels
    line_50 = plt.axhline(0.5, linestyle="--", color='#FF9F1C', linewidth=2.5, alpha=0.85, label="50% threshold (Yellow)")
    line_80 = plt.axhline(0.8, linestyle="--", color='#2EC4B6', linewidth=2.5, alpha=0.85, label="80% threshold (Green)")
    
    plt.xlabel("Number of Venues (Top-k)",       fontsize=15, fontweight='bold')
    plt.ylabel("Cumulative Share of Papers",      fontsize=15, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, which="major", linestyle="-", linewidth=0.8, color='#E5E7EB')
    plt.minorticks_on()
    ax2.grid(True, which="minor", linestyle=":", linewidth=0.5, color='#F3F4F6')
    
    # Enhanced legend with caption definitions
    ax2.legend(fontsize=12, loc="lower right", framealpha=0.95, edgecolor='black')
    
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