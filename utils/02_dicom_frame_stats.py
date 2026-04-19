#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import math
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm


# =========================================================
# PATH CONFIG
# =========================================================
ROOT_DIR = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
OUTPUT_DIR = Path("/data/Deep_Angiography/DICOM-metadata-stats")

FRAME_STATS_CSV = OUTPUT_DIR / "frame_statistics.csv"
HIST_PNG = OUTPUT_DIR / "number_of_frames_histogram.png"
BOX_PNG = OUTPUT_DIR / "number_of_frames_boxplot.png"
QQ_PNG = OUTPUT_DIR / "number_of_frames_qqplot.png"
STATS_TXT = OUTPUT_DIR / "number_of_frames_basic_stats.txt"


# =========================================================
# PART 1: FRAME STATS GENERATION
# =========================================================
def path_to_file_url(p: Path) -> str:
    """
    Create a file:// URL that many terminals/IDEs can open.
    Note: Some remote HPC terminals won't click-open these, but the URL is still correct.
    """
    return "file://" + quote(str(p))


def count_files_in_dir(d: Path) -> int:
    """Count only files directly inside d (not recursive)."""
    try:
        return sum(1 for x in d.iterdir() if x.is_file())
    except Exception:
        return 0


def generate_frame_stats(root_dir: Path, output_csv: Path) -> None:
    root_dir = Path(root_dir)
    output_csv = Path(output_csv)

    print(f"Scanning for 'frames' directories under: {root_dir}")
    frames_dirs = [p for p in root_dir.rglob("frames") if p.is_dir()]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not frames_dirs:
        print("No 'frames' directories found.")
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "outer_dir_name",
                "inner_dir_name",
                "number_of_frames",
                "frames_dir_path",
                "frames_dir_url"
            ])
        return

    rows = []
    for frames_dir in tqdm(frames_dirs, desc="Counting frames", unit="frames_dir"):
        # Expected structure: .../<outer>/<inner>/frames
        inner_dir = frames_dir.parent
        outer_dir = inner_dir.parent if inner_dir.parent is not None else None

        outer_dir_name = outer_dir.name if outer_dir is not None else "NA"
        inner_dir_name = inner_dir.name if inner_dir is not None else "NA"
        frame_count = count_files_in_dir(frames_dir)

        rows.append([
            outer_dir_name,
            inner_dir_name,
            frame_count,
            str(frames_dir),
            path_to_file_url(frames_dir),
        ])

    rows.sort(key=lambda x: x[2], reverse=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "outer_dir_name",
            "inner_dir_name",
            "number_of_frames",
            "frames_dir_path",
            "frames_dir_url"
        ])
        writer.writerows(rows)

    print(f"Frame statistics CSV written to: {output_csv}")


# =========================================================
# PART 2: ANALYZE FRAME COUNT DISTRIBUTION
# =========================================================
def analyze_frame_statistics(input_csv: Path, output_dir: Path) -> None:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hist_png = output_dir / "number_of_frames_histogram.png"
    box_png = output_dir / "number_of_frames_boxplot.png"
    qq_png = output_dir / "number_of_frames_qqplot.png"
    stats_txt = output_dir / "number_of_frames_basic_stats.txt"

    df = pd.read_csv(input_csv)

    if "number_of_frames" not in df.columns:
        raise ValueError("Column 'number_of_frames' not found in the CSV.")

    values = pd.to_numeric(df["number_of_frames"], errors="coerce").dropna()

    if len(values) == 0:
        raise ValueError("No valid numeric values found in 'number_of_frames'.")

    values_np = values.to_numpy()
    n = len(values_np)

    # -------------------------
    # Basic stats
    # -------------------------
    mean_val = np.mean(values_np)
    median_val = np.median(values_np)
    std_val = np.std(values_np, ddof=1) if n > 1 else 0.0
    var_val = np.var(values_np, ddof=1) if n > 1 else 0.0
    min_val = np.min(values_np)
    max_val = np.max(values_np)
    q1 = np.percentile(values_np, 25)
    q3 = np.percentile(values_np, 75)
    iqr = q3 - q1
    skew_val = stats.skew(values_np, bias=False) if n > 2 else float("nan")
    kurtosis_val = stats.kurtosis(values_np, bias=False) if n > 3 else float("nan")

    # -------------------------
    # Normality tests
    # -------------------------
    if n <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(values_np)
    else:
        shapiro_stat, shapiro_p = float("nan"), float("nan")

    if n >= 8:
        dagostino_stat, dagostino_p = stats.normaltest(values_np)
    else:
        dagostino_stat, dagostino_p = float("nan"), float("nan")

    anderson_result = stats.anderson(values_np, dist="norm")

    # -------------------------
    # Interpretation
    # -------------------------
    interpretation_lines = []

    if not math.isnan(shapiro_p):
        if shapiro_p < 0.05:
            interpretation_lines.append(
                f"Shapiro-Wilk test: p={shapiro_p:.6g} < 0.05 -> reject normality."
            )
        else:
            interpretation_lines.append(
                f"Shapiro-Wilk test: p={shapiro_p:.6g} >= 0.05 -> data does not significantly deviate from normality."
            )
    else:
        interpretation_lines.append(
            "Shapiro-Wilk test not reported because sample size is greater than 5000."
        )

    if not math.isnan(dagostino_p):
        if dagostino_p < 0.05:
            interpretation_lines.append(
                f"D'Agostino-Pearson test: p={dagostino_p:.6g} < 0.05 -> reject normality."
            )
        else:
            interpretation_lines.append(
                f"D'Agostino-Pearson test: p={dagostino_p:.6g} >= 0.05 -> data does not significantly deviate from normality."
            )
    else:
        interpretation_lines.append(
            "D'Agostino-Pearson test not run because at least 8 samples are required."
        )

    interpretation_lines.append(
        "Also inspect the histogram and Q-Q plot visually, since statistical tests can be overly sensitive for large sample sizes."
    )

    # -------------------------
    # Histogram
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(values_np, bins=30, edgecolor="black")
    plt.title("Distribution of number_of_frames")
    plt.xlabel("number_of_frames")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(hist_png, dpi=300)
    plt.close()

    # -------------------------
    # Boxplot
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.boxplot(values_np, vert=False)
    plt.title("Boxplot of number_of_frames")
    plt.xlabel("number_of_frames")
    plt.tight_layout()
    plt.savefig(box_png, dpi=300)
    plt.close()

    # -------------------------
    # Q-Q plot
    # -------------------------
    plt.figure(figsize=(8, 8))
    stats.probplot(values_np, dist="norm", plot=plt)
    plt.title("Q-Q Plot of number_of_frames")
    plt.tight_layout()
    plt.savefig(qq_png, dpi=300)
    plt.close()

    # -------------------------
    # Save txt report
    # -------------------------
    with open(stats_txt, "w", encoding="utf-8") as f:
        f.write("Basic Statistics for 'number_of_frames'\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Input CSV: {input_csv}\n")
        f.write(f"Total valid observations: {n}\n\n")

        f.write("Descriptive Statistics\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean               : {mean_val:.6f}\n")
        f.write(f"Median             : {median_val:.6f}\n")
        f.write(f"Standard Deviation : {std_val:.6f}\n")
        f.write(f"Variance           : {var_val:.6f}\n")
        f.write(f"Minimum            : {min_val:.6f}\n")
        f.write(f"25th Percentile    : {q1:.6f}\n")
        f.write(f"75th Percentile    : {q3:.6f}\n")
        f.write(f"IQR                : {iqr:.6f}\n")
        f.write(f"Maximum            : {max_val:.6f}\n")
        f.write(f"Skewness           : {skew_val:.6f}\n")
        f.write(f"Kurtosis           : {kurtosis_val:.6f}\n\n")

        f.write("Normality Tests\n")
        f.write("-" * 30 + "\n")

        if not math.isnan(shapiro_stat):
            f.write(f"Shapiro-Wilk statistic : {shapiro_stat:.6f}\n")
            f.write(f"Shapiro-Wilk p-value   : {shapiro_p:.6g}\n")
        else:
            f.write("Shapiro-Wilk           : Not reported (n > 5000)\n")

        if not math.isnan(dagostino_stat):
            f.write(f"D'Agostino statistic   : {dagostino_stat:.6f}\n")
            f.write(f"D'Agostino p-value     : {dagostino_p:.6g}\n")
        else:
            f.write("D'Agostino-Pearson     : Not run (n < 8)\n")

        f.write("\nAnderson-Darling Test\n")
        f.write(f"Statistic              : {anderson_result.statistic:.6f}\n")
        f.write(
            "Critical Values        : "
            + ", ".join(f"{x:.6f}" for x in anderson_result.critical_values)
            + "\n"
        )
        f.write(
            "Significance Levels    : "
            + ", ".join(f"{x}%" for x in anderson_result.significance_level)
            + "\n\n"
        )

        f.write("Interpretation\n")
        f.write("-" * 30 + "\n")
        for line in interpretation_lines:
            f.write(line + "\n")

        f.write("\nSaved Outputs\n")
        f.write("-" * 30 + "\n")
        f.write(f"Histogram : {hist_png}\n")
        f.write(f"Boxplot   : {box_png}\n")
        f.write(f"Q-Q Plot  : {qq_png}\n")
        f.write(f"Stats TXT : {stats_txt}\n")

    print("Analysis complete.")
    print(f"Histogram saved to: {hist_png}")
    print(f"Boxplot saved to  : {box_png}")
    print(f"Q-Q plot saved to : {qq_png}")
    print(f"Stats saved to    : {stats_txt}")


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate frame statistics CSV
    generate_frame_stats(
        root_dir=ROOT_DIR,
        output_csv=FRAME_STATS_CSV,
    )

    # Step 2: Analyze the generated CSV
    analyze_frame_statistics(
        input_csv=FRAME_STATS_CSV,
        output_dir=OUTPUT_DIR,
    )

    print("\nAll done.")
    print(f"Frame statistics CSV : {FRAME_STATS_CSV}")
    print(f"Histogram            : {HIST_PNG}")
    print(f"Boxplot              : {BOX_PNG}")
    print(f"Q-Q plot             : {QQ_PNG}")
    print(f"Stats report         : {STATS_TXT}")


if __name__ == "__main__":
    main()