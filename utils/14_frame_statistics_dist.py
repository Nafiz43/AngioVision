#!/usr/bin/env python3

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =========================================================
# PATHS
# =========================================================
INPUT_CSV = "/data/Deep_Angiography/DICOM-metadata-stats/frame_statistics.csv"
OUTPUT_DIR = "/data/Deep_Angiography/DICOM-metadata-stats"

HIST_PNG = os.path.join(OUTPUT_DIR, "number_of_frames_histogram.png")
BOX_PNG = os.path.join(OUTPUT_DIR, "number_of_frames_boxplot.png")
QQ_PNG = os.path.join(OUTPUT_DIR, "number_of_frames_qqplot.png")
STATS_TXT = os.path.join(OUTPUT_DIR, "number_of_frames_basic_stats.txt")

# =========================================================
# LOAD DATA
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

if "number_of_frames" not in df.columns:
    raise ValueError("Column 'number_of_frames' not found in the CSV.")

# Convert safely to numeric and drop missing/non-numeric values
values = pd.to_numeric(df["number_of_frames"], errors="coerce").dropna()

if len(values) == 0:
    raise ValueError("No valid numeric values found in 'number_of_frames'.")

values_np = values.to_numpy()

# =========================================================
# BASIC STATS
# =========================================================
n = len(values_np)
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

# =========================================================
# NORMALITY TESTS
# =========================================================
# Shapiro-Wilk is recommended for smaller samples, but scipy warns for very large n
if n <= 5000:
    shapiro_stat, shapiro_p = stats.shapiro(values_np)
else:
    shapiro_stat, shapiro_p = float("nan"), float("nan")

# D'Agostino and Pearson
if n >= 8:
    dagostino_stat, dagostino_p = stats.normaltest(values_np)
else:
    dagostino_stat, dagostino_p = float("nan"), float("nan")

# Anderson-Darling
anderson_result = stats.anderson(values_np, dist="norm")

# =========================================================
# INTERPRETATION
# =========================================================
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

# =========================================================
# SAVE HISTOGRAM
# =========================================================
plt.figure(figsize=(10, 6))
plt.hist(values_np, bins=30, edgecolor="black")
plt.title("Distribution of number_of_frames")
plt.xlabel("number_of_frames")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(HIST_PNG, dpi=300)
plt.close()

# =========================================================
# SAVE BOXPLOT
# =========================================================
plt.figure(figsize=(8, 5))
plt.boxplot(values_np, vert=False)
plt.title("Boxplot of number_of_frames")
plt.xlabel("number_of_frames")
plt.tight_layout()
plt.savefig(BOX_PNG, dpi=300)
plt.close()

# =========================================================
# SAVE QQ PLOT
# =========================================================
plt.figure(figsize=(8, 8))
stats.probplot(values_np, dist="norm", plot=plt)
plt.title("Q-Q Plot of number_of_frames")
plt.tight_layout()
plt.savefig(QQ_PNG, dpi=300)
plt.close()

# =========================================================
# SAVE TXT REPORT
# =========================================================
with open(STATS_TXT, "w", encoding="utf-8") as f:
    f.write("Basic Statistics for 'number_of_frames'\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Input CSV: {INPUT_CSV}\n")
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
    f.write("Critical Values        : " + ", ".join(f"{x:.6f}" for x in anderson_result.critical_values) + "\n")
    f.write("Significance Levels    : " + ", ".join(f"{x}%" for x in anderson_result.significance_level) + "\n\n")

    f.write("Interpretation\n")
    f.write("-" * 30 + "\n")
    for line in interpretation_lines:
        f.write(line + "\n")

    f.write("\nSaved Outputs\n")
    f.write("-" * 30 + "\n")
    f.write(f"Histogram : {HIST_PNG}\n")
    f.write(f"Boxplot   : {BOX_PNG}\n")
    f.write(f"Q-Q Plot  : {QQ_PNG}\n")
    f.write(f"Stats TXT : {STATS_TXT}\n")

print("Done.")
print(f"Histogram saved to: {HIST_PNG}")
print(f"Boxplot saved to  : {BOX_PNG}")
print(f"Q-Q plot saved to : {QQ_PNG}")
print(f"Stats saved to    : {STATS_TXT}")