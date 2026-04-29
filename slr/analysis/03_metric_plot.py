import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =========================
# OUTPUT DIRECTORY
# =========================
out_dir = "/data/Deep_Angiography/AngioVision/slr/analysis-results"
os.makedirs(out_dir, exist_ok=True)

# =========================
# 1. LOAD DATA
# =========================
path = "/data/Deep_Angiography/AngioVision/slr/results/stage2_results.jsonl"

rows = []

def normalize_value(val):
    """
    Normalize metric values to [0, 100]
    Rules:
    - If value <= 1 → assume it's in [0,1] → multiply by 100
    - If value > 1 → assume already percentage
    - Clip to [0, 100] to avoid noise/outliers
    """
    try:
        val = float(val)
    except:
        return None

    if val <= 1:
        val = val * 100

    # Clip to valid range
    val = max(0, min(val, 100))

    return val


with open(path, "r") as f:
    for i, line in enumerate(f):
        data = json.loads(line)

        paper_id = data.get("_source_file", f"paper_{i}")
        eval_block = data.get("evaluation", {})

        metric_name = eval_block.get("primary_metric")
        metric_value = eval_block.get("primary_metric_value")

        if metric_name is None or metric_value is None:
            continue

        # Clean string values
        if isinstance(metric_value, str):
            metric_value = metric_value.replace("%", "").strip()

        metric_value = normalize_value(metric_value)

        if metric_value is None:
            continue

        rows.append([paper_id, metric_name, metric_value])

primary_df = pd.DataFrame(rows, columns=["paper_id", "metric_name", "value"])


# =========================
# 2. NORMALIZE METRIC NAMES
# =========================
def normalize_metric(name):
    name = name.lower()

    if "f1" in name or "dice" in name:
        return "Overlap-based (F1/Dice)"
    elif "iou" in name or "jaccard" in name:
        return "Overlap-based (IoU)"
    elif "auc" in name or "roc" in name:
        return "Ranking-based (AUC)"
    elif "accuracy" in name:
        return "Accuracy"
    elif "precision" in name or "recall" in name:
        return "Detection-based (Precision/Recall)"
    else:
        return "Other"

primary_df["metric_group"] = primary_df["metric_name"].apply(normalize_metric)


# =========================
# 3. REMOVE "OTHER"
# =========================
primary_df = primary_df[primary_df["metric_group"] != "Other"].copy()


# =========================
# 4. SUMMARY TABLES
# =========================
summary_table = (
    primary_df
    .groupby("metric_group")["value"]
    .agg(["count", "mean", "std", "min", "max"])
    .reset_index()
    .sort_values("count", ascending=False)
)

variance_table = (
    primary_df
    .groupby("metric_group")["value"]
    .var()
    .reset_index()
    .rename(columns={"value": "variance"})
)

summary_table.to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)
variance_table.to_csv(os.path.join(out_dir, "variance_table.csv"), index=False)


# =========================
# 5. FIGURE 1: HISTOGRAM
# =========================
plt.figure(figsize=(8,5))

for group in primary_df["metric_group"].unique():
    subset = primary_df[primary_df["metric_group"] == group]["value"]
    plt.hist(subset, alpha=0.5, bins=10, label=group)

plt.title("Distribution of Primary Metrics Across Studies (0–100 Normalized)")
plt.xlabel("Metric Value (%)")
plt.ylabel("Number of Studies")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "fig1_histogram_primary_metrics.png"), dpi=300)
plt.close()


# =========================
# 6. FIGURE 2: KDE
# =========================
plt.figure(figsize=(8,5))

sns.kdeplot(data=primary_df, x="value", hue="metric_group", fill=True)

plt.title("Density Distribution of Primary Metrics (Normalized)")
plt.xlabel("Metric Value (%)")
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "fig2_kde_primary_metrics.png"), dpi=300)
plt.close()


# =========================
# 7. FIGURE 3: BOXPLOT
# =========================
plt.figure(figsize=(9,5))

sns.boxplot(data=primary_df, x="metric_group", y="value")
plt.xticks(rotation=30)

plt.title("Primary Metric Distribution by Metric Type (0–100 Scale)")
plt.ylabel("Metric Value (%)")
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "fig3_boxplot_metrics.png"), dpi=300)
plt.close()


# =========================
# 8. Z-SCORE ANALYSIS
# =========================
primary_df["z_score"] = primary_df.groupby("metric_group")["value"].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
)

z_summary = primary_df.groupby("metric_group")["z_score"].agg(["mean", "std"])
z_summary.to_csv(os.path.join(out_dir, "zscore_summary.csv"))


# =========================
# 9. DEBUG CHECK (VERY IMPORTANT)
# =========================
print("\n=== VALUE RANGE CHECK ===")
print(primary_df["value"].describe())

print("\nMin value:", primary_df["value"].min())
print("Max value:", primary_df["value"].max())

print("\nAll figures saved to:", out_dir)