import json
import math
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


# =========================
# CONFIG
# =========================
DATA_PATH = "/data/Deep_Angiography/AngioVision/slr/results/stage2_results.jsonl"
OUT_DIR = "/data/Deep_Angiography/AngioVision/slr/analysis-results"

OUT_PNG = Path(OUT_DIR) / "forest_plot_reporting_quality.png"
OUT_PDF = Path(OUT_DIR) / "forest_plot_reporting_quality.pdf"
OUT_MD  = Path(OUT_DIR) / "reporting_quality_summary.md"


# =========================
# TARGET QUALITY INDICATORS
# =========================
QUALITY_FIELDS = {
    "Clinical expert involved": "clinical_validation.clinical_expert_involved",
    "Reader study performed": "clinical_validation.reader_study_performed",
    "Patient outcome measured": "clinical_validation.patient_outcome_measured",
    "Regulatory approval mentioned": "clinical_validation.regulatory_approval_mentioned",
    "Open-source code available": "reproducibility.open_source_code",
    "Open-source data available": "reproducibility.open_source_data",
    "Ablation study performed": "evaluation.ablation_study_performed",
    "Failure case analysis": "evaluation.failure_case_analysis",
    "Statistical significance reported": "evaluation.statistical_significance_reported",
    "Inter-rater agreement reported": "dataset.inter_rater_agreement_reported",
    "Data augmentation reported": "dataset.data_augmentation_used",
}


# =========================
# UTILITIES
# =========================
def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_bools(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(extract_bools(v, key))
    elif isinstance(obj, bool):
        out[prefix] = obj
    return out


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return 0, 0
    phat = p / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / denom
    margin = (z * math.sqrt((phat*(1-phat)/n) + (z**2/(4*n**2)))) / denom
    return center - margin, center + margin


# =========================
# LOAD DATA
# =========================
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

records = list(load_jsonl(DATA_PATH))
N = len(records)

counts = defaultdict(int)

# =========================
# COUNT OCCURRENCES
# =========================
for r in records:
    flat = extract_bools(r)

    for label, path in QUALITY_FIELDS.items():
        if path in flat and flat[path] is True:
            counts[label] += 1


# =========================
# PREP DATA
# =========================
rows = []

labels = []
means = []
lowers = []
uppers = []

for label, path in QUALITY_FIELDS.items():
    c = counts[label]
    mean = c / N if N else 0
    lo, hi = wilson_ci(c, N)

    rows.append({
        "indicator": label,
        "field_path": path,
        "count": c,
        "N": N,
        "proportion": mean,
        "ci_low": lo,
        "ci_high": hi
    })

    labels.append(label)
    means.append(mean)
    lowers.append(mean - lo)
    uppers.append(hi - mean)


# =========================
# WRITE MARKDOWN SUMMARY
# =========================
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

with open(OUT_MD, "w") as f:
    f.write("# Reporting Quality Summary\n\n")
    f.write(f"Total studies (N): **{N}**\n\n")

    f.write("| Indicator | Field Path | Count | N | Proportion | CI Low | CI High |\n")
    f.write("|-----------|------------|-------|---|------------|--------|---------|\n")

    for r in rows:
        f.write(
            f"| {r['indicator']} | {r['field_path']} | "
            f"{r['count']} | {r['N']} | "
            f"{r['proportion']:.3f} | "
            f"{r['ci_low']:.3f} | {r['ci_high']:.3f} |\n"
        )

print(f"Saved markdown summary: {OUT_MD}")


# =========================
# REVERSE ORDER FOR PLOT
# =========================
labels = labels[::-1]
means = means[::-1]
lowers = lowers[::-1]
uppers = uppers[::-1]


# =========================
# PLOT
# =========================
plt.figure(figsize=(11, 6))

y = list(range(len(labels)))

tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))[:len(labels)]
colors = [(c[0], c[1], c[2], 0.9) for c in tab20_colors]

for i in range(len(labels)):
    plt.errorbar(
        means[i],
        y[i],
        xerr=[[lowers[i]], [uppers[i]]],
        fmt='o',
        color=colors[i],
        markersize=10,
        capsize=6,
        linestyle='none'
    )

plt.yticks(y, labels)
plt.xlabel("Proportion of Studies Reporting Indicator")
plt.title("Forest Plot: Reporting Quality in Systematic Literature Review")
plt.xlim(0, 1)
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

print(f"Saved:\n- {OUT_PNG}\n- {OUT_PDF}")
