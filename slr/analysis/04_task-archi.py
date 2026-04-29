import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "/data/Deep_Angiography/AngioVision/slr/results/stage2_results.jsonl"
OUT_DIR = Path("/data/Deep_Angiography/AngioVision/slr/analysis-results")

OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "task_to_architecture_clean.csv"
MD_PATH = OUT_DIR / "task_to_architecture_clean.md"
LATEX_PATH = OUT_DIR / "task_to_architecture_clean.tex"


# =========================================================
# LOAD JSONL
# =========================================================
def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# =========================================================
# LIGHT CLEANING ONLY (NO SEMANTIC CHANGES)
# =========================================================
def clean_arch_name(name):
    """
    Fix only formatting issues:
    - remove extra whitespace
    - unify weird spacing
    - remove redundant spaces around punctuation
    """
    if not isinstance(name, str):
        return ""

    name = name.strip()

    # normalize whitespace
    name = re.sub(r"\s+", " ", name)

    # fix spacing around punctuation (light touch)
    name = re.sub(r"\s*\+\s*", "+", name)
    name = re.sub(r"\s*/\s*", "/", name)
    name = re.sub(r"\s*-\s*", "-", name)

    return name


# =========================================================
# BUILD TASK → ARCH SET
# =========================================================
task_to_arch = defaultdict(set)

for rec in load_jsonl(DATA_PATH):

    task = rec.get("task", {}).get("primary_task")
    arch = rec.get("method", {}).get("architecture_name")

    if not task or not arch:
        continue

    task = task.strip()

    # IMPORTANT: split only on comma (original format)
    for a in str(arch).split(","):
        cleaned = clean_arch_name(a)

        if cleaned:
            task_to_arch[task].add(cleaned)


# =========================================================
# CONVERT TO DATAFRAME
# =========================================================
rows = []

for task, archs in task_to_arch.items():
    rows.append({
        "Task": task,
        "Architectures": ", ".join(sorted(archs))
    })

df = pd.DataFrame(rows).sort_values("Task")


# =========================================================
# SAVE CSV
# =========================================================
df.to_csv(CSV_PATH, index=False)


# =========================================================
# SAVE MARKDOWN
# =========================================================
with open(MD_PATH, "w") as f:
    f.write("# Task → Architecture Mapping (Cleaned)\n\n")
    f.write("| Task | Architectures |\n")
    f.write("|------|--------------|\n")

    for _, row in df.iterrows():
        f.write(f"| {row['Task']} | {row['Architectures']} |\n")


# =========================================================
# SAVE LATEX
# =========================================================
def latex_escape(text):
    if not isinstance(text, str):
        return ""

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


with open(LATEX_PATH, "w") as f:

    f.write(r"\begin{table*}[t]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\small" + "\n")
    f.write(r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{3.5cm}>{\raggedright\arraybackslash}X}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"\textbf{Task} & \textbf{Architectures} \\" + "\n")
    f.write(r"\midrule" + "\n")

    rows_list = list(df.iterrows())
    for i, (_, row) in enumerate(rows_list):
        task = latex_escape(row["Task"])
        arch = latex_escape(row["Architectures"])

        if i < len(rows_list) - 1:
            f.write(f"{task} & {arch} \\\\[2pt]\n\\hdashline\n")
        else:
            # no dashed line after the last row — \bottomrule handles it
            f.write(f"{task} & {arch} \\\\\n")

    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabularx}" + "\n")
    f.write(r"\caption{Task-wise architecture mapping with light formatting normalization and deduplication.}" + "\n")
    f.write(r"\label{tab:task_architecture_clean}" + "\n")
    f.write(r"\end{table*}" + "\n")


# =========================================================
# DONE
# =========================================================
print("✅ Clean task–architecture mapping generated.")
print(f"CSV  : {CSV_PATH}")
print(f"MD   : {MD_PATH}")
print(f"TeX  : {LATEX_PATH}")