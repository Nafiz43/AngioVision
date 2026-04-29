import json
from collections import defaultdict, Counter
from pathlib import Path


DATA_PATH = "/data/Deep_Angiography/AngioVision/slr/results/stage2_results.jsonl"
OUTPUT_MD = "/data/Deep_Angiography/AngioVision/slr/analysis-results/binary_field_statistics.md"


# -----------------------------
# Utility: Flatten nested JSON
# -----------------------------
def extract_boolean_fields(obj, prefix=""):
    """
    Recursively extract boolean fields from nested JSON.
    Returns dict: {field_path: value}
    """
    fields = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else k
            fields.update(extract_boolean_fields(v, new_key))

    elif isinstance(obj, list):
        # lists may contain dicts (e.g. metrics list), ignore deep expansion for binary stats
        pass

    else:
        if isinstance(obj, bool):
            fields[prefix] = obj

    return fields


# -----------------------------
# Load JSONL
# -----------------------------
def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# -----------------------------
# Aggregate stats
# -----------------------------
def compute_binary_stats(records):
    stats = defaultdict(lambda: Counter({"true": 0, "false": 0, "missing": 0}))
    total_docs = 0

    for rec in records:
        total_docs += 1
        bool_fields = extract_boolean_fields(rec)

        # track seen fields in this document
        seen_fields = set(bool_fields.keys())

        # update observed boolean fields
        for field, value in bool_fields.items():
            if value is True:
                stats[field]["true"] += 1
            elif value is False:
                stats[field]["false"] += 1

        # handle missing boolean fields
        all_fields_so_far = set(stats.keys())
        for field in all_fields_so_far:
            if field not in seen_fields:
                stats[field]["missing"] += 1

    return stats, total_docs


# -----------------------------
# Markdown report generator
# -----------------------------
def generate_markdown(stats, total_docs):
    lines = []
    lines.append("# Binary Field Statistics (Systematic Literature Review)\n")
    lines.append(f"Total records analyzed: **{total_docs}**\n")

    # sort fields by prevalence of True values (descending)
    sorted_fields = sorted(
        stats.items(),
        key=lambda x: x[1]["true"],
        reverse=True
    )

    for field, counts in sorted_fields:
        true_c = counts["true"]
        false_c = counts["false"]
        missing_c = max(total_docs - (true_c + false_c), 0)

        true_pct = (true_c / total_docs) * 100 if total_docs else 0
        false_pct = (false_c / total_docs) * 100 if total_docs else 0
        miss_pct = (missing_c / total_docs) * 100 if total_docs else 0

        lines.append(f"## `{field}`\n")
        lines.append("| Category | Count | Percentage |")
        lines.append("|----------|-------|------------|")
        lines.append(f"| True     | {true_c} | {true_pct:.2f}% |")
        lines.append(f"| False    | {false_c} | {false_pct:.2f}% |")
        lines.append(f"| Missing  | {missing_c} | {miss_pct:.2f}% |\n")

    return "\n".join(lines)


# -----------------------------
# Main execution
# -----------------------------
def main():
    print("Loading dataset...")
    records = list(load_jsonl(DATA_PATH))

    print("Computing binary statistics...")
    stats, total_docs = compute_binary_stats(records)

    print("Generating Markdown report...")
    md = generate_markdown(stats, total_docs)

    Path(OUTPUT_MD).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(md)

    print(f"Saved report to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()