"""
Merge Validation Results (444-record human-annotated set)
===========================================================

End-to-end consumer of the human validation sized by
09_expand_validation_sample.py:

  validation/full_validation_master.json -- the full 444-record candidate
    set (in-scope pilot + expansion), stratified on the pipeline's own
    INCLUDE/EXCLUDE decision (pipeline-UNCERTAIN/ERROR records are out of
    scope for this audit -- see 09_expand_validation_sample.py) and sized
    with Cochran's formula + finite population correction so precision and
    recall can both be estimated within a tight 95% CI margin.
  validation/annotator_A.csv, validation/annotator_B.csv -- the same 444
    records, blinded (no llm_decision shown), independently filled in by
    two human annotators with INCLUDE/EXCLUDE (or UNCERTAIN, if the
    annotator personally can't decide).
  validation/final_adjudication.json -- manual adjudication of any
    annotator_A/annotator_B disagreements (see format below).

This script merges these into one pool and reports:
  1. Cohen's kappa between the two independent human annotators.
  2. Precision/recall/F1 of the pipeline's llm_decision against the
     adjudicated ground truth.

Adjudication file format (validation/final_adjudication.json), needed only
for sample_ids where annotator_A and annotator_B disagree:
    {
      "finalDecisions": [
        {"sample_id": 12, "final_decision": "INCLUDE"},
        ...
      ]
    }
Run this script before adjudication is complete to get the list of
disagreements still needing a final call; it excludes them from the
computed metrics until resolved.

Usage:
    python3 10_merge_validation_results.py
"""

import csv
import json
from pathlib import Path

FULL_MASTER_PATH = Path("validation/full_validation_master.json")
ANNOTATOR_A_PATH = Path("validation/annotator_A.csv")
ANNOTATOR_B_PATH = Path("validation/annotator_B.csv")
FINAL_ADJUDICATION_PATH = Path("validation/final_adjudication.json")

OUTPUT_PATH = Path("validation/combined_validation_results.json")

VALID_DECISIONS = {"INCLUDE", "EXCLUDE", "UNCERTAIN"}


def cohens_kappa(labels_a, labels_b):
    n = len(labels_a)
    assert n == len(labels_b) and n > 0
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    categories = sorted(set(labels_a) | set(labels_b))
    pe = sum(
        (sum(1 for a in labels_a if a == c) / n) * (sum(1 for b in labels_b if b == c) / n)
        for c in categories
    )
    kappa = 1.0 if pe == 1.0 else (po - pe) / (1 - pe)
    return kappa, po, pe


def precision_recall_f1(y_true, y_pred, positive_label="INCLUDE"):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p != positive_label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def read_annotator_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["sample_id"]): row["decision"].strip().upper() for row in csv.DictReader(f)}


def load_rows():
    master_by_id = {r["sample_id"]: r for r in json.loads(FULL_MASTER_PATH.read_text())}
    dec_a = read_annotator_csv(ANNOTATOR_A_PATH)
    dec_b = read_annotator_csv(ANNOTATOR_B_PATH)

    unfilled = [sid for sid in master_by_id if not dec_a.get(sid) or not dec_b.get(sid)]
    if unfilled:
        print(f"{len(unfilled)} of {len(master_by_id)} records still have a blank decision from "
              f"annotator A and/or B -- excluding them from this run until filled in.")

    adjudication = {}
    if FINAL_ADJUDICATION_PATH.exists():
        adjudication = {
            d["sample_id"]: d["final_decision"]
            for d in json.loads(FINAL_ADJUDICATION_PATH.read_text())["finalDecisions"]
        }

    rows = []
    disagreements_needing_adjudication = []
    for sid, rec in master_by_id.items():
        a, b = dec_a.get(sid), dec_b.get(sid)
        if not a or not b:
            continue
        if a not in VALID_DECISIONS or b not in VALID_DECISIONS:
            raise ValueError(f"sample_id {sid}: decision must be one of {VALID_DECISIONS}, got a={a!r} b={b!r}")
        if a == b:
            final = a
        elif sid in adjudication:
            final = adjudication[sid]
        else:
            disagreements_needing_adjudication.append(sid)
            continue
        rows.append({
            "sample_id": sid,
            "llm_decision": rec["llm_decision"],
            "reviewer_a": a,
            "reviewer_b": b,
            "final_decision": final,
        })

    if disagreements_needing_adjudication:
        print(f"{len(disagreements_needing_adjudication)} records have an annotator A/B disagreement "
              f"and no entry yet in {FINAL_ADJUDICATION_PATH}. Add a final_decision for these "
              f"sample_ids to include them:")
        print(sorted(disagreements_needing_adjudication))

    return rows, len(master_by_id)


def main():
    rows, n_total_target = load_rows()

    if not rows:
        print("\nNo fully-decided records yet -- nothing to compute.")
        return

    labels_a = [r["reviewer_a"] for r in rows]
    labels_b = [r["reviewer_b"] for r in rows]
    kappa, po, pe = cohens_kappa(labels_a, labels_b)
    agree_n = sum(1 for r in rows if r["reviewer_a"] == r["reviewer_b"])

    y_true = [r["final_decision"] for r in rows]
    y_pred = ["INCLUDE" if r["llm_decision"] == "INCLUDE" else "EXCLUDE" for r in rows]
    metrics = precision_recall_f1(y_true, y_pred, positive_label="INCLUDE")

    results = {
        "n_target": n_total_target,
        "n_resolved": len(rows),
        "reviewer_agreement": {
            "agreed": agree_n,
            "disagreed": len(rows) - agree_n,
            "observed_agreement_po": round(po, 4),
            "expected_agreement_pe": round(pe, 4),
            "cohens_kappa": round(kappa, 4),
        },
        "pipeline_vs_adjudicated_ground_truth": {
            "positive_label": "INCLUDE",
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote combined results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
