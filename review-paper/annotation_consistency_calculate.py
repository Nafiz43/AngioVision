"""
Annotation Consistency Calculator
==================================

Validates the LLM-assisted title/abstract screening pipeline described in
Section "LLM-Assisted Screening Pipeline" of the manuscript.

Two independent LLM-based reviewer passes ("Reviewer A", "Reviewer B") were
run over a stratified 200-record sample drawn from the stage-1 screening log
(results/stage1_results.csv), applying the same inclusion/exclusion criteria
(I1-I4, E1-E5) used by the primary screening pipeline. Disagreements between
the two passes were adjudicated by a third automated pass shown both prior
decisions and rationales, producing a final adjudicated label per record.

This script computes:
  1. Cohen's kappa between Reviewer A and Reviewer B (independent-pass
     agreement, before adjudication).
  2. Precision, recall, and F1 of the original stage-1 pipeline decision
     (llm_decision in stage1_results.csv) against the adjudicated label,
     for the INCLUDE class.

Inputs:
  - validation/sample_200.json           (the 200 sampled records + original
                                           pipeline llm_decision)
  - validation/screening_validation_raw.json (Reviewer A / B / adjudicated
                                           decisions produced by the
                                           validation workflow)

Usage:
    python3 annotation_consistency_calculate.py
"""

import json
from pathlib import Path

SAMPLE_PATH = Path("validation/sample_200.json")
VALIDATION_PATH = Path("validation/screening_validation_raw.json")
OUTPUT_PATH = Path("validation/annotation_consistency_results.json")


def cohens_kappa(labels_a, labels_b):
    """Cohen's kappa for two raters over a shared set of binary labels."""
    n = len(labels_a)
    assert n == len(labels_b) and n > 0

    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    categories = sorted(set(labels_a) | set(labels_b))
    pe = 0.0
    for c in categories:
        pa = sum(1 for a in labels_a if a == c) / n
        pb = sum(1 for b in labels_b if b == c) / n
        pe += pa * pb

    if pe == 1.0:
        return 1.0, po, pe
    kappa = (po - pe) / (1 - pe)
    return kappa, po, pe


def precision_recall_f1(y_true, y_pred, positive_label="INCLUDE"):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p != positive_label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    return {
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy,
    }


def main():
    samples = json.loads(SAMPLE_PATH.read_text())
    validation = json.loads(VALIDATION_PATH.read_text())

    sample_by_id = {s["sample_id"]: s for s in samples}
    reviewer_a = {d["sample_id"]: d["decision"] for d in validation["reviewerA"]}
    reviewer_b = {d["sample_id"]: d["decision"] for d in validation["reviewerB"]}
    final_by_id = {d["sample_id"]: d["final_decision"] for d in validation["finalDecisions"]}

    common_ids = sorted(set(reviewer_a) & set(reviewer_b))
    assert len(common_ids) == 200, f"expected 200 shared decisions, got {len(common_ids)}"

    labels_a = [reviewer_a[i] for i in common_ids]
    labels_b = [reviewer_b[i] for i in common_ids]

    kappa, po, pe = cohens_kappa(labels_a, labels_b)

    agree_n = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    disagree_n = len(common_ids) - agree_n

    ground_truth_ids = sorted(final_by_id)
    y_true = [final_by_id[i] for i in ground_truth_ids]
    y_pred = [sample_by_id[i]["llm_decision"] for i in ground_truth_ids]
    # Collapse UNCERTAIN pipeline outputs to a non-INCLUDE prediction, since
    # the operational pipeline does not auto-advance UNCERTAIN records.
    y_pred_binarized = ["INCLUDE" if p == "INCLUDE" else "EXCLUDE" for p in y_pred]

    metrics = precision_recall_f1(y_true, y_pred_binarized, positive_label="INCLUDE")

    results = {
        "n_samples": len(common_ids),
        "reviewer_agreement": {
            "agreed": agree_n,
            "disagreed": disagree_n,
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
    print(f"\nWrote results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
