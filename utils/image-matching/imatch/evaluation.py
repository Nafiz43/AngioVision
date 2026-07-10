"""K@N matching evaluation, cross-fold aggregation and K=1 example collection.

The matching rule: each test sequence's frame embeddings query the train
collection for K nearest neighbours each; retrieved neighbours vote by their
source sequence, the winning sequence's angio_run label is the prediction.
Stdlib-only (ChromaDB collection objects are passed in, never created here).
"""

import logging
from collections import Counter, defaultdict
from typing import Optional

log = logging.getLogger(__name__)


def evaluate_at_k(
    embedded_test: list[tuple[dict, Optional[list]]],
    splits: dict[str, tuple[list, list]],
    collection, k: int,
) -> dict[str, dict]:
    effective_k = min(k, collection.count())
    if effective_k < k:
        log.warning(f"  K={k} capped to {effective_k}")

    results: dict[str, dict] = {
        label: {"n_train": len(train), "n_test": len(test),
                "correct": 0, "total_evaled": 0, "skipped": 0, "predictions": []}
        for label, (train, test) in splits.items()
    }
    for seq, embs in embedded_test:
        label = seq["angio_run"]
        if label not in results:
            continue
        r = results[label]
        if embs is None:
            r["skipped"] += 1; continue

        qr = collection.query(query_embeddings=embs, n_results=effective_k, include=["metadatas"])
        seq_votes: Counter = Counter(); seq_labels: dict[str, str] = {}
        for meta_list in qr["metadatas"]:
            for m in meta_list:
                sid = m.get("sequence_id", ""); lbl = m.get("angio_run", "")
                seq_votes[sid] += 1; seq_labels[sid] = lbl
        if not seq_votes:
            r["skipped"] += 1; continue

        pred = seq_labels[seq_votes.most_common(1)[0][0]]
        r["total_evaled"] += 1; r["correct"] += int(pred == label)
        r["predictions"].append((label, pred))
    return results


def run_all_k_evaluations(
    embedded_test: list, splits: dict, collection, k_values: list[int],
) -> dict[int, dict[str, dict]]:
    all_results: dict[int, dict] = {}
    for k in k_values:
        log.info(f"─── Evaluating K={k} ───")
        all_results[k] = evaluate_at_k(embedded_test, splits, collection, k)
    return all_results


def aggregate_cv_results(
    fold_results: list[dict[int, dict[str, dict]]],
    k_values: list[int], n_folds: int,
) -> dict[int, dict[str, dict]]:
    """
    Pool per-fold results:
      correct / total_evaled — summed (pooled micro accuracy)
      fold_accs              — per-fold accuracy list for mean±std
      n_train / n_test       — averaged per fold
    """
    all_labels: set[str] = set()
    for fr in fold_results:
        for k in k_values:
            if k in fr:
                all_labels.update(fr[k].keys())

    aggregated: dict[int, dict[str, dict]] = {}
    for k in k_values:
        aggregated[k] = {}
        for label in sorted(all_labels):
            total_correct = total_evaled = total_skip = n_train_sum = n_test_sum = 0
            fold_accs: list[float] = []
            for fr in fold_results:
                if k not in fr or label not in fr[k]:
                    continue
                r = fr[k][label]
                total_correct += r["correct"]; total_evaled += r["total_evaled"]
                total_skip    += r["skipped"]; n_train_sum  += r["n_train"]
                n_test_sum    += r["n_test"]
                if r["total_evaled"] > 0:
                    fold_accs.append(r["correct"] / r["total_evaled"])
            aggregated[k][label] = {
                "correct": total_correct, "total_evaled": total_evaled,
                "skipped": total_skip,
                "n_train": round(n_train_sum / n_folds),
                "n_test":  round(n_test_sum / n_folds),
                "fold_accs": fold_accs, "predictions": [],
            }
    return aggregated


def collect_k1_retrieval_examples(
    embedded_test: list[tuple[dict, Optional[list]]],
    splits: dict[str, tuple[list, list]],
    collection, n_per_cat: int = 5,
) -> dict[str, list[dict]]:
    """
    Run a K=1 pass and collect up to n_per_cat examples per category.
    Interleaves hits and misses so both appear at the top of each docx section.
    """
    from .config import code

    train_index: dict[str, dict] = {
        seq["stem"]: seq for _, (train, _) in splits.items() for seq in train
    }
    by_cat: dict[str, list] = defaultdict(list)
    for seq, embs in embedded_test:
        if embs is not None:
            by_cat[seq["angio_run"]].append((seq, embs))

    log.info(f"Collecting K=1 retrieval examples (up to {n_per_cat}/category) …")
    examples: dict[str, list[dict]] = {}

    for label, cat_seqs in sorted(by_cat.items()):
        hits: list[dict] = []; misses: list[dict] = []
        for seq, embs in cat_seqs:
            if len(hits) >= n_per_cat and len(misses) >= n_per_cat:
                break
            qr = collection.query(query_embeddings=embs, n_results=1, include=["metadatas"])
            seq_votes: Counter = Counter(); seq_labels: dict[str, str] = {}
            for meta_list in qr["metadatas"]:
                for m in meta_list:
                    sid = m.get("sequence_id", ""); lbl = m.get("angio_run", "")
                    seq_votes[sid] += 1; seq_labels[sid] = lbl
            if not seq_votes:
                continue
            best_sid = seq_votes.most_common(1)[0][0]
            retr_label = seq_labels[best_sid]; hit = (retr_label == label)
            entry = {"test_seq": seq, "retr_stem": best_sid, "retr_label": retr_label,
                     "hit": hit, "train_seq": train_index.get(best_sid)}
            if hit and len(hits) < n_per_cat:
                hits.append(entry)
            elif not hit and len(misses) < n_per_cat:
                misses.append(entry)

        collected: list[dict] = []; hi = mi = 0
        while len(collected) < n_per_cat:
            added = False
            if hi < len(hits):
                collected.append(hits[hi]); hi += 1; added = True
            if len(collected) < n_per_cat and mi < len(misses):
                collected.append(misses[mi]); mi += 1; added = True
            if not added:
                break
        if collected:
            examples[label] = collected
            n_h = sum(1 for e in collected if e["hit"])
            log.info(f"  {code(label):8s}  {len(collected)} examples "
                     f"({n_h} hits, {len(collected)-n_h} misses)")

    return examples


# ── Shared accuracy helpers (used by reporting) ───────────────────────────────

def micro(rk: dict[str, dict]) -> tuple[int, int]:
    return (sum(r["correct"] for r in rk.values()),
            sum(r["total_evaled"] for r in rk.values()))


def macro(rk: dict[str, dict]) -> float:
    accs = [r["correct"] / r["total_evaled"] for r in rk.values() if r["total_evaled"] > 0]
    return sum(accs) / len(accs) if accs else 0.0


def fmt_cv(r: dict) -> str:
    """Format mean±std from fold_accs; fall back to pooled if unavailable."""
    accs = r.get("fold_accs", [])
    if len(accs) >= 2:
        mean = sum(accs) / len(accs)
        std  = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
        return f"{mean:.1%}±{std:.1%}"
    if len(accs) == 1:
        return f"{accs[0]:.1%}"
    te = r.get("total_evaled", 0)
    return f"{r['correct']/te:.1%}" if te > 0 else "N/A"
