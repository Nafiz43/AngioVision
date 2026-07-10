"""Stratified train/test splitting: n-fold CV or single holdout.

Both return ``list[dict[label, (train_list, test_list)]]`` so the downstream
fold loop treats CV and holdout uniformly (holdout = one-element list).
Stdlib-only.
"""

import logging
import random

from .config import code

log = logging.getLogger(__name__)


def create_cv_folds(
    groups: dict[str, list[dict]], n_folds: int, min_seqs: int, seed: int,
) -> list[dict[str, tuple[list, list]]]:
    """
    Stratified n-fold CV: sequence j → fold (j % n_folds).
    Every sequence appears in the test set exactly once across all folds.
    Categories with fewer than max(min_seqs, n_folds) sequences are excluded.
    """
    rng = random.Random(seed)
    kept: dict[str, list[dict]] = {}
    skipped: list[tuple[str, int]] = []
    threshold = max(min_seqs, n_folds)

    for label, seqs in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(seqs) < threshold:
            skipped.append((label, len(seqs))); continue
        s = list(seqs); rng.shuffle(s); kept[label] = s
        log.info(f"  {code(label):8s}  total={len(s):4d}  "
                 f"~train={len(s)*(n_folds-1)//n_folds:4d}  ~test={len(s)//n_folds:4d}/fold")

    if skipped:
        log.info(f"  CV skipped (< {threshold} seqs): "
                 + ", ".join(f"{code(l)}({n})" for l, n in skipped))
    log.info(f"CV: {n_folds} folds × {len(kept)} categories "
             f"({sum(len(s) for s in kept.values()):,} sequences)")

    folds: list[dict[str, tuple[list, list]]] = []
    for fold_idx in range(n_folds):
        fold_splits: dict[str, tuple[list, list]] = {}
        for label, seqs in kept.items():
            test  = [seqs[j] for j in range(len(seqs)) if     j % n_folds == fold_idx]
            train = [seqs[j] for j in range(len(seqs)) if not j % n_folds == fold_idx]
            if test and train:
                fold_splits[label] = (train, test)
        folds.append(fold_splits)
    return folds


def create_holdout_split(
    groups: dict[str, list[dict]], scale_down: float, min_seqs: int, seed: int,
) -> list[dict[str, tuple[list, list]]]:
    """
    Single stratified holdout split (replicates the legacy eval_k1*.py behaviour).
    scale_down is the TRAIN fraction per category (e.g. 0.80 → 80% train / 20% test).
    Categories with fewer than min_seqs sequences are excluded.
    """
    rng = random.Random(seed)
    split: dict[str, tuple[list, list]] = {}
    skipped: list[tuple[str, int]] = []

    log.info(f"Holdout split (scale_down={scale_down:.0%} train) …")
    for label, seqs in sorted(groups.items(), key=lambda x: -len(x[1])):
        n = len(seqs)
        if n < min_seqs:
            skipped.append((label, n)); continue
        shuffled = list(seqs); rng.shuffle(shuffled)
        n_train = max(1, round(n * scale_down))
        n_test  = n - n_train
        if n_test == 0:
            n_train -= 1; n_test = 1   # guarantee ≥1 test sequence
        if n_train < 1:
            continue
        split[label] = (shuffled[:n_train], shuffled[n_train:])
        log.info(f"  {code(label):8s}  total={n:4d}  train={n_train:4d}  test={n_test:4d}")

    if skipped:
        log.info(f"  Skipped (< {min_seqs} seqs): "
                 + ", ".join(f"{code(l)}({n})" for l, n in skipped))
    log.info(f"Holdout: 1 split × {len(split)} categories "
             f"({sum(len(tr)+len(te) for tr, te in split.values()):,} sequences)")
    return [split]
