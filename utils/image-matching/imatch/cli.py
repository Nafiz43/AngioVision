"""Argument parsing and pipeline orchestration for the image-matching tool.

Behaviour and defaults are identical to the original
``utils/metadata_db/eval_kn_retrieval.py`` — same flags, same auto-named
outputs, same seed handling — so existing commands and results carry over.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import (
    DEFAULT_DICOM_ROOT, DEFAULT_K_VALUES, DEFAULT_LABELED_CSV,
    DEFAULT_SQLITE_DB, DEFAULT_WORKERS, FRAME_MODES, auto_output_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _check_required_deps() -> None:
    """Fail fast with an actionable message before any work starts."""
    missing = []
    for mod, pkg in (("pydicom", "pydicom"), ("chromadb", "chromadb"),
                     ("numpy", "numpy"), ("tqdm", "tqdm")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: missing required packages — pip install {' '.join(missing)}")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AngioVision Image Matching — K@N cross-validated DSA sequence retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--labeled-csv", default=DEFAULT_LABELED_CSV)
    parser.add_argument("--dicom-root",  default=DEFAULT_DICOM_ROOT)
    parser.add_argument("--sqlite-db",   default=DEFAULT_SQLITE_DB)
    parser.add_argument("--out-plot",    default=None, help="Bar chart PNG (auto-named if omitted)")
    parser.add_argument("--out-md",      default=None, help="Markdown results (auto-named if omitted)")
    parser.add_argument("--out-docx",    default=None, help="Word doc examples (auto-named if omitted)")
    parser.add_argument("--out-csv",     default=None, help="Machine-readable results CSV (auto-named if omitted)")
    parser.add_argument("--model",       default="rad-dino",
        help="rad-dino | vit-b16 | vit-l16 | openclip-b32 | openclip-l14 | <HF model ID>  [default: rad-dino]")
    parser.add_argument("--frame-mode",  default="fl", choices=FRAME_MODES,
        help="best=Best_Image | fl=First→Last diag window | all=all frames  [default: fl]")
    parser.add_argument("--max-frames",  type=int, default=0,
        help="Max frames per seq when --frame-mode=all  (0=all)  [default: 0]")
    parser.add_argument("--k-values",    type=int, nargs="+", default=DEFAULT_K_VALUES,
        help="K values to evaluate  [default: 1 3 5 7 9 11 13 15]")
    parser.add_argument("--split-mode",  default="cv", choices=("cv", "holdout"),
        help="cv=n-fold cross-validation | holdout=single train/test split  [default: cv]")
    parser.add_argument("--n-folds",     type=int, default=10,
        help="Number of CV folds (split-mode=cv)  [default: 10]")
    parser.add_argument("--scale-down",  type=float, default=0.80,
        help="TRAIN fraction per category (split-mode=holdout); e.g. 0.80 = 80%% train / 20%% test  [default: 0.80]")
    parser.add_argument("--embed-batch", type=int, default=16,
        help="Frames per model forward pass  [default: 16]")
    parser.add_argument("--temporal",    action="store_true",
        help="Mean+std temporal aggregation (2×D per sequence)  [default: off]")
    parser.add_argument("--min-seqs",    type=int, default=3)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--workers",     type=int, default=DEFAULT_WORKERS,
        help=f"DICOM reader threads  [default: {DEFAULT_WORKERS}]")
    parser.add_argument("--limit-cats",  type=int, default=0)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    _check_required_deps()

    # Heavy-dep modules imported only after the dependency check above.
    from .data_loading import build_dicom_index, load_labeled_csv_grouped, resolve_paths
    from .embedding import load_embedding_model
    from .evaluation import (
        aggregate_cv_results, collect_k1_retrieval_examples, run_all_k_evaluations,
    )
    from .reporting import (
        print_results_table, save_bar_chart, save_retrieval_docx, write_csv, write_markdown,
    )
    from .splits import create_cv_folds, create_holdout_split
    from .vector_store import (
        create_ephemeral_collection, ingest_fold_from_precomputed, precompute_all_embeddings,
    )

    args     = build_parser().parse_args(argv)
    k_values = sorted(set(args.k_values))

    # Split-mode → effective "fold" count + self-documenting filename tag
    if args.split_mode == "holdout":
        n_folds_eff = 1
        split_tag   = f"holdout{round(args.scale_down*100)}"
        split_desc  = f"holdout ({args.scale_down:.0%} train / {1-args.scale_down:.0%} test)"
    else:
        n_folds_eff = args.n_folds
        split_tag   = f"{args.n_folds}fold"
        split_desc  = f"{args.n_folds}-fold cross-validation"

    _auto_md, _auto_png, _auto_docx, _auto_csv = auto_output_paths(
        args.model, args.frame_mode, args.temporal, split_tag,
    )
    out_md   = Path(args.out_md   or _auto_md)
    out_plot = Path(args.out_plot or _auto_png)
    out_docx = Path(args.out_docx or _auto_docx)
    out_csv  = Path(args.out_csv  or _auto_csv)

    log.info("═" * 60)
    log.info("  AngioVision Image Matching — K@N Retrieval Evaluation")
    log.info("═" * 60)
    log.info(f"  model         : {args.model}")
    log.info(f"  frame-mode    : {args.frame_mode}" +
             (f"  (max-frames={args.max_frames})" if args.frame_mode == "all" and args.max_frames else ""))
    log.info(f"  temporal      : {'ON  (mean+std pooling)' if args.temporal else 'OFF'}")
    log.info(f"  split-mode    : {split_desc}")
    log.info(f"  K values      : {k_values}")
    log.info(f"  workers       : {args.workers}  |  embed-batch: {args.embed_batch}")
    log.info(f"  min-seqs      : {args.min_seqs}  |  seed: {args.seed}")
    log.info(f"  out-md        : {out_md}")
    log.info(f"  out-plot      : {out_plot}")
    log.info(f"  out-docx      : {out_docx}")
    log.info(f"  out-csv       : {out_csv}")
    log.info("═" * 60)

    # 1. CSV
    groups = load_labeled_csv_grouped(Path(args.labeled_csv), args.frame_mode)

    # 2. DICOM index
    dicom_index = build_dicom_index(Path(args.dicom_root), Path(args.sqlite_db))

    # 3. Resolve paths
    groups = resolve_paths(groups, dicom_index)
    if not groups:
        log.error("No sequences with resolvable DICOM paths — aborting"); sys.exit(1)

    # 4. Limit categories (smoke-test)
    if args.limit_cats > 0:
        groups = dict(list(groups.items())[:args.limit_cats])
        log.info(f"--limit-cats={args.limit_cats}: {list(groups.keys())}")

    # 5. Build splits (n-fold CV or single holdout)
    if args.split_mode == "holdout":
        log.info(f"Building holdout split (scale-down={args.scale_down:.0%}) …")
        folds = create_holdout_split(groups, args.scale_down, args.min_seqs, args.seed)
    else:
        log.info(f"Building {args.n_folds}-fold CV splits …")
        folds = create_cv_folds(groups, args.n_folds, args.min_seqs, args.seed)
    if not folds or not folds[0]:
        log.error("No categories remain after split filtering — aborting"); sys.exit(1)

    # 6. Load model
    embed_fn, model_id, emb_dim = load_embedding_model(args.model, args.device)

    # 7. Precompute ALL embeddings ONCE  (reused across all folds — 10× speedup)
    all_embs = precompute_all_embeddings(
        groups, embed_fn, args.frame_mode, args.max_frames,
        args.embed_batch, args.workers, temporal=args.temporal,
    )

    # 8. Evaluation loop (1 iteration for holdout, n_folds for CV)
    fold_results: list[dict[int, dict[str, dict]]] = []
    for fold_idx, fold_splits in enumerate(folds):
        log.info(f"═══ Fold {fold_idx+1}/{n_folds_eff} ═══")
        collection = create_ephemeral_collection()
        n_entries  = ingest_fold_from_precomputed(
            fold_splits, collection, all_embs, args.frame_mode, args.temporal)
        log.info(f"  Fold {fold_idx+1}: {n_entries:,} ChromaDB entries")
        if collection.count() == 0:
            log.warning(f"  Fold {fold_idx+1}: empty, skipping"); continue

        embedded_test_fold = [
            (seq, all_embs.get(seq["stem"]))
            for _, (_, test) in fold_splits.items() for seq in test
        ]
        fold_results.append(run_all_k_evaluations(embedded_test_fold, fold_splits, collection, k_values))

    if not fold_results:
        log.error("No folds completed — aborting"); sys.exit(1)

    # 9. Aggregate across folds
    all_results = aggregate_cv_results(fold_results, k_values, n_folds_eff)

    # 10. Report
    print_results_table(all_results, k_values, n_folds_eff, model_id, args.frame_mode)
    save_bar_chart(all_results, k_values, n_folds_eff, model_id, args.frame_mode, out_plot)
    write_markdown(all_results, k_values, n_folds_eff, args.seed, model_id,
                   args.frame_mode, emb_dim, args.temporal, out_md)
    write_csv(all_results, k_values, n_folds_eff, args.seed, model_id,
              args.frame_mode, args.temporal, out_csv)

    # 11. Visual examples docx — use last fold's collection
    last_splits = folds[-1]
    last_col    = create_ephemeral_collection()
    ingest_fold_from_precomputed(last_splits, last_col, all_embs, args.frame_mode, args.temporal)
    last_test = [(seq, all_embs.get(seq["stem"]))
                 for _, (_, test) in last_splits.items() for seq in test]
    examples  = collect_k1_retrieval_examples(last_test, last_splits, last_col, n_per_cat=5)
    save_retrieval_docx(examples, all_results[k_values[0]], out_docx,
                        model_id, args.frame_mode, args.temporal)
