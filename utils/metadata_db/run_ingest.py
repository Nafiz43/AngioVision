#!/usr/bin/env python3
"""
DICOM → SQLite Ingestion + DICOM Images → ChromaDB  (CLI entry point)

Recursively walks a DICOM directory, parses metadata from every .dcm file in
parallel (ProcessPoolExecutor), and stores everything in a flat SQLite table
keyed on SOPInstanceUID. Also ingests a radiology report CSV and embeds labeled
DICOM sequences into ChromaDB using microsoft/rad-dino.

This is the structured replacement for the legacy single-file `ingest.py`; all
logic now lives in the `ingestion/` package.

Requirements:
    pip install pydicom tqdm chromadb transformers torch pillow numpy
"""

import logging
import argparse

from ingestion import config
from ingestion.pipeline import run_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest DICOM metadata + radiology reports into SQLite "
            "AND/OR ingest labeled DICOM sequences into ChromaDB (parallel, resumable)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (metadata + reports + images)
  python3 run_ingest.py

  # Metadata + reports only (no images)
  python3 run_ingest.py --skip-images

  # Image ingestion only (metadata already loaded)
  python3 run_ingest.py --images-only

  # Build the ViT-Base collection (separate from RAD-DINO's; needed before the
  # query UI can search with the "ViT-Base" embedding option)
  python3 run_ingest.py --images-only --embedding-model vit-base

  # Image ingestion — process at most 50 sequences then stop
  python3 run_ingest.py --images-only --limit-sequences 50

  # Dry-run the DICOM metadata walk (nothing written)
  python3 run_ingest.py --dry-run --skip-images

  # Just print DB + ChromaDB statistics
  python3 run_ingest.py --summary-only
        """,
    )

    parser.add_argument("--root",            default=str(config.DICOM_ROOT))
    parser.add_argument("--db",              default=str(config.SQLITE_DB))
    parser.add_argument("--reports",         default=str(config.REPORTS_CSV))
    parser.add_argument("--labeled-csv",     default=str(config.LABELED_CSV))
    parser.add_argument("--chromadb",        default=str(config.CHROMADB_PATH))
    parser.add_argument("--workers",         type=int, default=config.PARSE_WORKERS)
    parser.add_argument("--flush",           type=int, default=config.SQL_FLUSH)
    parser.add_argument("--chunk",           type=int, default=config.SUBMIT_CHUNK)
    parser.add_argument("--chroma-batch",    type=int, default=config.CHROMA_BATCH)
    parser.add_argument("--limit",           type=int, default=0)
    parser.add_argument("--limit-sequences", type=int, default=0)
    parser.add_argument(
        "--embedding-model",
        default=config.DEFAULT_EMBEDDING_MODEL,
        choices=list(config.EMBEDDING_MODELS.keys()),
        help=(
            "Embedding model used to build the image ChromaDB collection. Each "
            "model writes to its own collection. Default: %(default)s"
        ),
    )
    parser.add_argument("--dry-run",         action="store_true")
    parser.add_argument("--summary-only",    action="store_true")
    parser.add_argument("--reports-only",    action="store_true")
    parser.add_argument("--images-only",     action="store_true")
    parser.add_argument("--skip-reports",    action="store_true")
    parser.add_argument("--skip-images",     action="store_true")

    args = parser.parse_args()

    run_ingestion(
        root            = args.root,
        db              = args.db,
        reports         = args.reports,
        labeled_csv     = args.labeled_csv,
        chromadb_path   = args.chromadb,
        workers         = args.workers,
        flush           = args.flush,
        chunk           = args.chunk,
        chroma_batch    = args.chroma_batch,
        limit           = args.limit,
        limit_sequences = args.limit_sequences,
        dry_run         = args.dry_run,
        summary_only    = args.summary_only,
        reports_only    = args.reports_only,
        images_only     = args.images_only,
        skip_reports    = args.skip_reports,
        skip_images     = args.skip_images,
        embedding_model = args.embedding_model,
    )


if __name__ == "__main__":
    main()
