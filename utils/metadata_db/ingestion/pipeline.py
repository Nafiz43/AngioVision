"""Top-level orchestration for the DICOM → SQLite + images → ChromaDB pipeline."""

import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from . import config
from .store import open_db, flush_batch, db_summary
from .reports import ingest_reports
from .images import ingest_images_to_chromadb
from .dicom_parser import (
    iter_dicom_files,
    chunked,
    parse_dicom_file,
    _init_worker,
)

log = logging.getLogger(__name__)


def run_ingestion(
    *,
    root: Path,
    db: Path,
    reports: Path,
    labeled_csv: Path,
    chromadb_path: Path,
    workers: int = config.PARSE_WORKERS,
    flush: int = config.SQL_FLUSH,
    chunk: int = config.SUBMIT_CHUNK,
    chroma_batch: int = config.CHROMA_BATCH,
    limit: int = 0,
    limit_sequences: int = 0,
    dry_run: bool = False,
    summary_only: bool = False,
    reports_only: bool = False,
    images_only: bool = False,
    skip_reports: bool = False,
    skip_images: bool = False,
    embedding_model: str = config.DEFAULT_EMBEDDING_MODEL,
) -> None:
    """Run any combination of metadata, report, and image ingestion stages."""
    dicom_root  = Path(root)
    db_path     = Path(db)
    reports_csv = Path(reports)
    labeled     = Path(labeled_csv)
    chroma_path = Path(chromadb_path)

    log.info(f"DICOM root     : {dicom_root}")
    log.info(f"SQLite DB      : {db_path}")
    log.info(f"Reports CSV    : {reports_csv}")
    log.info(f"Labeled CSV    : {labeled}")
    log.info(f"ChromaDB path  : {chroma_path}")
    log.info(f"Workers        : {workers}")
    log.info(f"Flush size     : {flush}")
    log.info(f"Chunk size     : {chunk}")
    log.info(f"Chroma batch   : {chroma_batch}")
    log.info(f"Limit (meta)   : {limit or 'none'}")
    log.info(f"Limit (seqs)   : {limit_sequences or 'none'}")
    log.info(f"Embed model    : {embedding_model}")
    log.info(f"Dry run        : {dry_run}")

    if summary_only:
        db_summary(open_db(db_path))
        return

    con = open_db(db_path) if not dry_run else None

    run_metadata = not images_only and not reports_only
    run_reports  = not images_only and not skip_reports
    run_images   = not skip_images

    if run_reports and con is not None:
        ingest_reports(con, reports_csv)

    if reports_only:
        if con:
            db_summary(con)
            con.close()
        return

    # ── DICOM metadata ───────────────────────────────────────────────────────
    if run_metadata:
        if not dicom_root.exists():
            log.error(f"DICOM root does not exist: {dicom_root}")
            raise SystemExit(1)

        existing_uids: frozenset = frozenset()
        if con is not None:
            log.info("Loading existing SOPInstanceUIDs from SQLite …")
            existing_uids = frozenset(
                row[0]
                for row in con.execute(
                    "SELECT sop_instance_uid FROM dicom_files "
                    "WHERE sop_instance_uid IS NOT NULL"
                ).fetchall()
            )
            log.info(
                f"  {len(existing_uids):,} UIDs loaded — "
                "matching files will skip full parse"
            )

        log.info("Collecting .dcm file paths …")
        all_paths = list(iter_dicom_files(dicom_root))
        if limit:
            all_paths = all_paths[:limit]
        total = len(all_paths)
        log.info(f"Found {total:,} .dcm files")

        if total == 0:
            log.warning("No .dcm files found — skipping metadata ingestion.")
        else:
            inserted  = 0
            duplicate = 0
            skipped   = 0
            errored   = 0
            buffer: list[dict] = []

            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(existing_uids,),
            ) as pool:
                with tqdm(
                    total=total, unit="file",
                    desc="Ingesting DICOM metadata",
                    dynamic_ncols=True,
                ) as pbar:

                    for path_chunk in chunked(all_paths, chunk):
                        futures = {
                            pool.submit(parse_dicom_file, p): p
                            for p in path_chunk
                        }

                        for fut in as_completed(futures):
                            try:
                                result = fut.result()
                            except Exception as exc:
                                log.warning(f"Worker crashed: {exc}")
                                skipped += 1
                                pbar.update(1)
                                continue

                            if result is None:
                                skipped += 1
                            else:
                                if result.get("parse_error"):
                                    errored += 1
                                if not dry_run:
                                    buffer.append(result)

                            if con and len(buffer) >= flush:
                                new, ign = flush_batch(con, buffer)
                                inserted  += new
                                duplicate += ign
                                buffer = []

                            pbar.update(1)
                            pbar.set_postfix(
                                ins=inserted, dup=duplicate,
                                skip=skipped, err=errored,
                            )

            if con and buffer:
                new, ign = flush_batch(con, buffer)
                inserted  += new
                duplicate += ign

            log.info("─" * 60)
            log.info(f"Files found         : {total:>10,}")
            log.info(f"Rows inserted       : {inserted:>10,}")
            log.info(f"Duplicates skipped  : {duplicate:>10,}  (INSERT OR IGNORE)")
            log.info(f"No-UID / pre-exist  : {skipped:>10,}  (not DICOM, no UID, or already in DB)")
            log.info(f"Parse errors stored : {errored:>10,}  (stub rows with parse_error set)")
            if dry_run:
                log.info("(dry run — nothing written to SQLite)")

    # ── Image ingestion into ChromaDB ────────────────────────────────────────
    if run_images and con is not None:
        ingest_images_to_chromadb(
            con               = con,
            labeled_csv       = labeled,
            dicom_root        = dicom_root,
            chroma_path       = chroma_path,
            limit_sequences   = limit_sequences,
            chroma_batch_size = chroma_batch,
            embedding_model   = embedding_model,
        )

    if con:
        db_summary(con)
        con.close()
