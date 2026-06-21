"""SQLite connection management, batch inserts, and the summary report."""

import logging
import sqlite3
from pathlib import Path

from .schema import (
    SQLITE_SCHEMA,
    ALL_COLUMNS,
    INSERT_SQL,
    MIGRATE_IMAGE_STATUS_ADD_MODEL,
)

log = logging.getLogger(__name__)


def _migrate_image_status(con: sqlite3.Connection) -> None:
    """
    Bring a pre-existing image_ingestion_status table up to the per-embedding-model
    schema. No-op when the table is already current (or freshly created).
    """
    cols = [r[1] for r in con.execute("PRAGMA table_info(image_ingestion_status)").fetchall()]
    if not cols or "embedding_model" in cols:
        return
    log.info("Migrating image_ingestion_status \u2192 per-embedding-model schema (tagging existing rows as 'rad-dino') \u2026")
    con.executescript(MIGRATE_IMAGE_STATUS_ADD_MODEL)
    con.commit()


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SQLITE_SCHEMA)
    _migrate_image_status(con)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-65536")
    con.commit()
    log.info(f"SQLite DB opened: {db_path}")
    return con


def flush_batch(con: sqlite3.Connection, batch: list[dict]) -> tuple[int, int]:
    """INSERT OR IGNORE batch. Returns (inserted, ignored)."""
    if not batch:
        return 0, 0
    before = con.execute("SELECT COUNT(*) FROM dicom_files").fetchone()[0]
    con.executemany(INSERT_SQL, [[r.get(col) for col in ALL_COLUMNS] for r in batch])
    con.commit()
    after   = con.execute("SELECT COUNT(*) FROM dicom_files").fetchone()[0]
    new     = after - before
    ignored = len(batch) - new
    return new, ignored


def db_summary(con: sqlite3.Connection) -> None:
    total       = con.execute("SELECT COUNT(*) FROM dicom_files").fetchone()[0]
    with_error  = con.execute("SELECT COUNT(*) FROM dicom_files WHERE parse_error IS NOT NULL").fetchone()[0]
    missing_acc = con.execute("SELECT COUNT(*) FROM dicom_files WHERE accession_number LIKE 'MISSING_%'").fetchone()[0]
    rpt_total   = con.execute("SELECT COUNT(*) FROM radiology_reports").fetchone()[0]
    linked      = con.execute(
        "SELECT COUNT(DISTINCT d.accession_number) "
        "FROM dicom_files d "
        "JOIN radiology_reports r USING (accession_number)"
    ).fetchone()[0]

    img_completed = con.execute(
        "SELECT COUNT(*) FROM image_ingestion_status WHERE status = 'completed'"
    ).fetchone()[0]
    img_error = con.execute(
        "SELECT COUNT(*) FROM image_ingestion_status WHERE status = 'error'"
    ).fetchone()[0]
    img_frames = con.execute(
        "SELECT COALESCE(SUM(frames_ingested), 0) FROM image_ingestion_status "
        "WHERE status = 'completed'"
    ).fetchone()[0]
    img_by_model = con.execute(
        "SELECT embedding_model, "
        "       COUNT(*) AS seqs, "
        "       COALESCE(SUM(frames_ingested), 0) AS frames "
        "FROM image_ingestion_status WHERE status = 'completed' "
        "GROUP BY embedding_model ORDER BY embedding_model"
    ).fetchall()

    log.info("─" * 60)
    log.info(f"Total DICOM rows    : {total:>10,}")
    log.info(f"Parse errors        : {with_error:>10,}")
    log.info(f"Missing accession   : {missing_acc:>10,}")
    log.info("─" * 60)
    log.info(f"Radiology reports   : {rpt_total:>10,}")
    log.info(f"Accessions linked   : {linked:>10,}  (DICOM ∩ reports)")
    log.info("─" * 60)
    log.info(f"Image seqs done     : {img_completed:>10,}")
    log.info(f"Image seqs errored  : {img_error:>10,}")
    log.info(f"Frames in ChromaDB  : {img_frames:>10,}  (completed seqs)")
    for r in img_by_model:
        log.info(f"  · {(r['embedding_model'] or '?'):14s}: {r['seqs']:>8,} seqs, {r['frames']:>10,} frames")
    log.info("─" * 60)
    log.info("Modality breakdown:")
    for r in con.execute(
        "SELECT modality, COUNT(*) AS n FROM dicom_files "
        "WHERE parse_error IS NULL GROUP BY modality ORDER BY n DESC LIMIT 15"
    ).fetchall():
        log.info(f"  {(r['modality'] or 'NULL'):12s}  {r['n']:>10,}")
    log.info("─" * 60)
