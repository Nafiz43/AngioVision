"""End-to-end image ingestion: labeled sequences → RAD-DINO embeddings → ChromaDB."""

import logging
import sqlite3
import datetime
from pathlib import Path

from tqdm import tqdm

from .config import CHROMA_AVAILABLE, RAD_DINO_AVAILABLE, CHROMA_BATCH
from .labels import load_labeled_csv
from .dicom_parser import (
    build_dicom_index,
    parse_dicom_stem_from_csv_path,
    extract_frames,
)
from .embeddings import load_rad_dino_model, setup_chromadb, embed_frames_rad_dino

log = logging.getLogger(__name__)


def ingest_images_to_chromadb(
    con: sqlite3.Connection,
    labeled_csv: Path,
    dicom_root: Path,
    chroma_path: Path,
    limit_sequences: int = 0,
    chroma_batch_size: int = CHROMA_BATCH,
) -> None:
    """
    End-to-end image ingestion:
      1. Load and filter the labeled CSV
      2. Build a DICOM stem→path index (SQLite-backed, filesystem fallback)
      3. Load microsoft/rad-dino ONCE
      4. Connect to ChromaDB (no embedding function — we supply embeddings)
      5. Iterate sequences: extract frames → embed → store
      6. Record status in image_ingestion_status for safe resume

    Resume behaviour:
      • Sequences with status='completed' in SQLite are skipped.
      • Sequences with status='error' or 'in_progress' are retried;
        any previously written ChromaDB frames are deleted first.
    """
    if not CHROMA_AVAILABLE:
        log.error("chromadb not installed. Run: pip install chromadb")
        return
    if not RAD_DINO_AVAILABLE:
        log.error("transformers or torch not installed. Run: pip install transformers torch pillow")
        return

    # ── 1. Load CSV ──────────────────────────────────────────────────────────
    sequences = load_labeled_csv(labeled_csv)
    if not sequences:
        log.warning("No sequences to ingest after filtering.")
        return

    # ── 2. Build DICOM index ─────────────────────────────────────────────────
    dicom_index = build_dicom_index(dicom_root, con=con)

    # ── 3. Load RAD-DINO once — model stays in memory for all batches ───────
    rad_dino_model, rad_dino_processor, rad_dino_device = load_rad_dino_model()

    # ── 4. Connect to ChromaDB ───────────────────────────────────────────────
    _, collection = setup_chromadb(chroma_path)

    # ── 5. Determine already-completed sequences ─────────────────────────────
    completed: set[str] = {
        row[0]
        for row in con.execute(
            "SELECT sequence_id FROM image_ingestion_status WHERE status = 'completed'"
        ).fetchall()
    }
    log.info(f"Already completed sequences (will be skipped): {len(completed):,}")

    # ── 6. Build work list ───────────────────────────────────────────────────
    todo: list[dict] = []
    not_found = 0

    for seq in sequences:
        stem = parse_dicom_stem_from_csv_path(seq["file_path"])
        if not stem:
            continue

        actual_path = dicom_index.get(stem)
        if actual_path is None:
            log.debug(f"Not found on disk: {stem}")
            not_found += 1
            continue

        seq_id             = stem
        seq["dicom_path"]  = actual_path
        seq["stem"]        = stem
        seq["sequence_id"] = seq_id

        if seq_id in completed:
            continue

        todo.append(seq)

    log.info(f"Files not found on disk : {not_found:,}")
    log.info(f"Sequences queued        : {len(todo):,}  (after resume filter)")

    if limit_sequences > 0:
        todo = todo[:limit_sequences]
        log.info(f"Capped to {limit_sequences:,} sequences via --limit-sequences")

    if not todo:
        log.info("Nothing to ingest — all sequences are already completed.")
        return

    # ── 7. Ingest loop ───────────────────────────────────────────────────────
    succeeded         = 0
    failed            = 0
    total_frames_done = 0

    with tqdm(total=len(todo), unit="seq", desc="Ingesting images → ChromaDB") as pbar:
        for seq in todo:
            seq_id     = seq["sequence_id"]
            path_str   = seq["dicom_path"]
            accession  = seq["accession"]
            series_uid = seq["series_uid"]
            now        = datetime.datetime.utcnow().isoformat()

            con.execute(
                """
                INSERT OR REPLACE INTO image_ingestion_status
                    (sequence_id, accession_number, series_uid, source_path,
                     frames_ingested, status, error_msg, ingested_at)
                VALUES (?, ?, ?, ?, 0, 'in_progress', NULL, ?)
                """,
                (seq_id, accession, series_uid, path_str, now),
            )
            con.commit()

            try:
                frames, meta = extract_frames(path_str)
                n_frames     = len(frames)

                if n_frames == 0:
                    raise ValueError("No frames extracted from DICOM file")

                # Clean up any frames from a previous partial run
                prev_row = con.execute(
                    "SELECT frames_ingested FROM image_ingestion_status "
                    "WHERE sequence_id = ?",
                    (seq_id,),
                ).fetchone()
                prev_n = int(prev_row[0]) if prev_row and prev_row[0] else 0
                if prev_n > 0:
                    stale_ids = [f"{seq_id}_f{i:06d}" for i in range(prev_n)]
                    try:
                        collection.delete(ids=stale_ids)
                    except Exception:
                        pass

                # Embed and add in batches — model already loaded, no reload
                frames_added = 0
                for batch_start in range(0, n_frames, chroma_batch_size):
                    batch = frames[batch_start: batch_start + chroma_batch_size]

                    embeddings = embed_frames_rad_dino(
                        batch, rad_dino_model, rad_dino_processor, rad_dino_device
                    )

                    ids = [
                        f"{seq_id}_f{(batch_start + j):06d}"
                        for j in range(len(batch))
                    ]
                    metas = [
                        {
                            "accession_number": str(accession or meta["accession_number"]),
                            "series_uid":       str(series_uid or meta["series_uid"]),
                            "sop_uid":          str(meta["sop_uid"]),
                            "frame_index":      int(batch_start + j),
                            "total_frames":     int(n_frames),
                            "source_path":      str(path_str),
                            "angio_run":        str(seq.get("angio_run", "")),
                            "run_type":         str(seq.get("run_type", "")),
                            "modality":         str(meta.get("modality", "")),
                        }
                        for j in range(len(batch))
                    ]

                    collection.add(embeddings=embeddings, ids=ids, metadatas=metas)
                    frames_added += len(batch)

                con.execute(
                    """
                    UPDATE image_ingestion_status
                    SET frames_ingested = ?, status = 'completed', ingested_at = ?
                    WHERE sequence_id = ?
                    """,
                    (frames_added, datetime.datetime.utcnow().isoformat(), seq_id),
                )
                con.commit()

                succeeded         += 1
                total_frames_done += frames_added

            except Exception as exc:
                err_msg = str(exc)[:500]
                log.warning(f"Failed [{seq_id}]: {err_msg}")
                con.execute(
                    """
                    UPDATE image_ingestion_status
                    SET status = 'error', error_msg = ?, ingested_at = ?
                    WHERE sequence_id = ?
                    """,
                    (err_msg, datetime.datetime.utcnow().isoformat(), seq_id),
                )
                con.commit()
                failed += 1

            pbar.update(1)
            pbar.set_postfix(ok=succeeded, fail=failed, frames=total_frames_done)

    log.info("─" * 60)
    log.info("Image ingestion complete")
    log.info(f"  Sequences succeeded : {succeeded:>8,}")
    log.info(f"  Sequences failed    : {failed:>8,}")
    log.info(f"  Total frames stored : {total_frames_done:>8,}")
    log.info(f"  ChromaDB total      : {collection.count():>8,}  items")
    log.info("─" * 60)
