"""End-to-end image ingestion: labeled sequences → RAD-DINO embeddings → ChromaDB."""

import logging
import sqlite3
import datetime
from pathlib import Path

from tqdm import tqdm

from .config import (
    CHROMA_AVAILABLE,
    RAD_DINO_AVAILABLE,
    CHROMA_BATCH,
    EMBEDDING_MODELS,
    resolve_embedding_model,
)
from .labels import load_labeled_csv
from .dicom_parser import (
    build_dicom_index,
    parse_dicom_stem_from_csv_path,
    extract_frames,
)
from .embeddings import load_embedding_model, setup_chromadb, embed_frames

log = logging.getLogger(__name__)


def ingest_images_to_chromadb(
    con: sqlite3.Connection,
    labeled_csv: Path,
    dicom_root: Path,
    chroma_path: Path,
    limit_sequences: int = 0,
    chroma_batch_size: int = CHROMA_BATCH,
    embedding_model: str = "rad-dino",
) -> None:
    """
    End-to-end image ingestion:
      1. Load and filter the labeled CSV
      2. Build a DICOM stem→path index (SQLite-backed, filesystem fallback)
      3. Load the chosen embedding model ONCE
      4. Connect to ChromaDB (no embedding function — we supply embeddings)
      5. Iterate sequences: extract frames → embed → store
      6. Record status in image_ingestion_status for safe resume

    Each embedding model writes to its own ChromaDB collection and tracks resume
    state under its own embedding_model key, so the same sequences can be ingested
    independently by several models without clobbering one another.

    Resume behaviour (scoped to this embedding model):
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

    model_key       = resolve_embedding_model(embedding_model)
    collection_name = EMBEDDING_MODELS[model_key]["collection"]
    log.info(f"Embedding model : {model_key}  →  ChromaDB collection '{collection_name}'")

    # ── 3. Load the embedding model once — stays in memory for all batches ──
    emb_model, emb_processor, emb_device = load_embedding_model(model_key)

    # ── 4. Connect to ChromaDB (per-model collection) ────────────────────────
    _, collection = setup_chromadb(chroma_path, collection_name)

    # ── 5. Determine already-completed sequences (for THIS model) ────────────
    completed: set[str] = {
        row[0]
        for row in con.execute(
            "SELECT sequence_id FROM image_ingestion_status "
            "WHERE status = 'completed' AND embedding_model = ?",
            (model_key,),
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
                    (sequence_id, embedding_model, accession_number, series_uid,
                     source_path, frames_ingested, status, error_msg, ingested_at)
                VALUES (?, ?, ?, ?, ?, 0, 'in_progress', NULL, ?)
                """,
                (seq_id, model_key, accession, series_uid, path_str, now),
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
                    "WHERE sequence_id = ? AND embedding_model = ?",
                    (seq_id, model_key),
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

                    embeddings = embed_frames(
                        batch, emb_model, emb_processor, emb_device
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
                    WHERE sequence_id = ? AND embedding_model = ?
                    """,
                    (frames_added, datetime.datetime.utcnow().isoformat(), seq_id, model_key),
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
                    WHERE sequence_id = ? AND embedding_model = ?
                    """,
                    (err_msg, datetime.datetime.utcnow().isoformat(), seq_id, model_key),
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
