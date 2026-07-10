"""ChromaDB collection management, embedding precompute and fold ingestion.

Embeddings are computed exactly once for the whole dataset (10-fold CV would
otherwise embed every sequence 10×); each fold then just copies its TRAIN
subset from the cache into a fresh in-memory collection.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from tqdm import tqdm

from .config import CHROMA_COLLECTION
from .embedding import compute_temporal_embedding
from .frames import chunked, make_frame_reader

log = logging.getLogger(__name__)


def create_ephemeral_collection():
    import chromadb
    try:
        client = chromadb.EphemeralClient()
    except AttributeError:
        client = chromadb.Client()
    # Newer ChromaDB versions make EphemeralClient() a singleton — the same
    # in-memory store persists across calls in the same process.
    # Delete the collection first so each fold always starts with an empty one.
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass   # doesn't exist yet on first call — fine
    collection = client.create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    log.info("In-memory ChromaDB collection created (no disk I/O)")
    return collection


def precompute_all_embeddings(
    groups: dict[str, list[dict]], embed_fn: Callable,
    frame_mode: str, max_frames: int, embed_batch: int,
    workers: int, temporal: bool,
) -> dict[str, Optional[list]]:
    """
    Embed every sequence exactly ONCE before the fold loop.
    Returns { stem: embeddings } where embeddings is:
      temporal=False → list of N per-frame vectors
      temporal=True  → single-element list [temporal_descriptor]
      None           → unreadable
    """
    all_seqs = [seq for seqs in groups.values() for seq in seqs]
    prefetch_chunk = max(workers * 4, 32)
    reader = make_frame_reader(frame_mode, max_frames)
    mode_desc = f"mode={frame_mode}" + (" + temporal" if temporal else "")
    log.info(f"Pre-computing embeddings for {len(all_seqs):,} sequences "
             f"({mode_desc}, workers={workers}) …")

    all_embs: dict[str, Optional[list]] = {}
    skipped = 0
    with tqdm(total=len(all_seqs), unit="seq", desc="  Embedding ALL") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for chunk in chunked(all_seqs, prefetch_chunk):
                for seq, frames in pool.map(reader, chunk):
                    stem = seq["stem"]
                    if not frames:
                        all_embs[stem] = None; skipped += 1
                    else:
                        embs = embed_fn(frames, embed_batch)
                        all_embs[stem] = [compute_temporal_embedding(embs)] if temporal else embs
                    pbar.update(1)
    log.info(f"Embeddings ready — {len(all_seqs)-skipped:,} ok, {skipped} skipped")
    return all_embs


def ingest_fold_from_precomputed(
    fold_splits: dict[str, tuple[list, list]],
    collection,
    all_embs:   dict[str, Optional[list]],
    frame_mode: str,
    temporal:   bool,
) -> int:
    """Add precomputed TRAIN embeddings for this fold into ChromaDB. No GPU work."""
    train_seqs = [seq for _, (train, _) in fold_splits.items() for seq in train]
    total_entries = 0; skipped = 0
    for seq in train_seqs:
        stem = seq["stem"]; embs = all_embs.get(stem)
        if not embs:
            skipped += 1; continue
        meta_base = {"sequence_id": stem, "angio_run": seq["angio_run"],
                     "accession": seq["accession"], "run_type": seq["run_type"]}
        if temporal:
            collection.add(embeddings=embs, ids=[stem], metadatas=[{**meta_base, "n_frames": 1}])
            total_entries += 1
        else:
            dicom_start = (max(0, seq["first_diag_idx"])
                           if frame_mode == "fl" and seq["first_diag_idx"] >= 0 else 0)
            collection.add(
                embeddings=embs,
                ids=[f"{stem}_f{dicom_start+i:05d}" for i in range(len(embs))],
                metadatas=[{**meta_base, "frame_idx": dicom_start + i} for i in range(len(embs))],
            )
            total_entries += len(embs)
    return total_entries
