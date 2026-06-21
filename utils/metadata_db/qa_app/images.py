"""ChromaDB collection access, image decoding, result enrichment, and frame rendering."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import config, deps
from .deps import (
    io, base64,
    IMAGE_DEPS_OK, np, PilImage, _chromadb_mod,
    PYDICOM_OK, _pydicom_mod,
)
from .state import state
from .db import open_db

log = logging.getLogger(__name__)


def get_chroma_collection(model_key: Optional[str] = None) -> Optional[Any]:
    """
    Get or lazy-initialize the ChromaDB collection for the given embedding model.

    Each embedding model owns a separate collection (see config.EMBEDDING_MODELS)
    so query vectors are only ever compared against vectors produced by the same
    model. Collections are cached per model key in state.chroma_collections.

    Uses PersistentClient with cosine distance metric. No embedding function is
    attached here — query embeddings are computed by the matching model and passed
    explicitly as query_embeddings.

    Returns:
        ChromaDB collection object if available, else None
    """
    key = config.resolve_embedding_model(model_key)
    cached = state.chroma_collections.get(key)
    if cached is not None:
        return cached
    if not IMAGE_DEPS_OK:
        return None

    collection_name = config.EMBEDDING_MODELS[key]["collection"]
    try:
        client = _chromadb_mod.PersistentClient(path=str(state.chromadb_path))
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        state.chroma_collections[key] = collection
        log.info(
            f"ChromaDB collection '{collection_name}' (model '{key}') ready at "
            f"{state.chromadb_path} — {collection.count():,} items"
        )
        return collection
    except Exception as exc:
        log.error(f"ChromaDB init failed for model '{key}': {exc}")
        return None


def decode_image_to_uint8_rgb(b64_str: str) -> np.ndarray:
    """
    Decode base64-encoded image to uint8 RGB numpy array (H×W×3).

    Raises:
        Exception: On base64 decode or image format errors
    """
    img_bytes = base64.b64decode(b64_str)
    img = PilImage.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def enrich_results_from_sqlite(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich ChromaDB search results with SQLite metadata and report excerpts.

    Adds study_date, series_description, modality, patient demographics, and the
    first 400 characters of the radiology report for each result by accession number.
    Non-fatal: returns original results if enrichment fails.
    """
    accessions = list({r["accession_number"] for r in results if r.get("accession_number")})

    if not accessions:
        return results

    enrichment_map: Dict[str, Dict[str, Any]] = {}
    try:
        placeholders = ", ".join(["?"] * len(accessions))
        with state.lock:
            con = open_db()
            try:
                rows = con.execute(
                    f"""
                    SELECT
                        d.accession_number,
                        MAX(d.study_date)        AS study_date,
                        MAX(d.series_description) AS series_description,
                        MAX(d.modality)           AS modality,
                        MAX(d.patient_age)        AS patient_age,
                        MAX(d.patient_sex)        AS patient_sex,
                        SUBSTR(MAX(r.radrpt), 1, 400) AS radrpt_excerpt
                    FROM dicom_files d
                    LEFT JOIN radiology_reports r USING (accession_number)
                    WHERE d.accession_number IN ({placeholders})
                      AND d.parse_error IS NULL
                    GROUP BY d.accession_number
                    """,
                    list(accessions),
                ).fetchall()
            finally:
                con.close()

        for row in rows:
            acc = row["accession_number"]
            enrichment_map[acc] = {
                "study_date": row["study_date"] or "",
                "series_description": row["series_description"] or "",
                "modality": row["modality"] or "",
                "patient_age": row["patient_age"] or "",
                "patient_sex": row["patient_sex"] or "",
                "radrpt_excerpt": row["radrpt_excerpt"] or "",
            }
    except Exception as exc:
        log.warning(f"SQLite enrichment failed (non-fatal): {exc}")

    for r in results:
        acc = r.get("accession_number")
        if acc and acc in enrichment_map:
            r.update(enrichment_map[acc])

    return results


def load_dicom_frame_as_b64(
    source_path: str,
    frame_index: int,
    thumb_px: int = 320,
) -> Optional[str]:
    """
    Extract and encode a DICOM frame as a base64-encoded PNG.

    Opens the DICOM file, extracts the frame at frame_index, normalises pixel values
    to 0–255 uint8 RGB, optionally inverts for MONOCHROME1, resizes to max dimension
    thumb_px, and encodes as PNG. Returns None on any error.
    """
    if not IMAGE_DEPS_OK or not PYDICOM_OK:
        return None
    try:
        ds          = _pydicom_mod.dcmread(str(source_path), force=False)
        photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        pixels      = ds.pixel_array

        # ── Normalise array shape → extract target frame ──────────────
        if pixels.ndim == 2:
            frame = pixels  # single grayscale
        elif pixels.ndim == 3:
            if pixels.shape[2] in (3, 4):
                frame = pixels  # single RGB/RGBA (H,W,C)
            else:
                fi = min(max(0, frame_index), pixels.shape[0] - 1)
                frame = pixels[fi]  # multi-frame grayscale (N,H,W)
        elif pixels.ndim == 4:
            fi = min(max(0, frame_index), pixels.shape[0] - 1)
            frame = pixels[fi]  # multi-frame colour (N,H,W,C)
        else:
            return None

        # ── Convert to uint8 RGB ──────────────────────────────────────
        if frame.ndim == 2:
            f = frame.astype(np.float32)
            lo, hi = f.min(), f.max()
            f = (f - lo) / (hi - lo + 1e-8) * 255.0
            f = f.astype(np.uint8)
            if "MONOCHROME1" in photometric:
                f = 255 - f  # invert: high value = dark
            rgb = np.stack([f, f, f], axis=-1)
        elif frame.ndim == 3:
            f = frame.astype(np.float32)
            lo, hi = f.min(), f.max()
            f = (f - lo) / (hi - lo + 1e-8) * 255.0
            f = f.astype(np.uint8)
            rgb = f[:, :, :3]  # drop alpha channel if present
        else:
            return None

        # ── Resize and encode ──────────────────────────────────────────
        img = PilImage.fromarray(rgb, "RGB")
        img.thumbnail((thumb_px, thumb_px), PilImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as exc:
        log.debug(f"Frame extraction failed [{source_path}:{frame_index}]: {exc}")
        return None
