"""Configuration constants and optional-dependency flags for the ingestion pipeline."""

import os
from pathlib import Path

# ── Optional dependencies (checked at import time) ──────────────────────────────
try:
    import chromadb  # noqa: F401
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import torch  # noqa: F401
    from PIL import Image as _PILImage  # noqa: F401
    from transformers import AutoImageProcessor as _AIP, AutoModel as _AM  # noqa: F401
    RAD_DINO_AVAILABLE = True
except ImportError:
    RAD_DINO_AVAILABLE = False

# ── Default paths ───────────────────────────────────────────────────────────────
DICOM_ROOT    = Path("/data/Deep_Angiography/DICOM")
SQLITE_DB     = Path("/data/Deep_Angiography/AngioVision/dicom_staging.db")
REPORTS_CSV   = Path("/data/Deep_Angiography/Reports/Report_List_v01_01_merged_raw.csv")
LABELED_CSV   = Path("/data/Deep_Angiography/labeled_DSA_2023_10_24.csv")
CHROMADB_PATH = Path("/data/Deep_Angiography/AngioVision/chromadb")

# ── ChromaDB / model ────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "dicom_images"   # default (RAD-DINO) collection — kept for back-compat
RAD_DINO_MODEL_ID = "microsoft/rad-dino"

# ── Image-embedding model registry ───────────────────────────────────────────────
# Must stay in sync with qa_app/config.py:EMBEDDING_MODELS. Each model is ingested
# into its OWN ChromaDB collection (vectors from different models are NOT
# comparable). RAD-DINO keeps the original "dicom_images" collection so existing
# data is reused. Use run_ingest.py --embedding-model <key> to build another.
EMBEDDING_MODELS = {
    "rad-dino": {
        "label":      "RAD-DINO (radiology)",
        "hf_id":      "microsoft/rad-dino",
        "collection": "dicom_images",
    },
    "vit-base": {
        "label":      "ViT-Base (google/vit-base-patch16-224)",
        "hf_id":      "google/vit-base-patch16-224",
        "collection": "dicom_images_vit_base",
    },
}
DEFAULT_EMBEDDING_MODEL = "rad-dino"


def resolve_embedding_model(model_key):
    """Return a valid embedding-model key, falling back to the default."""
    if model_key and model_key in EMBEDDING_MODELS:
        return model_key
    return DEFAULT_EMBEDDING_MODEL

# ── Performance tuning ──────────────────────────────────────────────────────────
PARSE_WORKERS = max(1, (os.cpu_count() or 2) - 1)
SQL_FLUSH     = 1000   # DICOM rows buffered before each SQLite commit
SUBMIT_CHUNK  = 2000   # futures submitted per chunk
CHROMA_BATCH  = 32     # frames per ChromaDB add() call
