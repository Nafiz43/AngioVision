"""Default configuration constants for the AngioVision query web server."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DB        = Path("/data/Deep_Angiography/AngioVision/dicom_staging.db")
DEFAULT_CHROMADB  = Path("/data/Deep_Angiography/AngioVision/chromadb")
CHROMA_COLLECTION = "dicom_images"   # default (RAD-DINO) collection — kept for back-compat
DEFAULT_MODEL     = "qwen3.6:35b"
DEFAULT_PORT      = 5050

# ── Image-embedding model registry ───────────────────────────────────────────
# Each model embeds queries into its OWN ChromaDB collection: vectors produced by
# different models are NOT comparable, so a query MUST be embedded with the same
# model used to ingest the collection it searches. RAD-DINO keeps the original
# "dicom_images" collection so existing data is reused untouched; every other
# model needs its collection ingested via run_ingest.py before it returns results.
# Add new models here with their EXACT HuggingFace identifier.
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


def resolve_embedding_model(model_key: str | None) -> str:
    """Return a valid embedding-model key, falling back to the default."""
    if model_key and model_key in EMBEDDING_MODELS:
        return model_key
    return DEFAULT_EMBEDDING_MODEL

MAX_ROWS_FOR_SYNTHESIS = 200
MAX_RETRIES            = 2      # legacy, kept for reference; the agent self-corrects instead
N_SIMILAR_DEFAULT      = 5      # results (series) returned to UI
N_SIMILAR_OVERFETCH    = 40     # frames fetched from ChromaDB before grouping

DEFAULT_OLLAMA_HOST     = "http://localhost:11434"
DEFAULT_AGENT_MAX_STEPS = 10    # max sql_query tool calls the NL→SQL agent may make per question

# ── Clarification gate ────────────────────────────────────────────────────────
# When enabled, /api/query runs a quick pre-flight check that asks the user a
# focused clarifying question (with selectable options) when the request is
# genuinely ambiguous, instead of guessing. Per-request flags can override this.
CLARIFY_ENABLED = True

# ── Heavy-endpoint concurrency gate (/api/query + /api/image-query) ───────────
# Both endpoints share ONE Ollama server/GPU + the embedding model, so only a
# few may run at once; overflow requests queue FIFO (see qa_app/concurrency.py).
DEFAULT_MAX_CONCURRENCY = 1     # heavy jobs allowed to run simultaneously
DEFAULT_MAX_QUEUE       = 20    # waiting requests beyond this are rejected with "busy"
