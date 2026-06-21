"""Default configuration constants for the AngioVision query web server."""

from pathlib import Path

DEFAULT_DB        = Path("/data/Deep_Angiography/AngioVision/dicom_staging.db")
DEFAULT_CHROMADB  = Path("/data/Deep_Angiography/AngioVision/chromadb")
CHROMA_COLLECTION = "dicom_images"
DEFAULT_MODEL     = "qwen3:1.7b"
DEFAULT_PORT      = 5050

MAX_ROWS_FOR_SYNTHESIS = 200
MAX_RETRIES            = 2      # legacy, kept for reference; the agent self-corrects instead
N_SIMILAR_DEFAULT      = 5      # results (series) returned to UI
N_SIMILAR_OVERFETCH    = 40     # frames fetched from ChromaDB before grouping

DEFAULT_OLLAMA_HOST     = "http://localhost:11434"
DEFAULT_AGENT_MAX_STEPS = 10    # max sql_query tool calls the NL→SQL agent may make per question
