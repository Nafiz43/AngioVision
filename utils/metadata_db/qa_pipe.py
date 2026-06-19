#!/usr/bin/env python3
"""
DICOM Query Web Server
Serves the browser UI and exposes a REST API that bridges:
  - Natural language → SQL, via an AGENTIC pipeline (smolagents ToolCallingAgent)
    that can call the database as many times as it needs — exploring the
    schema, checking real column values, recovering from SQL errors, and
    refining its query — before producing a final answer. This replaces the
    old single-shot "generate SQL once, retry-on-error-twice" approach.
  - SQL → SQLite execution
  - Results → synthesis (via the same agent's final answer)

NEW ── Image RAG (with RAD-DINO embeddings):
  POST /api/image-query accepts a base64-encoded image, queries ChromaDB via
  RAD-DINO embeddings for the top visually similar DICOM sequences (grouped
  by SOPInstanceUID), enriches each hit with SQLite metadata and radiology
  report excerpts, and streams a synthesised narrative answer back to the UI.

  GET /api/thumbnail returns a thumbnail for a given DICOM frame.
  GET /api/frame returns a full-resolution frame for display.

Tables served:
  dicom_files          — one row per .dcm file (DICOM metadata)
  radiology_reports    — one row per accession number (radrpt text)
  image_ingestion_status — one row per ingested sequence (for ChromaDB stats)

Usage:
    python3 dicom_query_server.py
    python3 dicom_query_server.py --db /path/to/dicom_staging.db --port 5050
    python3 dicom_query_server.py --model qwen3:14b --no-think
    python3 dicom_query_server.py --chromadb /path/to/chromadb
    python3 dicom_query_server.py --ollama-host http://localhost:11434 --agent-max-steps 10

Requires (new): pip install 'smolagents[openai]'
  The agentic NL→SQL pipeline talks to your local Ollama server through its
  OpenAI-compatible API (http://<ollama-host>/v1), so the model you pick in
  the UI dropdown needs to support tool/function calling in Ollama (qwen3,
  llama3.1+, mistral, etc. all work).
"""

import os
import re
import sys
import json
import time
import queue
import sqlite3
import logging
import argparse
import traceback
import threading
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List

try:
    from flask import Flask, request, jsonify, Response, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("ERROR: pip install flask flask-cors")
    sys.exit(1)

try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    print("ERROR: pip install langchain-ollama")
    sys.exit(1)

# ── Image / vector search dependencies (soft — server starts without them) ────
try:
    import io
    import base64
    import numpy as np
    from PIL import Image as PilImage
    import chromadb as _chromadb_mod
    IMAGE_DEPS_OK = True
except ImportError:
    IMAGE_DEPS_OK = False
    np = None
    _chromadb_mod = None

# ── RAD-DINO embedding function ───────────────────────────────────────────────
try:
    import torch
    from transformers import AutoModel, AutoImageProcessor
    RADDINO_OK = True
except ImportError:
    RADDINO_OK = False
    torch = None
    AutoModel = None
    AutoImageProcessor = None

# ── DICOM pixel reading — for frame extraction ──────────────────────────────
try:
    import pydicom as _pydicom_mod
    PYDICOM_OK = True
except ImportError:
    PYDICOM_OK = False
    _pydicom_mod = None

# ── Agentic NL→SQL pipeline (soft — server starts without it, /api/query 503s) ─
try:
    from smolagents import ToolCallingAgent, Tool, OpenAIServerModel
    import openai as _openai_mod  # required by OpenAIServerModel under the hood
    SMOLAGENTS_OK = True
except ImportError as _sma_err:
    SMOLAGENTS_OK = False
    _smolagents_err = str(_sma_err)
    ToolCallingAgent = None
    OpenAIServerModel = None

    class Tool:  # minimal fallback so SQLQueryTool can still be defined/imported
        pass

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DB        = Path("/data/Deep_Angiography/AngioVision/dicom_staging.db")
DEFAULT_CHROMADB  = Path("/data/Deep_Angiography/AngioVision/chromadb")
CHROMA_COLLECTION = "dicom_images"
DEFAULT_MODEL     = "qwen3:1.7b"
DEFAULT_PORT      = 5050
MAX_ROWS_FOR_SYNTHESIS = 200
MAX_RETRIES       = 2      # legacy, kept for reference; the agent self-corrects instead
N_SIMILAR_DEFAULT = 5      # results (series) returned to UI
N_SIMILAR_OVERFETCH = 40   # frames fetched from ChromaDB before grouping

DEFAULT_OLLAMA_HOST     = "http://localhost:11434"
DEFAULT_AGENT_MAX_STEPS = 10   # how many sql_query calls the NL→SQL agent may make per question

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────
SCHEMA_CONTEXT = """
You are an expert SQLite query generator for a DICOM medical imaging database.

## Table: dicom_files

One row per .dcm file. All values are stored as TEXT — cast when needed.

### Hierarchy columns:
  accession_number       — primary lookup key; may be 'MISSING_<uid_prefix>'
  study_instance_uid     — one study per patient encounter
  series_instance_uid    — one series = one acquisition run within a study
  sop_instance_uid       — PRIMARY KEY; one row per .dcm file

### Patient columns:
  patient_id, patient_name, patient_sex, patient_age
  pregnancy_status, patient_identity_removed, deidentification_method

### Study columns:
  study_date             — YYYYMMDD string
  study_time             — HHMMSS string
  study_description, referring_physician
  requested_procedure_description
  performed_procedure_step_description

### Series columns:
  series_date, series_time, series_description
  series_number, acquisition_number
  modality               — 'XA' = X-ray Angiography; 'RF' = Fluoroscopy; etc.
  protocol_name

### Instance / acquisition columns:
  sop_class_uid, instance_number, image_type
  acquisition_date, acquisition_time, content_date, content_time

### Image geometry:
  rows, columns          — image dimensions (cast to INTEGER)
  bits_allocated, bits_stored, high_bit, samples_per_pixel
  photometric_interpretation
  number_of_frames       — raw DICOM tag value
  frame_count            — normalised frame count (always >= 1, use this)
  frame_time             — ms per frame (cast to REAL)
  cine_rate              — frames per second (cast to INTEGER)
  images_in_acquisition, representative_frame_number
  start_trim, stop_trim, recommended_display_frame_rate

### Radiation / acquisition parameters:
  kvp                    — peak kilovoltage (cast to REAL)
  exposure_time          — ms (cast to INTEGER)
  xray_tube_current      — mA (cast to INTEGER)
  avg_pulse_width        — ms (cast to REAL)
  radiation_setting, radiation_mode
  dose_product           — µGy·m² (cast to REAL)

### Geometry / positioning:
  distance_source_to_detector   — mm (cast to REAL)
  distance_source_to_patient    — mm (cast to REAL)
  positioner_primary_angle, positioner_secondary_angle  — degrees (cast to REAL)
  patient_position, intensifier_size, imager_pixel_spacing, focal_spots

### Display / contrast / equipment:
  window_center, window_width, voi_lut_function
  contrast_bolus_agent, contrast_bolus_ingredient
  manufacturer, manufacturer_model_name, station_name
  software_versions, device_serial_number
  detector_id, detector_description

### File location:
  source_file            — filename only (e.g. 'IM-0001-0001.dcm')
  source_path            — full absolute path to the .dcm file

### Pipeline bookkeeping:
  parse_error            — NULL for good rows; non-NULL = file was unreadable
  sqlite_inserted_at     — UTC ISO8601 timestamp when row was written

---

## Table: radiology_reports

One row per unique accession number. Joinable to dicom_files on accession_number.
Contains the FULL free-text radiology report for each study in the `radrpt` column.

  accession_number   TEXT PRIMARY KEY   — links directly to dicom_files.accession_number
  radrpt             TEXT               — full free-text radiology report
  source_csv         TEXT               — path to the source CSV file
  csv_inserted_at    TEXT               — UTC ISO8601 timestamp when row was ingested

### CRITICAL: radrpt is a free-text narrative field in radiology_reports, NOT in dicom_files.
  Do NOT search dicom_files columns for report content.
  Always search r.radrpt when the user asks about anything written IN a report.

### Typical radrpt structure (interventional radiology):
  PROCEDURE, DATE, INDICATION, PHYSICIANS, FLUOROSCOPY TIME, FLUOROSCOPY DOSE,
  CONTRAST, TECHNIQUE/FINDINGS, COMPLICATION, IMPRESSION, PLAN,
  SIGNED BY / Electronically Signed By

### Key text-search mappings:
  "signed by <name>"         → LOWER(r.radrpt) LIKE '%signed by%<lowercase name>%'
  "electronically signed by" → LOWER(r.radrpt) LIKE '%electronically signed by%<name>%'
  "fluoroscopy time"         → extract after 'FLUOROSCOPY TIME:' label
  "contrast volume"          → LOWER(r.radrpt) LIKE '%contrast%'
  "indication / diagnosis"   → LOWER(r.radrpt) LIKE '%indication%<term>%'
  "complication"             → LOWER(r.radrpt) LIKE '%complication%'
  "impression"               → LOWER(r.radrpt) LIKE '%impression%<term>%'

### Join example:
  SELECT d.accession_number, d.series_description, d.frame_count,
         d.modality, d.study_date, r.radrpt
  FROM   dicom_files d
  JOIN   radiology_reports r USING (accession_number)
  WHERE  d.parse_error IS NULL
  LIMIT  5;

---

## SQL Rules

1. Always add `WHERE parse_error IS NULL` (on dicom_files) unless asking about errors.
2. All numeric columns in dicom_files are TEXT — cast explicitly:
     CAST(frame_count AS INTEGER), CAST(kvp AS REAL), CAST(dose_product AS REAL)
3. Text search on radrpt: always use LOWER() + LIKE with % wildcards.
4. NEVER search dicom_files columns for physician names that appear in report text.
5. Date range on study_date (YYYYMMDD): WHERE d.study_date BETWEEN '20090101' AND '20121231'
6. Use DISTINCT when counting or listing unique accession numbers or studies.
7. Default LIMIT 25 unless user asks for all or it's an aggregation.
8. DSA series: LOWER(d.series_description) LIKE '%dsa%'
9. accession_number LIKE 'MISSING_%' means DICOM had no AccessionNumber tag.
10. When joining radiology_reports use JOIN ... USING (accession_number).
12. Output ONLY the SQL query — no explanation, no markdown fences, no preamble.
"""

SYNTHESIS_SYSTEM = """
You are a clinical informatics assistant helping radiologists and researchers
query a DICOM imaging database.

You receive: the question, the SQL executed, and the raw results (JSON).

Produce a clear, concise, human-readable answer in PLAIN TEXT only.
- Do NOT use markdown tables (no | pipes, no --- separators)
- Do NOT use markdown headers (no ## or **bold**)
- Use plain dashes (-) for lists when listing items
- Summarise patterns when many results; enumerate each item when ≤ 10 rows
- Highlight clinically or technically interesting findings
- If results reference radiology report text (radrpt), summarise key findings
- If results are empty, say so and suggest why
- Do NOT re-state the SQL; be direct and informative
- If the results include source_path columns, tell the user that thumbnail
  previews are visible in the Table tab (the UI renders them automatically)
"""

IMAGE_SYNTHESIS_SYSTEM = """
You are a clinical informatics assistant specialising in DICOM angiography imaging.

You receive: the user's question, and the top visually similar cases retrieved from a
ChromaDB vector database using RAD-DINO image embedding similarity.

Each retrieved case includes: rank, similarity score (%), accession number, study date,
modality, frame index within the sequence, series description, patient demographics if
available, and optionally a radiology report excerpt.

Your task:
- Summarise the retrieved similar cases clearly and concisely
- Highlight any shared clinical or technical characteristics visible in the metadata
- If radiology report excerpts are present, note any relevant clinical findings
- Be factual; do not speculate about pathology from image embeddings alone
- Use plain text only — no markdown tables, no bold headers, no pipe characters
- List cases with plain dashes (-) and keep each entry brief
- If similarity scores are low (<50%), note that the visual match may be approximate
"""

ERROR_REPAIR_SYSTEM = """
You are an expert SQLite query debugger for a DICOM database.
You receive: the original question, the failed SQL, the SQLite error, and schema context.
Produce a corrected SQL query that fixes the error.
Output ONLY the corrected SQL — no explanation, no markdown fences.

Key rules:
- All columns in dicom_files are TEXT — always CAST numeric values before comparison
- Always include parse_error IS NULL on dicom_files unless asking about errors
- radiology_reports joins to dicom_files on accession_number
- radrpt is a TEXT column in radiology_reports, NOT in dicom_files
- For "signed by <name>" queries: WHERE LOWER(r.radrpt) LIKE '%signed by%<name>%'
"""
# NOTE: ERROR_REPAIR_SYSTEM / MAX_RETRIES above are kept for reference but are no
# longer used by /api/query — the smolagents agent now sees SQL errors directly
# as tool output and corrects itself across multiple turns instead of a single
# blind "repair" pass.

# ── Global state ──────────────────────────────────────────────────────────────
app: Flask = Flask(__name__, static_folder=None)
_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:*")
CORS(app, origins=_cors_origins.split(","))

_db_path: Path = DEFAULT_DB
_chromadb_path: Path = DEFAULT_CHROMADB
_ollama: Optional[ChatOllama] = None
_think: bool = True
_db_stats_cache: Optional[Dict[str, Any]] = None
_chroma_collection: Optional[Any] = None   # lazy-initialised ChromaDB collection
_raddino_model: Optional[object] = None  # lazy-initialised RAD-DINO embedding model
_lock: threading.Lock = threading.Lock()
_ollama_host: str = DEFAULT_OLLAMA_HOST  # base URL for Ollama OpenAI-compatible API
_agent_max_steps: int = DEFAULT_AGENT_MAX_STEPS  # max tool calls per question

# ═══════════════════════════════════════════════════════════════════════════════
# RAD-DINO Embedding Function
# ═══════════════════════════════════════════════════════════════════════════════

class RADDINOEmbeddingFunction:
    """
    RAD-DINO embedding function for medical imaging DICOM frames.
    
    Uses microsoft/rad-dino (ViT-B/16 trained on ~1M radiology images) via HuggingFace
    transformers. MUST match the embedding model used during ChromaDB ingestion so that
    query embeddings live in the same vector space as indexed embeddings.
    
    The CLS token (index 0 of last_hidden_state) is extracted and L2-normalised as
    the 768-dimensional embedding vector.
    """

    # HuggingFace model identifier — MUST match ingestion pipeline
    MODEL_ID = "microsoft/rad-dino"

    @classmethod
    def name(cls) -> str:
        """Return the name of this embedding function."""
        return "raddino"

    def __init__(self, model_id: str = MODEL_ID) -> None:
        """
        Initialize RAD-DINO model for embedding computation.
        
        Args:
            model_id: HuggingFace model identifier (default: microsoft/rad-dino)
        
        Raises:
            ImportError: If torch or transformers are not installed
        """
        if not RADDINO_OK:
            raise ImportError(
                "torch and transformers required: "
                "pip install torch transformers"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        log.info(f"RAD-DINO ({model_id}) loaded on {self.device}")

    def __call__(self, input_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Embed a batch of images using RAD-DINO.
        
        Args:
            input_list: List of uint8 RGB numpy arrays (H×W×3)
        
        Returns:
            List of L2-normalised 768-dimensional embedding vectors (float32)
        """
        embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for img_array in input_list:
                # Ensure RGB uint8
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                img = PilImage.fromarray(img_array.astype("uint8"), "RGB")

                # AutoImageProcessor handles resize + normalisation
                inputs = self.processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                # CLS token from last hidden state → 768-dim
                cls_emb = outputs.last_hidden_state[:, 0, :]   # (1, 768)

                # L2 normalise
                cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=-1)
                embeddings.append(cls_emb.cpu().numpy().flatten())

        return embeddings


def get_raddino_model() -> Optional[RADDINOEmbeddingFunction]:
    """
    Get or lazy-initialize the RAD-DINO embedding model (singleton pattern).
    
    Returns:
        RADDINOEmbeddingFunction instance if dependencies available, else None
    """
    global _raddino_model
    if _raddino_model is None:
        if not RADDINO_OK:
            log.warning("RAD-DINO dependencies not available")
            return None
        try:
            _raddino_model = RADDINOEmbeddingFunction()
        except Exception as exc:
            log.error(f"Failed to load RAD-DINO model: {exc}")
            return None
    return _raddino_model


# ═══════════════════════════════════════════════════════════════════════════════
# LLM helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_ollama() -> ChatOllama:
    """Get or initialize the Ollama ChatOllama instance (singleton pattern)."""
    global _ollama
    if _ollama is None:
        _ollama = ChatOllama(model=DEFAULT_MODEL)
    return _ollama


def set_model(model: str) -> None:
    """Update the global Ollama model instance to use a different model."""
    global _ollama
    _ollama = ChatOllama(model=model)
    log.info(f"Model set to: {model}")


def llm_call(messages: List[Dict[str, str]], think: bool = True) -> str:
    """
    Call the Ollama model with the given messages.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        think: Whether to prepend '/no_think' directive to user messages (for qwen3)
    
    Returns:
        The model's text response with any <think> tags stripped
    """
    ollama = get_ollama()
    lc_messages: List[Any] = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            if not think:
                content = "/no_think\n" + content
            lc_messages.append(HumanMessage(content=content))
    response = ollama.invoke(lc_messages)
    text = response.content
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def clean_sql(raw_sql: str) -> str:
    """
    Remove markdown code fence markers from SQL text.
    
    Args:
        raw_sql: Raw SQL text possibly wrapped in markdown code fences
    
    Returns:
        Clean SQL statement
    """
    sql = re.sub(r"```(?:sql|sqlite)?\s*", "", raw_sql, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql)
    return sql.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite helpers
# ═══════════════════════════════════════════════════════════════════════════════

def open_db() -> sqlite3.Connection:
    """
    Open a read-write connection to the DICOM SQLite database.
    
    Returns:
        A sqlite3 Connection with Row factory enabled
    """
    con = sqlite3.connect(str(_db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def run_sql_query(sql: str) -> List[Dict[str, Any]]:
    """
    Execute a read-only SQL query against the database and return results.
    
    Args:
        sql: SQL SELECT statement
    
    Returns:
        List of result rows as dictionaries
    
    Raises:
        sqlite3.Error: On SQL syntax or execution errors
    """
    with _lock:
        con = open_db()
        try:
            con.execute("PRAGMA query_only = ON")
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            con.close()


def get_db_stats() -> Dict[str, Any]:
    """
    Compute and cache database statistics (instance, patient, study, series counts).
    
    Returns cached stats if available; otherwise queries the database and caches result.
    
    Returns:
        Dictionary with keys: instances, patients, studies, series, errors,
        modalities, db_path, rpt_total, rpt_linked, rpt_unlinked
    """
    global _db_stats_cache
    if _db_stats_cache:
        return _db_stats_cache
    with _lock:
        con = open_db()
        try:
            total = con.execute(
                "SELECT COUNT(*) FROM dicom_files WHERE parse_error IS NULL"
            ).fetchone()[0]
            patients = con.execute(
                "SELECT COUNT(DISTINCT patient_id) FROM dicom_files WHERE parse_error IS NULL"
            ).fetchone()[0]
            studies = con.execute(
                "SELECT COUNT(DISTINCT study_instance_uid) FROM dicom_files WHERE parse_error IS NULL"
            ).fetchone()[0]
            series = con.execute(
                "SELECT COUNT(DISTINCT series_instance_uid) FROM dicom_files WHERE parse_error IS NULL"
            ).fetchone()[0]
            errors = con.execute(
                "SELECT COUNT(*) FROM dicom_files WHERE parse_error IS NOT NULL"
            ).fetchone()[0]
            modalities = con.execute(
                "SELECT modality, COUNT(*) as cnt FROM dicom_files "
                "WHERE parse_error IS NULL GROUP BY modality ORDER BY cnt DESC LIMIT 5"
            ).fetchall()

            rpt_table_exists = con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='radiology_reports'"
            ).fetchone()[0]
            if rpt_table_exists:
                rpt_total = con.execute("SELECT COUNT(*) FROM radiology_reports").fetchone()[0]
                rpt_linked = con.execute(
                    "SELECT COUNT(DISTINCT d.accession_number) "
                    "FROM dicom_files d "
                    "JOIN radiology_reports r USING (accession_number) "
                    "WHERE d.parse_error IS NULL"
                ).fetchone()[0]
                rpt_unlinked = con.execute(
                    "SELECT COUNT(DISTINCT d.accession_number) "
                    "FROM dicom_files d "
                    "LEFT JOIN radiology_reports r USING (accession_number) "
                    "WHERE d.parse_error IS NULL "
                    "  AND r.accession_number IS NULL "
                    "  AND d.accession_number NOT LIKE 'MISSING_%'"
                ).fetchone()[0]
            else:
                rpt_total = rpt_linked = rpt_unlinked = 0

            _db_stats_cache = {
                "instances": total,
                "patients": patients,
                "studies": studies,
                "series": series,
                "errors": errors,
                "modalities": [{"modality": r[0] or "?", "count": r[1]} for r in modalities],
                "db_path": str(_db_path),
                "rpt_total": rpt_total,
                "rpt_linked": rpt_linked,
                "rpt_unlinked": rpt_unlinked,
            }
            return _db_stats_cache
        finally:
            con.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Agentic NL→SQL pipeline (smolagents)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Instead of generating one SQL query and giving up (or doing a single blind
# "repair" pass) on failure, /api/query now runs a smolagents ToolCallingAgent
# that can call the `sql_query` tool as many times as it needs — exploring the
# schema, checking real column values, recovering from SQL errors, and
# refining its query — before producing a final natural-language answer. The
# agent talks to whichever local Ollama model is selected in the UI, via
# Ollama's OpenAI-compatible endpoint.

class SQLQueryTool(Tool):
    """
    A smolagents Tool enabling the agent to query the DICOM SQLite database.
    
    Allows the agent to execute read-only SELECT statements as many times as needed,
    exploring the schema, checking actual values, and recovering from SQL errors
    before producing a final answer.
    """

    name = "sql_query"
    description = (
        "Run a single read-only SQLite SELECT statement against the DICOM imaging "
        "database (tables: dicom_files, radiology_reports, image_ingestion_status) "
        "and get the matching rows back as JSON. You can call this tool multiple "
        "times in a row: explore first if you're unsure about a column's actual "
        "values (e.g. SELECT DISTINCT modality FROM dicom_files LIMIT 10), then "
        "write the real query. If a query fails, read the SQL ERROR message, fix "
        "the query, and call this tool again — never give up after one failed "
        "attempt."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "A single SQLite SELECT statement (no trailing semicolon-separated statements).",
        }
    }
    output_type = "string"

    def __init__(
        self,
        on_step_callback: Optional[Callable[[str, Optional[List[Dict[str, Any]]], int], None]] = None,
        max_rows: int = MAX_ROWS_FOR_SYNTHESIS,
    ) -> None:
        """
        Initialize the SQL query tool.
        
        Args:
            on_step_callback: Optional callback(sql, rows_or_None, row_count, error=None)
                             called after each query execution for streaming events
            max_rows: Maximum rows to return in payload (others truncated but counted)
        """
        super().__init__()
        self.on_step_callback = on_step_callback
        self.max_rows = max_rows

    def forward(self, query: str) -> str:
        """
        Execute a SQL query and return results as JSON.
        
        Args:
            query: SQL SELECT statement (may be wrapped in markdown code fences)
        
        Returns:
            JSON string with row_count, rows, and optional note/error messages
        """
        sql = clean_sql(query)

        try:
            rows = run_sql_query(sql)
        except Exception as exc:
            err = str(exc)
            if self.on_step_callback:
                self.on_step_callback(sql, None, 0, error=err)
            return (
                f"SQL ERROR: {err}\n\n"
                "Fix the query and call sql_query again. Reminders: every column in "
                "dicom_files is stored as TEXT — CAST numeric columns explicitly "
                "(e.g. CAST(frame_count AS INTEGER)); radrpt lives in "
                "radiology_reports, not dicom_files; join the two with "
                "JOIN radiology_reports USING (accession_number)."
            )

        def serialize_value(v: Any) -> Any:
            """Safely serialize a value to JSON-compatible format."""
            if v is None:
                return None
            try:
                json.dumps(v)
                return v
            except Exception:
                return str(v)

        clean_rows = [{k: serialize_value(v) for k, v in row.items()} for row in rows]
        if self.on_step_callback:
            self.on_step_callback(sql, clean_rows, len(clean_rows))

        display = clean_rows[: self.max_rows]
        payload: Dict[str, Any] = {"row_count": len(clean_rows), "rows": display}
        if len(clean_rows) > self.max_rows:
            payload["note"] = f"{len(clean_rows) - self.max_rows} additional rows truncated."
        elif not clean_rows:
            payload["note"] = "Query returned zero rows — consider relaxing filters or checking actual column values."
        return json.dumps(payload, default=str)


def get_smolagents_model(model_name: str) -> "OpenAIServerModel":
    """Build a smolagents model that talks to the local Ollama server via its
    OpenAI-compatible API, so the agent uses whatever model is selected in the UI."""
    if not SMOLAGENTS_OK:
        raise RuntimeError("smolagents is not installed. Run: pip install 'smolagents[openai]'")
    return OpenAIServerModel(
        model_id=model_name,
        api_base=f"{_ollama_host}/v1",
        api_key="ollama",   # Ollama ignores the key but the OpenAI client requires one
    )


def build_agent_task(question: str, think: bool) -> str:
    """Compose the full task text given to the agent: schema, SQL rules, answer
    style, and the user's question."""
    prefix = "" if think else "/no_think\n"
    return f"""{prefix}=== IMPORTANT INSTRUCTIONS ===

1. OFF-TOPIC FILTER:
   If the user's question is clearly NOT about the DICOM database (e.g., "Hi",
   "Hello", "What's the weather?"), respond politely WITHOUT using sql_query.

2. IMAGE / SEQUENCE REQUESTS:
   When the user asks to "show", "display", or "see" images, sequences, or cases:
   - ALWAYS include source_path in your SELECT — the UI renders thumbnail
     previews automatically from .dcm file paths in the results table.
   - Also include: frame_count, series_description, modality, study_date,
     accession_number so the user gets useful context alongside the images.

3. CLINICAL FINDING WORKFLOWS:
   When the user asks about a clinical procedure or finding (e.g., "TIPS",
   "stenosis", "embolization", "angioplasty"), follow this multi-step approach:

   Step 1 — SEARCH REPORTS: Find matching radiology reports first.
     SELECT accession_number, SUBSTR(radrpt, 1, 200) AS radrpt_excerpt
     FROM radiology_reports
     WHERE LOWER(radrpt) LIKE '%tips%'
     LIMIT 5

   Step 2 — FETCH SEQUENCES: Use the accession numbers from Step 1 to query
     DICOM sequences, including source_path for image display.
     SELECT source_path, frame_count, series_description, modality,
            study_date, accession_number
     FROM dicom_files
     WHERE accession_number IN ('acc1', 'acc2', ...)
       AND parse_error IS NULL
     ORDER BY CAST(frame_count AS INTEGER) DESC
     LIMIT 20

   This two-step approach (reports → accessions → sequences) is the correct
   workflow for any clinical finding query.

==============================

{SCHEMA_CONTEXT}

You are a clinical informatics assistant answering questions about a DICOM
angiography database using the sql_query tool. Some questions need more than
one query to answer correctly — explore the data first if you're unsure about
something (e.g. check distinct values, run a small LIMIT 5 sample), then
refine. Use the tool as many times as you need, and recover from SQL errors by
fixing the query and trying again, before giving your final answer.

{SYNTHESIS_SYSTEM}

Question: {question}

When you are confident in your answer, call final_answer with the plain-text
response described above.
"""


def run_nl_query_agent(question: str, think: bool, model_name: str) -> Any:
    """
    Run a smolagents ToolCallingAgent for NL→SQL query resolution.

    Emits rich SSE-compatible event dictionaries so the frontend can show
    exactly what the agent is doing at every step — which SQL it tried,
    how many rows came back, whether it hit an error, and what it's
    thinking.  This replaces the previous "black box" approach where the
    agent ran silently and only the final answer was visible.

    New events (in addition to the original protocol):
        agent_start   — agent is initializing (model, max_steps)
        agent_step    — one sql_query tool call completed (step#, sql,
                        row_count, error, max_steps)

    Original events (preserved for frontend compatibility):
        sql_done / sql_repaired, exec_start, exec_done,
        synth_start, answer, error
    """
    event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    call_count = {"n": 0}

    def on_query_step(
        sql: str,
        rows: Optional[List[Dict[str, Any]]],
        row_count: int,
        error: Optional[str] = None,
    ) -> None:
        """Callback invoked after each sql_query tool call."""
        call_count["n"] += 1

        # ── NEW: rich step event so the UI can show a step-by-step log ──
        event_queue.put({
            "event":     "agent_step",
            "step":      call_count["n"],
            "max_steps": _agent_max_steps,
            "sql":       sql,
            "row_count": row_count,
            "error":     error,
        })

        # ── Original events (kept for backward compat) ──────────────────
        event_queue.put({
            "event": "sql_done" if call_count["n"] == 1 else "sql_repaired",
            "sql": sql,
        })
        event_queue.put({"event": "exec_start"})
        event_queue.put({
            "event": "exec_done",
            "rows": rows or [],
            "row_count": row_count,
        })

    result_holder: Dict[str, Optional[str]] = {"answer": None, "error": None}

    def worker() -> None:
        """Background thread running the smolagents agent."""
        try:
            # ── Notify UI that the agent is starting ────────────────────
            event_queue.put({
                "event":     "agent_start",
                "model":     model_name,
                "max_steps": _agent_max_steps,
            })

            model = get_smolagents_model(model_name)
            tool  = SQLQueryTool(on_step_callback=on_query_step)
            agent = ToolCallingAgent(
                tools=[tool], model=model, max_steps=_agent_max_steps,
            )
            task = build_agent_task(question, think)

            log.info(
                f"Agent starting: model={model_name}, "
                f"max_steps={_agent_max_steps}, q={question!r:.80}"
            )

            raw_answer = agent.run(task)
            answer = re.sub(
                r"<think>.*?</think>", "", str(raw_answer), flags=re.DOTALL,
            ).strip()
            result_holder["answer"] = answer

            log.info(
                f"Agent finished: {call_count['n']} tool call(s), "
                f"answer length={len(answer)}"
            )

        except Exception as exc:
            tb = traceback.format_exc()
            log.error(f"Agent execution failed:\n{tb}")
            result_holder["error"] = f"{exc}\n\n--- traceback ---\n{tb}"
        finally:
            event_queue.put({"event": "__agent_done__"})

    threading.Thread(target=worker, daemon=True).start()

    # ── Yield events as the agent emits them ────────────────────────────
    while True:
        item = event_queue.get()
        if item["event"] == "__agent_done__":
            break
        yield item

    # ── Final outcome ───────────────────────────────────────────────────
    if result_holder["error"]:
        yield {
            "event":      "error",
            "message":    f"Agent failed: {result_holder['error']}",
            "tool_calls": call_count["n"],
        }
    else:
        yield {"event": "synth_start"}
        final_answer = (
            result_holder["answer"]
            or "(The agent did not produce an answer.)"
        )
        yield {"event": "answer", "text": final_answer}


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDB / image helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_chroma_collection() -> Optional[Any]:
    """
    Get or lazy-initialize the ChromaDB collection.
    
    Uses PersistentClient with cosine distance metric. The collection should have been
    pre-indexed with RAD-DINO embeddings during image ingestion. We do NOT pass an
    embedding_function here because the collection was created with a specific function
    during ingestion; we manually compute embeddings via RAD-DINO and pass them to
    the query as query_embeddings parameter.
    
    Returns:
        ChromaDB collection object if available, else None
    """
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    if not IMAGE_DEPS_OK:
        return None
    try:
        client = _chromadb_mod.PersistentClient(path=str(_chromadb_path))
        _chroma_collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            f"ChromaDB collection '{CHROMA_COLLECTION}' ready at {_chromadb_path} "
            f"— {_chroma_collection.count():,} items"
        )
        return _chroma_collection
    except Exception as exc:
        log.error(f"ChromaDB init failed: {exc}")
        return None


def decode_image_to_uint8_rgb(b64_str: str) -> np.ndarray:
    """
    Decode base64-encoded image to uint8 RGB numpy array.
    
    Args:
        b64_str: Base64-encoded image string
    
    Returns:
        Numpy array with shape (height, width, 3) and dtype uint8
    
    Raises:
        Exception: On base64 decode or image format errors
    """
    img_bytes = base64.b64decode(b64_str)
    img = PilImage.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def enrich_results_from_sqlite(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich ChromaDB search results with SQLite metadata and report excerpts.
    
    Adds study_date, series_description, modality, patient demographics, and
    first 400 characters of the radiology report for each result by accession number.
    
    Non-fatal: returns original results if enrichment fails.
    
    Args:
        results: ChromaDB search result dicts with accession_number keys
    
    Returns:
        Results with added enrichment fields
    """
    accessions = list({r["accession_number"] for r in results if r.get("accession_number")})
    
    if not accessions:
        return results

    enrichment_map: Dict[str, Dict[str, Any]] = {}
    try:
        placeholders = ", ".join(["?"] * len(accessions))
        with _lock:
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
    Extract and encode a DICOM frame as base64-encoded PNG.
    
    Opens the DICOM file, extracts the frame at frame_index, normalises pixel values
    to 0–255 uint8 RGB, optionally inverts for MONOCHROME1, resizes to max dimension
    thumb_px, and encodes as PNG.
    
    Args:
        source_path: Path to .dcm file
        frame_index: Frame number to extract (0-indexed)
        thumb_px: Maximum dimension in pixels for thumbnail resize
    
    Returns:
        Base64-encoded PNG string, or None on any error
    """
    if not IMAGE_DEPS_OK or not PYDICOM_OK:
        return None
    try:
        ds = _pydicom_mod.dcmread(str(source_path), force=False)
        photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        pixels = ds.pixel_array

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


# ═══════════════════════════════════════════════════════════════════════════════
# API routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/stats", methods=["GET"])
def api_stats() -> Dict[str, Any]:
    """
    GET /api/stats
    
    Returns database statistics: instance count, unique patients, studies, series,
    error count, modalities breakdown, and radiology report linkage coverage.
    """
    try:
        return jsonify(get_db_stats())
    except Exception as exc:
        log.exception("GET /api/stats failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/chroma-stats", methods=["GET"])
def api_chroma_stats() -> Dict[str, Any]:
    """
    GET /api/chroma-stats
    
    Returns ChromaDB collection statistics: total embedded frames, ingested sequences,
    collection name, and filesystem path. Returns 'available': False if dependencies
    or collection unavailable.
    """
    if not IMAGE_DEPS_OK:
        return jsonify({
            "available": False,
            "error": "Image dependencies not installed (chromadb, Pillow, numpy)",
        })
    try:
        col = get_chroma_collection()
        if col is None:
            return jsonify({"available": False, "count": 0, "sequences": 0})

        count = col.count()
        seq_count: Optional[int] = None
        try:
            with _lock:
                con = open_db()
                try:
                    seq_count = con.execute(
                        "SELECT COUNT(*) FROM image_ingestion_status WHERE status='completed'"
                    ).fetchone()[0]
                finally:
                    con.close()
        except Exception as exc:
            log.debug(f"Could not fetch sequence count: {exc}")

        return jsonify({
            "available": True,
            "count": count,
            "sequences": seq_count,
            "collection": CHROMA_COLLECTION,
            "path": str(_chromadb_path),
        })
    except Exception as exc:
        log.exception("GET /api/chroma-stats failed")
        return jsonify({"available": False, "error": str(exc)})


@app.route("/api/query", methods=["POST"])
def api_query() -> Response:
    """
    POST /api/query
    
    Agentic NL→SQL query resolution via smolagents ToolCallingAgent.
    
    Request body:
        {
            "question": "Natural language query",
            "think": true/false,
            "model": "qwen3:1.7b" or other Ollama model
        }
    
    Returns Server-Sent Events (SSE) stream with events:
        - sql_start: Agent beginning SQL generation
        - sql_done/sql_repaired: SQL statement ready
        - exec_start/exec_done: Query execution and results
        - synth_start: Answer synthesis in progress
        - answer: Final natural-language response
        - done: Query completed with elapsed_ms
        - error: Error message if any step fails
    
    The agent can call sql_query tool multiple times to explore, recover from
    errors, and refine its answer before responding. All SSE event names match
    the original single-shot pipeline for frontend compatibility.
    """
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    think = data.get("think", _think)
    model = (data.get("model") or DEFAULT_MODEL).strip()

    if not question:
        log.warning("POST /api/query: missing 'question' parameter")
        return jsonify({"error": "question required"}), 400

    if not SMOLAGENTS_OK:
        log.error("POST /api/query: smolagents not installed")
        return jsonify({
            "error": "smolagents is not installed on the server. Run: pip install 'smolagents[openai]'"
        }), 503

    def generate():
        t0 = time.time()

        def emit(obj: Dict[str, Any]) -> str:
            return "data: " + json.dumps(obj) + "\n\n"

        yield emit({"event": "sql_start"})

        had_error = False
        for evt in run_nl_query_agent(question, think, model):
            yield emit(evt)
            if evt.get("event") == "error":
                had_error = True
                break

        if not had_error:
            elapsed = int((time.time() - t0) * 1000)
            yield emit({"event": "done", "elapsed_ms": elapsed})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/image-query", methods=["POST"])
def api_image_query() -> Response:
    """
    POST {
      "image":     "<base64-encoded image>",
      "question":  "Show me similar cases",   // optional
      "n_results": 5,                          // optional
      "think":     true                        // optional
    }

    Streaming SSE events:
      decode_start  → decode_done  {width, height}
      chroma_start  → chroma_done  {results, count}
      enrich_start  → enrich_done  {results}
      synth_start   → answer {text} → done {elapsed_ms}
      error {message}

    Groups frames by SOPInstanceUID so each result represents a full series
    with all frames available for browsing.
    """
    data      = request.get_json(force=True)
    b64_image = data.get("image", "").strip()
    question  = data.get("question", "Show me the most similar cases to this image.").strip()
    n_results = max(1, min(20, int(data.get("n_results", N_SIMILAR_DEFAULT))))
    think     = data.get("think", _think)

    if not b64_image:
        return jsonify({"error": "image required"}), 400

    if not IMAGE_DEPS_OK:
        return jsonify({
            "error": (
                "Image query dependencies not installed on server. "
                "Run: pip install chromadb pillow numpy torch torchvision timm"
            )
        }), 503

    def generate():
        t0 = time.time()

        def emit(obj: dict) -> str:
            return "data: " + json.dumps(obj) + "\n\n"

        # ── Step 1: Decode image ─────────────────────────────────────────────
        yield emit({"event": "decode_start"})
        try:
            img_array = decode_image_to_uint8_rgb(b64_image)
            h, w      = img_array.shape[:2]
            yield emit({"event": "decode_done", "width": int(w), "height": int(h)})
        except Exception as exc:
            yield emit({"event": "error", "message": f"Image decode failed: {exc}"})
            return

        # ── Step 2: ChromaDB similarity search ───────────────────────────────
        yield emit({"event": "chroma_start"})
        try:
            col = get_chroma_collection()
            if col is None:
                yield emit({
                    "event":   "error",
                    "message": (
                        "ChromaDB collection not available. "
                        "Run image ingestion (dicom_ingest_sqlite.py --images-only) first."
                    ),
                })
                return

            available = col.count()
            if available == 0:
                yield emit({
                    "event":   "error",
                    "message": "ChromaDB collection is empty. Ingest images first.",
                })
                return

            # Over-fetch to allow grouping by SOPInstanceUID
            fetch_n = min(available, N_SIMILAR_OVERFETCH)

            # Get RAD-DINO model for query embedding
            ef = get_raddino_model()
            if ef is None:
                # Fallback to query_images parameter
                raw = col.query(
                    query_images=[img_array],
                    n_results=fetch_n,
                    include=["metadatas", "distances"],
                )
            else:
                # Use RAD-DINO embeddings
                query_embedding = ef([img_array])[0]
                raw = col.query(
                    query_embeddings=[query_embedding],
                    n_results=fetch_n,
                    include=["metadatas", "distances"],
                )
            
            ids       = raw["ids"][0]
            distances = raw["distances"][0]
            metadatas = raw["metadatas"][0]

            # Group by accession_number → then by sop_uid within each accession
            acc_groups: dict[str, dict] = {}
            for id_, dist, meta in zip(ids, distances, metadatas):
                acc = meta.get("accession_number") or "UNKNOWN"
                sop = meta.get("sop_uid", "") or "UNKNOWN"
                sim = max(0.0, 1.0 - float(dist))
                frame_entry = {
                    "chroma_id":      id_,
                    "similarity_pct": round(sim * 100, 1),
                    "distance":       round(float(dist), 4),
                    "source_path":    meta.get("source_path", ""),
                    "frame_index":    meta.get("frame_index", 0),
                    "sop_uid":        sop,
                }
                if acc not in acc_groups:
                    acc_groups[acc] = {
                        "accession_number": acc,
                        "_top_sim":         sim,
                        "sop_groups":       {},   # { sop_uid: {frames, _top_sim} }
                        **{k: (v if v is not None else "") for k, v in meta.items()},
                    }
                if sim > acc_groups[acc]["_top_sim"]:
                    acc_groups[acc]["_top_sim"] = sim

                # Inner group: one entry per sop_uid
                sop_dict = acc_groups[acc]["sop_groups"]
                if sop not in sop_dict:
                    sop_dict[sop] = {"sop_uid": sop, "_top_sim": sim, "frames": []}
                if sim > sop_dict[sop]["_top_sim"]:
                    sop_dict[sop]["_top_sim"] = sim
                sop_dict[sop]["frames"].append(frame_entry)

            # Finalise: sort frames within each SOP, sort SOPs within each accession
            for grp in acc_groups.values():
                sop_list = []
                for sg in grp["sop_groups"].values():
                    sg["frames"].sort(key=lambda f: f["similarity_pct"], reverse=True)
                    sg["similarity_pct"] = round(sg.pop("_top_sim") * 100, 1)
                    sop_list.append(sg)
                sop_list.sort(key=lambda s: s["similarity_pct"], reverse=True)
                grp["sop_groups"] = sop_list
                grp["similarity_pct"] = round(grp["_top_sim"] * 100, 1)

            # Sort accessions by top similarity, keep top n_results
            top = sorted(acc_groups.values(), key=lambda x: x["_top_sim"], reverse=True)[:n_results]
            results = []
            for i, r in enumerate(top):
                r.pop("_top_sim", None)
                r["rank"] = i + 1
                results.append(r)

            yield emit({"event": "chroma_done", "results": results, "count": len(results)})

        except Exception as exc:
            yield emit({"event": "error", "message": f"ChromaDB query failed: {exc}"})
            return

        # ── Step 3: Enrich from SQLite ───────────────────────────────────────
        yield emit({"event": "enrich_start"})
        try:
            enriched = enrich_results_from_sqlite(results)
        except Exception as exc:
            log.warning(f"Enrichment error (non-fatal): {exc}")
            enriched = results
        yield emit({"event": "enrich_done", "results": enriched})

        # ── Step 4: LLM synthesis ────────────────────────────────────────────
        yield emit({"event": "synth_start"})
        try:
            cases_text = "\n".join(
                f"#{r['rank']} ({len(r.get('sop_groups', []))} SOP(s), top similarity {r['similarity_pct']}%): "
                f"Accession={r.get('accession_number','?')}, "
                f"Date={r.get('study_date','?')}, "
                f"Modality={r.get('modality','?')}, "
                f"Series={r.get('series_description','?')}, "
                f"Age={r.get('patient_age','?')}, Sex={r.get('patient_sex','?')}"
                + (f"\n   Report excerpt: {str(r['radrpt_excerpt'])[:250]}"
                   if r.get("radrpt_excerpt") else "")
                for r in enriched
            )
            answer = llm_call([
                {"role": "system", "content": IMAGE_SYNTHESIS_SYSTEM},
                {"role": "user",   "content": (
                    f"Question: {question}\n\n"
                    f"Top {len(enriched)} visually similar series retrieved from ChromaDB "
                    f"(RAD-DINO embedding similarity):\n\n{cases_text}"
                )},
            ], think=False)
        except Exception as exc:
            answer = f"(Synthesis unavailable: {exc})"

        elapsed = int((time.time() - t0) * 1000)
        yield emit({"event": "answer",  "text": answer})
        yield emit({"event": "done",    "elapsed_ms": elapsed})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _validate_dicom_path(path: str) -> bool:
    """Reject path traversal attempts and restrict to .dcm files."""
    resolved = os.path.realpath(path)
    if ".." in path or not resolved.lower().endswith(".dcm"):
        return False
    allowed_root = os.environ.get("DICOM_DATA_ROOT", "/data")
    return resolved.startswith(os.path.realpath(allowed_root))


@app.route("/api/thumbnail", methods=["GET"])
def api_thumbnail() -> Response:
    """
    GET /api/thumbnail?path=<file_path>&frame=<frame_index>
    
    Extract and return a thumbnail (320px max dimension) PNG from a DICOM frame.
    
    Parameters:
        path: Full path to .dcm file (required, must be under DICOM_DATA_ROOT)
        frame: Frame index (0-indexed, default 0)
    
    Returns:
        Binary PNG image (image/png) or 404 if frame cannot be extracted
    """
    path = request.args.get("path", "").strip()
    if not path:
        log.warning("GET /api/thumbnail: missing 'path' parameter")
        return jsonify({"error": "path required"}), 400
    if not _validate_dicom_path(path):
        log.warning(f"GET /api/thumbnail: path rejected [{path}]")
        return jsonify({"error": "invalid path"}), 403

    try:
        frame_index = int(request.args.get("frame", 0))
    except ValueError:
        frame_index = 0

    thumbnail_b64 = load_dicom_frame_as_b64(path, frame_index, thumb_px=320)
    if thumbnail_b64:
        img_bytes = base64.b64decode(thumbnail_b64)
        return Response(
            img_bytes,
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    else:
        log.debug(f"GET /api/thumbnail: frame not extracted [{path}:{frame_index}]")
        return Response(status=404)


@app.route("/api/frame", methods=["GET"])
def api_frame() -> Response:
    """
    GET /api/frame?path=<file_path>&frame=<frame_index>
    
    Extract and return a full-resolution PNG from a DICOM frame (lightbox display).
    
    Parameters:
        path: Full path to .dcm file (required, must be under DICOM_DATA_ROOT)
        frame: Frame index (0-indexed, default 0)
    
    Returns:
        Binary PNG image (image/png) or 404 if frame cannot be extracted
    """
    path = request.args.get("path", "").strip()
    if not path:
        log.warning("GET /api/frame: missing 'path' parameter")
        return jsonify({"error": "path required"}), 400
    if not _validate_dicom_path(path):
        log.warning(f"GET /api/frame: path rejected [{path}]")
        return jsonify({"error": "invalid path"}), 403

    try:
        frame_index = int(request.args.get("frame", 0))
    except ValueError:
        frame_index = 0

    frame_b64 = load_dicom_frame_as_b64(path, frame_index, thumb_px=2048)
    if frame_b64:
        img_bytes = base64.b64decode(frame_b64)
        return Response(
            img_bytes,
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    else:
        log.debug(f"GET /api/frame: frame not extracted [{path}:{frame_index}]")
        return Response(status=404)


@app.route("/api/model", methods=["POST"])
def api_set_model() -> Dict[str, Any]:
    """
    POST /api/model
    
    Update the Ollama model used for NL→SQL agent and synthesis.
    
    Request body:
        {"model": "qwen3:14b" or other Ollama model identifier}
    
    Returns:
        {"ok": true, "model": "<model_name>"}
    """
    data = request.get_json(force=True)
    model = data.get("model", "").strip()
    if not model:
        log.warning("POST /api/model: missing 'model' parameter")
        return jsonify({"error": "model required"}), 400
    set_model(model)
    log.info(f"POST /api/model: switched to {model}")
    return jsonify({"ok": True, "model": model})


_SQL_BLOCKED_KEYWORDS = re.compile(
    r"\b(DROP|ALTER|CREATE|INSERT|UPDATE|DELETE|REPLACE|ATTACH|DETACH|REINDEX|VACUUM)\b",
    re.IGNORECASE,
)


@app.route("/api/sql", methods=["POST"])
def api_run_sql() -> Dict[str, Any]:
    """
    POST /api/sql
    
    Execute read-only SQL directly against the DICOM database.
    For advanced/debugging use; not used by the standard agentic pipeline.
    Only SELECT and PRAGMA statements are permitted.
    
    Request body:
        {"sql": "SELECT ... FROM dicom_files WHERE ..."}
    
    Returns:
        {"rows": [...], "row_count": <int>} or error
    """
    data = request.get_json(force=True)
    sql = data.get("sql", "").strip()
    if not sql:
        log.warning("POST /api/sql: missing 'sql' parameter")
        return jsonify({"error": "sql required"}), 400
    if _SQL_BLOCKED_KEYWORDS.search(sql):
        log.warning("POST /api/sql: blocked mutating statement")
        return jsonify({"error": "only SELECT queries are allowed"}), 403
    try:
        rows = run_sql_query(sql)

        def serialize_value(v: Any) -> Any:
            """Safely serialize a value to JSON-compatible format."""
            if v is None:
                return None
            try:
                json.dumps(v)
                return v
            except Exception:
                return str(v)

        clean_rows = [{k: serialize_value(v) for k, v in row.items()} for row in rows]
        return jsonify({"rows": clean_rows, "row_count": len(clean_rows)})
    except Exception as exc:
        log.exception("POST /api/sql: execution failed")
        return jsonify({"error": str(exc)}), 400


@app.route("/", methods=["GET"])
def index() -> str:
    """Serve the main HTML UI for the DICOM query engine."""
    creds_json = os.environ.get("ANGIOVISION_USERS", "[]")
    try:
        json.loads(creds_json)
    except (json.JSONDecodeError, TypeError):
        creds_json = "[]"
    return HTML_PAGE.replace("__CREDENTIALS_PLACEHOLDER__", creds_json)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML UI
# ═══════════════════════════════════════════════════════════════════════════════
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AngioVision · Query Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--bg2:#161b22;--bg3:#1e2530;--bg4:#252d3a;--bg5:#2d3748;
  --border:#30363d;--border2:#3d4450;
  --text:#e6edf3;--text2:#8b949e;--text3:#57636d;
  --accent:#58a6ff;--accent2:#1f6feb;--accent-bg:rgba(88,166,255,.1);
  --green:#3fb950;--amber:#d29922;--red:#f85149;--purple:#bc8cff;--teal:#39c5cf;
  --report:#f0883e;--report-bg:rgba(240,136,62,.1);
  --violet:#7c3aed;--violet-light:#a78bfa;--violet-bg:rgba(124,58,237,.12);
  --mono:'JetBrains Mono',monospace;
  --sans:'Syne',sans-serif;
}
[data-theme="light"]{
  --bg:#f0f2f5;--bg2:#ffffff;--bg3:#f5f7fa;--bg4:#eaecf0;--bg5:#dde1e7;
  --border:#d0d5dd;--border2:#c4c9d4;
  --text:#111827;--text2:#4b5563;--text3:#9ca3af;
  --accent:#1d6fdb;--accent2:#1558b0;--accent-bg:rgba(29,111,219,.08);
  --green:#16a34a;--amber:#b45309;--red:#dc2626;--purple:#7c3aed;--teal:#0891b2;
  --report:#c2410c;--report-bg:rgba(194,65,12,.08);
  --violet:#6d28d9;--violet-light:#7c3aed;--violet-bg:rgba(109,40,217,.08);
}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:var(--sans)}

/* ══ LOGIN OVERLAY ══ */
#loginOverlay{position:fixed;inset:0;z-index:9999;background:var(--bg);display:flex;align-items:center;justify-content:center;transition:opacity .4s ease,visibility .4s ease}
#loginOverlay.hidden{opacity:0;visibility:hidden;pointer-events:none}
#loginOverlay::before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(88,166,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(88,166,255,.04) 1px,transparent 1px);background-size:40px 40px;animation:gridDrift 20s linear infinite}
@keyframes gridDrift{from{background-position:0 0}to{background-position:40px 40px}}
#loginOverlay::after{content:'';position:absolute;width:500px;height:500px;background:radial-gradient(circle,rgba(31,111,235,.15) 0%,transparent 70%);border-radius:50%;pointer-events:none}
.login-card{position:relative;z-index:1;width:380px;background:var(--bg2);border:1px solid var(--border2);border-radius:16px;padding:36px 32px 32px;box-shadow:0 0 0 1px rgba(88,166,255,.08),0 24px 60px rgba(0,0,0,.6);animation:cardIn .5s cubic-bezier(.22,1,.36,1) both}
@keyframes cardIn{from{opacity:0;transform:translateY(20px) scale(.97)}to{opacity:1;transform:translateY(0) scale(1)}}
.login-logo{width:48px;height:48px;background:linear-gradient(135deg,var(--accent2),#0a3d7a);border-radius:12px;display:flex;align-items:center;justify-content:center;margin:0 auto 20px;box-shadow:0 0 20px rgba(31,111,235,.4)}
.login-logo svg{width:22px;height:22px;stroke:#fff;fill:none;stroke-width:2}
.login-title{text-align:center;font-size:19px;font-weight:800;letter-spacing:-.4px;margin-bottom:4px}
.login-sub{text-align:center;font-size:11px;font-family:var(--mono);color:var(--text3);letter-spacing:.4px;margin-bottom:28px}
.login-field{margin-bottom:16px}
.login-label{display:block;font-size:10px;font-family:var(--mono);color:var(--text3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:7px}
.login-input{width:100%;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:10px 14px;font-size:13px;font-family:var(--mono);color:var(--text);outline:none;transition:border-color .2s,box-shadow .2s}
.login-input:focus{border-color:var(--accent2);box-shadow:0 0 0 3px rgba(31,111,235,.18)}
.login-input::placeholder{color:var(--text3)}
.pw-wrap{position:relative}
.pw-wrap .login-input{padding-right:40px}
.pw-toggle{position:absolute;right:12px;top:50%;transform:translateY(-50%);background:none;border:none;cursor:pointer;padding:2px;color:var(--text3);transition:color .15s;line-height:0}
.pw-toggle:hover{color:var(--text2)}
.pw-toggle svg{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:2}
.login-error{display:none;background:rgba(248,81,73,.1);border:1px solid rgba(248,81,73,.3);border-radius:7px;padding:9px 12px;font-size:11px;font-family:var(--mono);color:#ffb3b3;margin-bottom:14px;text-align:center}
.login-error.visible{display:block;animation:errIn .2s ease}
@keyframes errIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
.login-btn{width:100%;padding:11px;background:var(--accent2);border:none;border-radius:8px;font-size:13px;font-family:var(--sans);font-weight:700;color:#fff;cursor:pointer;transition:background .2s,transform .1s,box-shadow .2s;letter-spacing:.2px;margin-top:4px}
.login-btn:hover{background:var(--accent);box-shadow:0 4px 16px rgba(88,166,255,.25)}
.login-btn:active{transform:scale(.98)}
.login-btn:disabled{background:var(--bg4);color:var(--text3);cursor:not-allowed;box-shadow:none}
@keyframes shake{0%,100%{transform:translateX(0)}15%{transform:translateX(-7px)}30%{transform:translateX(7px)}45%{transform:translateX(-5px)}60%{transform:translateX(5px)}75%{transform:translateX(-3px)}90%{transform:translateX(3px)}}
.login-card.shake{animation:shake .45s ease}
.login-footer{margin-top:20px;padding-top:16px;border-top:1px solid var(--border);font-size:10px;font-family:var(--mono);color:var(--text3);text-align:center;line-height:1.6}
.login-footer span{color:var(--accent);font-weight:600}
.attempt-bar{height:2px;background:var(--bg4);border-radius:1px;margin-top:10px;overflow:hidden}
.attempt-fill{height:2px;background:var(--red);border-radius:1px;width:0%;transition:width .3s ease}

/* ══ APP WRAPPER ══ */
#appWrapper{height:100%;opacity:0;transition:opacity .4s ease .1s;pointer-events:none}
#appWrapper.visible{opacity:1;pointer-events:auto}

/* ── Layout ── */
.layout{display:grid;grid-template-rows:56px 1fr;grid-template-columns:1fr 320px;height:100vh;grid-template-areas:"hdr hdr" "chat side"}
@media(max-width:900px){.layout{grid-template-columns:1fr;grid-template-areas:"hdr" "chat"}.sidebar{display:none!important}}

/* ── Header ── */
.header{grid-area:hdr;background:var(--bg2);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 20px;gap:14px;z-index:10}
.logo{width:30px;height:30px;background:var(--accent2);border-radius:7px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.logo svg{width:16px;height:16px;stroke:#fff;fill:none;stroke-width:2}
.hdr-title{font-size:15px;font-weight:800;letter-spacing:-.3px}
.hdr-sep{width:1px;height:20px;background:var(--border);margin:0 4px}
.hdr-sub{font-size:11px;font-family:var(--mono);color:var(--text3)}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:10px}
.status-pill{display:flex;align-items:center;gap:6px;font-size:11px;font-family:var(--mono);color:var(--text2);background:var(--bg3);border:1px solid var(--border);border-radius:100px;padding:4px 10px}
.status-dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 5px var(--green)}
.model-select{background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:4px 10px;font-size:11px;font-family:var(--mono);color:var(--accent);cursor:pointer;appearance:none;outline:none;transition:border-color .2s}
.model-select:hover{border-color:var(--accent)}
.logout-btn{background:none;border:1px solid var(--border);border-radius:6px;padding:4px 10px;font-size:10px;font-family:var(--mono);color:var(--text3);cursor:pointer;transition:all .15s;letter-spacing:.3px;text-transform:uppercase}
.logout-btn:hover{border-color:var(--red);color:var(--red)}
.theme-toggle{width:32px;height:32px;border-radius:7px;border:1px solid var(--border);background:var(--bg3);cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .2s;color:var(--text2)}
.theme-toggle:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-bg)}
.theme-toggle svg{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:2;transition:opacity .2s}

/* ── Chat column ── */
.chat-col{grid-area:chat;display:flex;flex-direction:column;overflow:hidden;min-width:0}
.messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px;scroll-behavior:smooth}
.messages::-webkit-scrollbar{width:5px}.messages::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

/* ── Messages ── */
.msg{display:flex;flex-direction:column;gap:0;animation:fadeUp .25s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.msg-user{align-items:flex-end}
.msg-bot{align-items:flex-start}
.bubble{padding:10px 14px;border-radius:12px;font-size:13.5px;line-height:1.65;max-width:72%}
.bubble-user{background:var(--accent2);color:#fff;border-bottom-right-radius:3px;font-weight:500}
.bubble-error{background:#2d1515;border:1px solid #6b2020;color:#ffb3b3;border-radius:10px;font-family:var(--mono);font-size:12px;padding:10px 14px;max-width:90%}

/* User bubble with image */
.bubble-with-img{display:flex;flex-direction:column;gap:8px;align-items:flex-end;background:var(--accent2);border-radius:12px;border-bottom-right-radius:3px;padding:10px 14px;max-width:72%}
.bubble-img-preview{max-width:220px;max-height:160px;border-radius:8px;object-fit:cover;border:2px solid rgba(255,255,255,.2);display:block}
.bubble-img-caption{font-size:13.5px;font-weight:500;color:#fff;line-height:1.55}

/* ── Result card (SQL queries) ── */
.result-card{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.rc-tabs{display:flex;border-bottom:1px solid var(--border);background:var(--bg3)}
.rc-tab{padding:9px 16px;font-size:11px;font-family:var(--mono);background:none;border:none;color:var(--text3);cursor:pointer;position:relative;transition:color .15s;letter-spacing:.3px;text-transform:uppercase}
.rc-tab:hover{color:var(--text2)}
.rc-tab.active{color:var(--accent)}
.rc-tab.active::after{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;background:var(--accent)}
.rc-meta{margin-left:auto;display:flex;align-items:center;gap:8px;padding-right:12px;font-size:10px;font-family:var(--mono);color:var(--text3)}
.rc-pane{display:none}.rc-pane.active{display:block}
.answer-body{padding:14px 16px;font-size:13.5px;line-height:1.75;color:var(--text);white-space:pre-wrap;font-family:var(--sans)}
.sql-block{padding:14px 16px;font-family:var(--mono);font-size:12px;line-height:1.8;overflow-x:auto;color:#c9d1d9;white-space:pre}
.kw{color:#ff7b72;font-weight:600}.fn{color:#d2a8ff}.str{color:#a5d6ff}.nm{color:#79c0ff}.cmt{color:var(--text3)}
.tbl-scroll{overflow-x:auto;max-height:400px;overflow-y:auto}
.tbl-scroll::-webkit-scrollbar{height:5px;width:5px}.tbl-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.data-table{border-collapse:collapse;width:max-content;min-width:100%;font-family:var(--mono);font-size:11.5px}
.data-table thead{position:sticky;top:0;z-index:2}
.data-table th{background:var(--bg4);padding:8px 14px;text-align:left;color:var(--text2);white-space:nowrap;border-bottom:1px solid var(--border2);font-size:10px;font-weight:600;letter-spacing:.5px;text-transform:uppercase}
.data-table td{padding:7px 14px;border-bottom:1px solid rgba(48,54,61,.5);white-space:nowrap;color:var(--text);vertical-align:middle;max-width:300px;overflow:hidden;text-overflow:ellipsis}
.data-table tr:last-child td{border-bottom:none}
.data-table tr:hover td{background:rgba(255,255,255,.03)}
.data-table tr:nth-child(even) td{background:rgba(255,255,255,.015)}
.v-null{color:var(--text3);font-style:italic}.v-num{color:#79c0ff}.v-path{color:var(--purple);font-size:10.5px}.v-id{color:var(--teal);font-size:10.5px}.v-report{color:var(--report);font-size:10.5px;font-style:italic}
.tbl-footer{padding:8px 14px;font-size:10px;font-family:var(--mono);color:var(--text3);border-top:1px solid var(--border);background:var(--bg3);display:flex;align-items:center;gap:8px}

/* ── Agent step log ── */
.agent-steps{padding:10px 14px;font-family:var(--mono);font-size:11.5px}
.agent-steps-empty{color:var(--text3);font-style:italic;padding:10px 0}
.agent-step-item{border-left:2px solid var(--border);padding:6px 0 6px 12px;margin-bottom:8px;transition:border-color .2s}
.agent-step-item:last-child{margin-bottom:0}
.agent-step-item.step-ok{border-left-color:var(--green)}
.agent-step-item.step-err{border-left-color:var(--red)}
.step-hdr{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.step-badge{font-size:9px;font-weight:700;padding:1px 6px;border-radius:100px;letter-spacing:.3px;text-transform:uppercase}
.step-ok .step-badge{background:rgba(63,185,80,.12);color:var(--green);border:1px solid rgba(63,185,80,.25)}
.step-err .step-badge{background:rgba(248,81,73,.12);color:var(--red);border:1px solid rgba(248,81,73,.25)}
.step-meta{font-size:10px;color:var(--text3)}
.step-sql{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:6px 10px;margin:4px 0;overflow-x:auto;white-space:pre-wrap;word-break:break-all;font-size:11px;line-height:1.6;color:#c9d1d9}
.step-result{font-size:10.5px;color:var(--green);margin-top:2px}
.step-error-msg{font-size:10.5px;color:var(--red);margin-top:2px;white-space:pre-wrap}

/* ── DICOM thumbnails in table cells ── */
.dcm-cell{display:flex;align-items:center;gap:8px;min-width:200px}
.tbl-thumb{width:52px;height:52px;object-fit:cover;border-radius:5px;border:1px solid var(--border);cursor:pointer;transition:transform .15s,border-color .15s,box-shadow .15s;background:var(--bg);flex-shrink:0}
.tbl-thumb:hover{transform:scale(1.15);border-color:var(--accent);box-shadow:0 0 10px rgba(88,166,255,.3)}
.tbl-thumb-err{width:52px;height:52px;border-radius:5px;border:1px dashed var(--border);display:flex;align-items:center;justify-content:center;font-size:9px;color:var(--text3);font-family:var(--mono);background:var(--bg);flex-shrink:0}

/* ── Image result card (ChromaDB) ── */
.img-result-card{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.irc-header{display:flex;align-items:center;gap:10px;padding:11px 16px;border-bottom:1px solid var(--border);background:var(--bg3)}
.irc-icon{width:28px;height:28px;background:linear-gradient(135deg,var(--violet),#4338ca);border-radius:7px;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 0 12px rgba(124,58,237,.35)}
.irc-icon svg{width:14px;height:14px;stroke:#fff;fill:none;stroke-width:2}
.irc-title{font-size:13px;font-weight:700;color:var(--text)}
.irc-badge{font-size:9px;font-family:var(--mono);background:var(--violet-bg);border:1px solid rgba(124,58,237,.3);color:var(--violet-light);border-radius:100px;padding:2px 8px;letter-spacing:.3px;text-transform:uppercase}
.irc-time{margin-left:auto;font-size:10px;font-family:var(--mono);color:var(--text3)}
.irc-tabs{display:flex;border-bottom:1px solid var(--border);background:var(--bg3)}
.irc-tab{padding:8px 14px;font-size:11px;font-family:var(--mono);background:none;border:none;color:var(--text3);cursor:pointer;position:relative;transition:color .15s;text-transform:uppercase;letter-spacing:.3px}
.irc-tab:hover{color:var(--text2)}
.irc-tab.active{color:var(--violet-light)}
.irc-tab.active::after{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;background:var(--violet-light)}
.irc-count{font-size:9px;background:var(--violet-bg);color:var(--violet-light);border-radius:100px;padding:1px 5px;margin-left:4px}
.irc-pane{display:none}.irc-pane.active{display:block}

/* ── Similar cases list ── */
.similar-cases{padding:4px 0}
.case-item{display:flex;align-items:center;gap:12px;padding:11px 16px;border-bottom:1px solid rgba(48,54,61,.5);transition:background .12s}
.case-item:last-child{border-bottom:none}
.case-item:hover{background:rgba(255,255,255,.02)}
.case-rank{width:26px;height:26px;border-radius:6px;background:var(--violet-bg);border:1px solid rgba(124,58,237,.25);display:flex;align-items:center;justify-content:center;font-size:11px;font-family:var(--mono);font-weight:700;color:var(--violet-light);flex-shrink:0}
.case-body{flex:1;min-width:0;display:flex;flex-direction:column;gap:5px}
.case-top{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.case-acc{font-size:12px;font-family:var(--mono);font-weight:600;color:var(--teal);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px}
.sim-wrap{display:flex;align-items:center;gap:7px;flex:1;min-width:120px}
.sim-bar{flex:1;height:4px;background:var(--bg4);border-radius:2px;overflow:hidden;max-width:140px}
.sim-fill{height:4px;border-radius:2px;transition:width .6s cubic-bezier(.4,0,.2,1)}
.sim-fill.high{background:var(--green)}
.sim-fill.med{background:var(--teal)}
.sim-fill.low{background:var(--amber)}
.sim-fill.vlow{background:var(--text3)}
.case-sim-pct{font-size:10px;font-family:var(--mono);font-weight:600;white-space:nowrap}
.case-sim-pct.high{color:var(--green)}.case-sim-pct.med{color:var(--teal)}.case-sim-pct.low{color:var(--amber)}.case-sim-pct.vlow{color:var(--text3)}
.case-chips{display:flex;flex-wrap:wrap;gap:4px}
.case-chip{font-size:10px;font-family:var(--mono);background:var(--bg4);border:1px solid var(--border);border-radius:4px;padding:2px 7px;color:var(--text2);white-space:nowrap;line-height:1.6}
.case-chip.mod{color:var(--accent);border-color:rgba(88,166,255,.3);background:rgba(88,166,255,.06)}
.case-rpt{font-size:11px;color:var(--report);line-height:1.55;font-style:italic;border-left:2px solid rgba(240,136,62,.3);padding-left:8px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.no-results{padding:28px 16px;text-align:center;font-size:12px;color:var(--text3);font-family:var(--mono);line-height:1.8}

/* ── Frame strip (inside each accession group) ── */
.frame-strip{display:flex;gap:6px;overflow-x:auto;padding:6px 4px 4px;background:var(--bg4);border-radius:8px;margin-top:8px}
.frame-strip::-webkit-scrollbar{height:4px}
.frame-strip::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
.frame-thumb{
  width:72px;height:72px;object-fit:cover;border-radius:6px;
  border:1px solid var(--border2);cursor:pointer;flex-shrink:0;
  transition:transform .15s,border-color .15s,box-shadow .15s;
  background:#000;display:block;
}
.frame-thumb:hover{
  transform:scale(1.1);
  border-color:var(--violet-light);
  box-shadow:0 0 10px rgba(124,58,237,.5);
  z-index:1;position:relative;
}
.frame-count-badge{
  font-size:9px;font-family:var(--mono);color:var(--text3);
  background:var(--bg4);padding:2px 7px;border-radius:4px;
}
/* ── SOP group within an accession ── */
.sop-group{display:flex;flex-direction:column;gap:4px;margin-top:8px}
.sop-header{display:flex;align-items:center;gap:8px}
.sop-uid{font-size:10px;font-family:var(--mono);color:var(--violet-light);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:420px;
  background:var(--violet-bg);border:1px solid rgba(124,58,237,.25);
  border-radius:4px;padding:2px 8px;cursor:default;flex:1;}
.sop-sim{font-size:9px;font-family:var(--mono);font-weight:600;flex-shrink:0}

/* ── Case thumbnail ── */
.case-thumb-wrap{width:88px;height:88px;flex-shrink:0;border-radius:8px;overflow:hidden;border:1px solid var(--border2);background:#000;cursor:zoom-in;position:relative}
.case-thumb-wrap img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .2s}
.case-thumb-wrap:hover img{transform:scale(1.06)}
.case-thumb-wrap .thumb-missing{width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:var(--text3);font-size:9px;font-family:var(--mono);text-align:center;padding:4px;line-height:1.4}

/* ── Lightbox ── */
#lightbox{position:fixed;inset:0;z-index:9500;background:rgba(0,0,0,.92);display:none;align-items:center;justify-content:center;flex-direction:column;gap:12px;padding:20px;animation:fadeIn .15s ease}
#lightbox.visible{display:flex}
#lightboxImg{max-width:90vw;max-height:80vh;object-fit:contain;border-radius:8px;box-shadow:0 0 60px rgba(0,0,0,.9)}
#lightboxMeta{font-family:var(--mono);font-size:11px;color:rgba(255,255,255,.5);text-align:center;line-height:1.7}
.lb-close{position:absolute;top:16px;right:20px;background:none;border:none;color:rgba(255,255,255,.6);font-size:30px;cursor:pointer;line-height:1;padding:4px;transition:color .15s}
.lb-close:hover{color:#fff}

/* ── Image attach & preview ── */
.attach-btn{width:36px;height:36px;border-radius:8px;border:1px solid var(--border);background:var(--bg3);cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .2s;color:var(--text3)}
.attach-btn:hover{border-color:var(--violet);color:var(--violet-light);background:var(--violet-bg)}
.attach-btn.active{border-color:var(--violet);color:var(--violet-light);background:var(--violet-bg);box-shadow:0 0 0 3px rgba(124,58,237,.15)}
.attach-btn svg{width:16px;height:16px;stroke:currentColor;fill:none;stroke-width:2}
.img-preview-bar{padding:6px 20px 0;display:none;align-items:center;gap:10px;animation:fadeUp .2s ease}
.img-preview-thumb{width:44px;height:44px;object-fit:cover;border-radius:8px;border:1px solid var(--border2);flex-shrink:0}
.img-preview-info{flex:1;min-width:0;display:flex;flex-direction:column;gap:2px}
.img-preview-label{font-size:9px;font-family:var(--mono);color:var(--violet-light);text-transform:uppercase;letter-spacing:.5px}
.img-preview-name{font-size:11px;font-family:var(--mono);color:var(--text2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.img-clear-btn{background:none;border:none;color:var(--text3);cursor:pointer;padding:3px;line-height:0;border-radius:4px;transition:color .15s;flex-shrink:0}
.img-clear-btn:hover{color:var(--red)}
.img-clear-btn svg{width:14px;height:14px;stroke:currentColor;fill:none;stroke-width:2.5}

/* ── Example chips ── */
.chips-wrap{padding:4px 20px 8px;display:flex;flex-direction:column;gap:4px}
.chip-group{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.chip-group-label{font-size:9px;font-family:var(--mono);color:var(--text3);text-transform:uppercase;letter-spacing:.6px;padding-right:2px;white-space:nowrap}
.chip{padding:5px 12px;background:var(--bg3);border:1px solid var(--border);border-radius:100px;font-size:11.5px;font-family:var(--mono);color:var(--text2);cursor:pointer;transition:all .15s;user-select:none}
.chip:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-bg)}
.chip.report{border-color:rgba(240,136,62,.3);color:var(--report)}
.chip.report:hover{border-color:var(--report);background:var(--report-bg)}
.chip.image{border-color:rgba(124,58,237,.3);color:var(--violet-light)}
.chip.image:hover{border-color:var(--violet-light);background:var(--violet-bg)}

/* ── Thinking ── */
.thinking{display:flex;align-items:center;gap:8px;padding:10px 14px;background:var(--bg3);border:1px solid var(--border);border-radius:10px;font-size:12px;color:var(--text2);font-family:var(--mono)}
.dots{display:flex;gap:3px}
.dot{width:4px;height:4px;border-radius:50%;background:var(--accent);animation:blink 1.4s ease-in-out infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.2;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}

/* ── Empty state ── */
.empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;text-align:center;padding:40px}
.empty-icon{width:56px;height:56px;background:var(--bg3);border:1px solid var(--border);border-radius:14px;display:flex;align-items:center;justify-content:center}
.empty-title{font-size:20px;font-weight:800}
.empty-sub{font-size:13px;color:var(--text2);max-width:340px;line-height:1.6}

/* ── Input bar ── */
.input-bar{padding:14px 20px;border-top:1px solid var(--border);background:var(--bg2);display:flex;gap:8px;align-items:flex-end}
.input-wrap{flex:1;background:var(--bg3);border:1px solid var(--border);border-radius:10px;display:flex;align-items:flex-end;padding:0 12px;transition:border-color .2s}
.input-wrap:focus-within{border-color:var(--accent)}
.input-wrap.img-mode{border-color:rgba(124,58,237,.5)}
.input-wrap.img-mode:focus-within{border-color:var(--violet-light)}
textarea.nl-input{flex:1;background:none;border:none;outline:none;color:var(--text);font-family:var(--sans);font-size:13.5px;padding:11px 0;resize:none;line-height:1.55;max-height:130px;overflow-y:auto}
textarea.nl-input::placeholder{color:var(--text3)}
.send-btn{width:40px;height:40px;border-radius:9px;border:none;background:var(--accent2);cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background .2s}
.send-btn:hover:not(:disabled){background:var(--accent)}
.send-btn:disabled{background:var(--bg4);cursor:not-allowed}
.send-btn.img-mode{background:linear-gradient(135deg,var(--violet),#4338ca)}
.send-btn.img-mode:hover:not(:disabled){background:linear-gradient(135deg,#8b5cf6,#6366f1)}
.send-btn svg{width:17px;height:17px;stroke:#fff;fill:none;stroke-width:2.2}
.stop-btn{width:40px;height:40px;border-radius:9px;border:none;background:var(--red);cursor:pointer;display:none;align-items:center;justify-content:center;flex-shrink:0;transition:background .2s}
.stop-btn:hover{background:#c0392b}
.stop-btn svg{width:13px;height:13px;fill:#fff;stroke:none}

/* ── Sidebar ── */
.sidebar{grid-area:side;background:var(--bg2);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}
.sb-hdr{padding:12px 16px;border-bottom:1px solid var(--border);font-size:10px;font-weight:600;font-family:var(--mono);color:var(--text3);text-transform:uppercase;letter-spacing:.7px}
.sb-sec{padding:12px 16px;border-bottom:1px solid var(--border)}
.sb-label{font-size:10px;color:var(--text3);font-family:var(--mono);margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.stat{background:var(--bg3);border-radius:7px;padding:9px 10px;border:1px solid var(--border)}
.stat-val{font-size:20px;font-weight:800;font-family:var(--mono);color:var(--accent);letter-spacing:-.5px}
.stat-val.report{color:var(--report)}
.stat-val.chroma{color:var(--violet-light)}
.stat-lab{font-size:9px;color:var(--text2);margin-top:1px;text-transform:uppercase;letter-spacing:.6px}
.db-path{font-family:var(--mono);font-size:9.5px;color:var(--text3);word-break:break-all;line-height:1.5}
.opt-row{display:flex;align-items:center;justify-content:space-between;padding:6px 0}
.opt-lab{font-size:11.5px;color:var(--text2);font-family:var(--mono)}
.toggle{position:relative;width:34px;height:18px;flex-shrink:0}
.toggle input{opacity:0;width:0;height:0}
.tslider{position:absolute;inset:0;background:var(--border2);border-radius:9px;cursor:pointer;transition:.2s}
.tslider::before{content:'';position:absolute;width:14px;height:14px;left:2px;top:2px;background:#fff;border-radius:50%;transition:.2s}
.toggle input:checked~.tslider{background:var(--accent2)}
.toggle input:checked~.tslider::before{transform:translateX(16px)}
.rpt-coverage{margin-top:8px;height:6px;background:var(--bg4);border-radius:3px;overflow:hidden}
.rpt-coverage-bar{height:6px;background:var(--report);border-radius:3px;transition:width .5s ease}
.rpt-coverage-label{font-size:9px;font-family:var(--mono);color:var(--text3);margin-top:4px}
.hist-scroll{flex:1;overflow-y:auto;padding:8px}
.hist-scroll::-webkit-scrollbar{width:4px}.hist-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.hist-item{padding:8px 10px;border-radius:7px;cursor:pointer;transition:background .15s;border:1px solid transparent;margin-bottom:4px}
.hist-item:hover{background:var(--bg3);border-color:var(--border)}
.hist-q{font-family:var(--sans);font-weight:600;color:var(--text);font-size:11.5px;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hist-q.img-q{color:var(--violet-light)}
.hist-meta{font-family:var(--mono);font-size:9.5px;color:var(--text3)}
.modality-row{display:flex;align-items:center;gap:8px;font-size:11px;font-family:var(--mono);padding:3px 0;color:var(--text2)}
.mod-bar-wrap{flex:1;background:var(--bg4);border-radius:3px;height:5px;overflow:hidden}
.mod-bar{height:5px;background:var(--accent2);border-radius:3px;transition:width .5s ease}
.copy-btn{font-size:10px;font-family:var(--mono);background:none;border:1px solid var(--border);color:var(--text3);border-radius:4px;padding:2px 7px;cursor:pointer;transition:all .15s;margin-left:auto}
.copy-btn:hover{border-color:var(--accent);color:var(--accent)}

/* ── Drag-over highlight ── */
.messages.drag-over{outline:2px dashed var(--violet-light);outline-offset:-12px;background:rgba(124,58,237,.04)}

@keyframes fadeIn{from{opacity:0}to{opacity:1}}
</style>
</head>
<body>

<!-- LOGIN OVERLAY -->
<div id="loginOverlay">
  <div class="login-card" id="loginCard">
    <div class="login-logo">
      <svg viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>
    </div>
    <div class="login-title">AngioVision</div>
    <div class="login-sub">QUERY ENGINE · RESTRICTED ACCESS</div>
    <div id="loginError" class="login-error">Invalid username or password.</div>
    <div class="login-field">
      <label class="login-label" for="loginUser">Username</label>
      <input class="login-input" id="loginUser" type="text" placeholder="Enter username" autocomplete="username" spellcheck="false">
    </div>
    <div class="login-field">
      <label class="login-label" for="loginPass">Password</label>
      <div class="pw-wrap">
        <input class="login-input" id="loginPass" type="password" placeholder="Enter password" autocomplete="current-password">
        <button class="pw-toggle" id="pwToggle" type="button" onclick="togglePwVisibility()">
          <svg id="eyeOpen" viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
          <svg id="eyeOff" viewBox="0 0 24 24" style="display:none"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
        </button>
      </div>
    </div>
    <button class="login-btn" id="loginBtn" onclick="handleLogin()">Sign In</button>
    <div class="login-footer">
      Authorised access only &nbsp;·&nbsp; Session: <span id="sessionUser">—</span>
      <div class="attempt-bar"><div class="attempt-fill" id="attemptFill"></div></div>
    </div>
  </div>
</div>

<!-- MAIN APP -->
<div id="appWrapper">
<div class="layout">

  <header class="header">
    <div class="logo">
      <svg viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h10"/><circle cx="19" cy="17" r="3"/></svg>
    </div>
    <span class="hdr-title">AngioVision · Query Engine</span>
    <div class="hdr-sep"></div>
    <span class="hdr-sub">NL → SQL · Image RAG (RAD-DINO)</span>
    <div class="hdr-right">
      <div class="status-pill"><span class="status-dot"></span><span id="statusTxt">connecting…</span></div>
      <select class="model-select" id="modelSelect" onchange="onModelChange()">
        <option>qwen3:1.7b</option>
        <option>qwen3:8b</option>
        <option>qwen3:14b</option>
        <option>qwen3:32b</option>
        <option>qwen3:35b</option>
        <option>llama3.2:latest</option>
      </select>
      <button class="theme-toggle" id="themeToggleBtn" onclick="toggleTheme()" title="Toggle light / dark mode">
        <!-- moon icon (shown in dark mode) -->
        <svg id="iconMoon" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
        <!-- sun icon (shown in light mode) -->
        <svg id="iconSun" viewBox="0 0 24 24" style="display:none"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
      </button>
      <button class="logout-btn" onclick="handleLogout()">Logout</button>
    </div>
  </header>

  <main class="chat-col">
    <div class="messages" id="messages">
      <div class="empty" id="emptyState">
        <div class="empty-icon">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="1.8"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>
        </div>
        <div class="empty-title">Ask about your DICOM data</div>
        <div class="empty-sub">Natural language → SQL for metadata &amp; reports. Upload an angiography image to find visually similar cases via ChromaDB vector search with RAD-DINO embeddings.</div>
      </div>
    </div>

    <div class="chips-wrap">
      <div class="chip-group">
        <span class="chip-group-label">DICOM</span>
        <span class="chip" onclick="fillQ('How many unique patients are in the database?')">Patient count</span>
        <span class="chip" onclick="fillQ('How many studies, series, and instances do we have in total?')">Overview</span>
        <span class="chip" onclick="fillQ('List all DSA series with more than 20 frames, showing accession number and source path.')">DSA &gt; 20 frames</span>
        <span class="chip" onclick="fillQ('What modalities are present and how many instances does each have?')">Modalities</span>
        <span class="chip" onclick="fillQ('How many studies were performed each year? Order by year.')">Studies/year</span>
        <span class="chip" onclick="fillQ('Which instances have the highest dose product? Show top 10.')">Top dose</span>
      </div>
      <div class="chip-group">
        <span class="chip-group-label">Reports</span>
        <span class="chip report" onclick="fillQ('How many accession numbers have a linked radiology report?')">Coverage</span>
        <span class="chip report" onclick="fillQ('Show 5 example radiology reports with their accession number and study date.')">Sample reports</span>
        <span class="chip report" onclick="fillQ('Find all reports electronically signed by [Doctor Name].')">By doctor</span>
        <span class="chip report" onclick="fillQ('Find all studies whose report mentions stenosis.')">Stenosis</span>
        <span class="chip report" onclick="fillQ('What is the average DICOM dose product for studies signed by a specific doctor?')">Dose by doctor</span>
      </div>
      <div class="chip-group">
        <span class="chip-group-label">Image RAG</span>
        <span class="chip image" onclick="promptImageUpload('Show me the 5 most similar angiography sequences to this image.')">Upload → find similar</span>
        <span class="chip image" onclick="promptImageUpload('Find cases similar to this image and show their radiology reports.')">Upload → with reports</span>
        <span class="chip image" onclick="promptImageUpload('What accession numbers are most visually similar to this DSA frame?')">Upload → accessions</span>
      </div>
    </div>

    <!-- Image preview bar (shown when image is attached) -->
    <div class="img-preview-bar" id="imgPreviewBar">
      <img class="img-preview-thumb" id="imgPreviewThumb" src="" alt="">
      <div class="img-preview-info">
        <div class="img-preview-label">Image attached · RAD-DINO vector search enabled</div>
        <div class="img-preview-name" id="imgPreviewName">—</div>
      </div>
      <button class="img-clear-btn" onclick="clearImage()" title="Remove image">
        <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>

    <div class="input-bar">
      <div class="input-wrap" id="inputWrap">
        <textarea class="nl-input" id="nlInput" rows="1"
          placeholder="Ask about DICOM metadata, radiology reports, or upload an image…"></textarea>
      </div>
      <!-- Hidden file input -->
      <input type="file" id="imageFileInput" accept="image/*" style="display:none" onchange="onImageSelected(event)">
      <!-- Attach button -->
      <button class="attach-btn" id="attachBtn" onclick="triggerImagePick()"
              title="Attach angiography image for visual similarity search">
        <svg viewBox="0 0 24 24"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
      </button>
      <!-- Stop button (visible only while loading) -->
      <button class="stop-btn" id="stopBtn" onclick="stopQuery()" title="Stop">
        <svg viewBox="0 0 10 10"><rect x="0" y="0" width="10" height="10" rx="1.5"/></svg>
      </button>
      <!-- Send button -->
      <button class="send-btn" id="sendBtn" onclick="handleSend()" title="Send (Enter)">
        <svg viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
      </button>
    </div>
  </main>

  <aside class="sidebar">
    <div class="sb-hdr">Database</div>
    <div class="sb-sec">
      <div class="sb-label">Path</div>
      <div class="db-path" id="dbPath">—</div>
    </div>
    <div class="sb-sec">
      <div class="sb-label">DICOM Instances</div>
      <div class="stat-grid">
        <div class="stat"><div class="stat-val" id="sInst">—</div><div class="stat-lab">Instances</div></div>
        <div class="stat"><div class="stat-val" id="sPat">—</div><div class="stat-lab">Patients</div></div>
        <div class="stat"><div class="stat-val" id="sStu">—</div><div class="stat-lab">Studies</div></div>
        <div class="stat"><div class="stat-val" id="sSer">—</div><div class="stat-lab">Series</div></div>
      </div>
    </div>
    <div class="sb-sec">
      <div class="sb-label">Radiology Reports</div>
      <div class="stat-grid">
        <div class="stat"><div class="stat-val report" id="sRptTotal">—</div><div class="stat-lab">Reports</div></div>
        <div class="stat"><div class="stat-val report" id="sRptLinked">—</div><div class="stat-lab">Linked</div></div>
      </div>
      <div class="rpt-coverage" id="rptCoverageWrap" style="display:none">
        <div class="rpt-coverage-bar" id="rptCoverageBar" style="width:0%"></div>
      </div>
      <div class="rpt-coverage-label" id="rptCoverageLabel"></div>
    </div>
    <div class="sb-sec" id="chromaSec">
      <div class="sb-label">ChromaDB · RAD-DINO</div>
      <div class="stat-grid">
        <div class="stat"><div class="stat-val chroma" id="sChromaFrames">—</div><div class="stat-lab">Frames</div></div>
        <div class="stat"><div class="stat-val chroma" id="sChromaSeqs">—</div><div class="stat-lab">Sequences</div></div>
      </div>
      <div class="rpt-coverage-label" id="chromaStatusLabel" style="margin-top:6px"></div>
    </div>
    <div class="sb-sec" id="modalitySec" style="display:none">
      <div class="sb-label">Modalities</div>
      <div id="modalityBars"></div>
    </div>
    <div class="sb-sec">
      <div class="sb-label">Options</div>
      <div class="opt-row"><span class="opt-lab">Thinking mode</span><label class="toggle"><input type="checkbox" id="thinkToggle" checked><span class="tslider"></span></label></div>
      <div class="opt-row"><span class="opt-lab">Auto-scroll</span><label class="toggle"><input type="checkbox" id="autoScrollToggle" checked><span class="tslider"></span></label></div>
    </div>
    <div class="sb-hdr" style="border-top:1px solid var(--border)">History</div>
    <div class="hist-scroll" id="histScroll"></div>
  </aside>

</div>
</div>



<script>
// ════════════════════════════════════════════════
// LOGIN
// ════════════════════════════════════════════════
// Credentials are injected server-side from environment variables.
// Set ANGIOVISION_USERS as a JSON array, e.g.:
//   export ANGIOVISION_USERS='[{"user":"admin","pass":"changeme"}]'
const CREDENTIALS = __CREDENTIALS_PLACEHOLDER__;
const MAX_ATTEMPTS = 5;
let loginAttempts = 0, lockoutTimer = null;

function togglePwVisibility(){
  const inp=document.getElementById('loginPass');
  const open=document.getElementById('eyeOpen'),off=document.getElementById('eyeOff');
  if(inp.type==='password'){inp.type='text';open.style.display='none';off.style.display='block';}
  else{inp.type='password';open.style.display='block';off.style.display='none';}
}

function handleLogin(){
  const u=document.getElementById('loginUser').value.trim();
  const p=document.getElementById('loginPass').value;
  const btn=document.getElementById('loginBtn'),err=document.getElementById('loginError');
  const card=document.getElementById('loginCard'),fill=document.getElementById('attemptFill');
  if(btn.disabled) return;
  const match=CREDENTIALS.find(c=>c.user===u&&c.pass===p);
  if(match){
    err.classList.remove('visible');
    document.getElementById('sessionUser').textContent=u;
    document.getElementById('loginOverlay').classList.add('hidden');
    document.getElementById('appWrapper').classList.add('visible');
    loadStats(); loadChromaStats();
    document.getElementById('nlInput').focus();
  } else {
    loginAttempts++;
    fill.style.width=Math.min(100,(loginAttempts/MAX_ATTEMPTS)*100)+'%';
    err.classList.remove('visible'); void err.offsetWidth; err.classList.add('visible');
    card.classList.remove('shake'); void card.offsetWidth; card.classList.add('shake');
    document.getElementById('loginPass').value='';
    document.getElementById('loginPass').focus();
    if(loginAttempts>=MAX_ATTEMPTS){
      btn.disabled=true; let secs=30;
      err.textContent=`Too many attempts. Try again in ${secs}s.`;
      lockoutTimer=setInterval(()=>{secs--;err.textContent=`Too many attempts. Try again in ${secs}s.`;
        if(secs<=0){clearInterval(lockoutTimer);loginAttempts=0;fill.style.width='0%';
          btn.disabled=false;err.textContent='Invalid username or password.';
          err.classList.remove('visible');document.getElementById('loginUser').focus();}},1000);
    }
  }
}
function handleLogout(){
  document.getElementById('appWrapper').classList.remove('visible');
  document.getElementById('loginOverlay').classList.remove('hidden');
  document.getElementById('loginUser').value='';
  document.getElementById('loginPass').value='';
  document.getElementById('loginError').classList.remove('visible');
  document.getElementById('sessionUser').textContent='—';
  document.getElementById('loginUser').focus();
}
document.addEventListener('DOMContentLoaded',()=>{
  ['loginUser','loginPass'].forEach(id=>{
    document.getElementById(id).addEventListener('keydown',e=>{if(e.key==='Enter')handleLogin();});
  });
  document.getElementById('loginUser').focus();
  setupDragDrop();
});

// ════════════════════════════════════════════════
// MAIN APP STATE
// ════════════════════════════════════════════════
const API='';
let chatHistory=[], isLoading=false;
let attachedImage=null;  // { dataUrl, base64, filename }
let _abortController = null;

const inp     = document.getElementById('nlInput');
const sendBtn = document.getElementById('sendBtn');
const stopBtn = document.getElementById('stopBtn');

function setLoading(on){
  isLoading = on;
  sendBtn.disabled = on;
  sendBtn.style.display = on ? 'none'  : 'flex';
  stopBtn.style.display = on ? 'flex'  : 'none';
}

function stopQuery(){
  if(_abortController) _abortController.abort();
}

// Auto-resize textarea
inp.addEventListener('input',()=>{inp.style.height='auto';inp.style.height=Math.min(inp.scrollHeight,130)+'px';});
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();handleSend();}});

function fillQ(q){inp.value=q;inp.dispatchEvent(new Event('input'));inp.focus();}

function promptImageUpload(q){
  inp.value=q;inp.dispatchEvent(new Event('input'));
  triggerImagePick();
}

// ════════════════════════════════════════════════
// IMAGE ATTACH
// ════════════════════════════════════════════════
function triggerImagePick(){
  document.getElementById('imageFileInput').click();
}

function onImageSelected(e){
  const file=e.target.files[0];
  if(!file) return;
  const reader=new FileReader();
  reader.onload=ev=>{
    const dataUrl=ev.target.result;
    attachedImage={dataUrl, base64: dataUrl.split(',')[1], filename: file.name};
    document.getElementById('imgPreviewThumb').src=dataUrl;
    document.getElementById('imgPreviewName').textContent=file.name;
    document.getElementById('imgPreviewBar').style.display='flex';
    document.getElementById('attachBtn').classList.add('active');
    document.getElementById('inputWrap').classList.add('img-mode');
    sendBtn.classList.add('img-mode');
    inp.placeholder='Ask about this image — e.g. "show me similar cases"…';
    inp.focus();
  };
  reader.readAsDataURL(file);
  e.target.value='';
}

function clearImage(){
  attachedImage=null;
  document.getElementById('imgPreviewBar').style.display='none';
  document.getElementById('imgPreviewThumb').src='';
  document.getElementById('attachBtn').classList.remove('active');
  document.getElementById('inputWrap').classList.remove('img-mode');
  sendBtn.classList.remove('img-mode');
  inp.placeholder='Ask about DICOM metadata, radiology reports, or upload an image…';
}

// Drag-and-drop onto the messages area
function setupDragDrop(){
  const msgs=document.getElementById('messages');
  msgs.addEventListener('dragover',e=>{e.preventDefault();msgs.classList.add('drag-over');});
  msgs.addEventListener('dragleave',()=>msgs.classList.remove('drag-over'));
  msgs.addEventListener('drop',e=>{
    e.preventDefault(); msgs.classList.remove('drag-over');
    const file=e.dataTransfer.files[0];
    if(file && file.type.startsWith('image/')){
      const fakeEvt={target:{files:[file]}};
      onImageSelected(fakeEvt);
    }
  });
}

// ════════════════════════════════════════════════
// THEME TOGGLE
// ════════════════════════════════════════════════
(function(){
  // Restore preference on load
  const saved = localStorage.getItem('av-theme');
  if(saved === 'light'){
    document.documentElement.setAttribute('data-theme','light');
    document.getElementById('iconMoon').style.display='none';
    document.getElementById('iconSun').style.display='block';
  }
})();

function toggleTheme(){
  const root   = document.documentElement;
  const isLight = root.getAttribute('data-theme') === 'light';
  const moon   = document.getElementById('iconMoon');
  const sun    = document.getElementById('iconSun');
  if(isLight){
    root.removeAttribute('data-theme');
    moon.style.display='block';
    sun.style.display='none';
    localStorage.setItem('av-theme','dark');
  } else {
    root.setAttribute('data-theme','light');
    moon.style.display='none';
    sun.style.display='block';
    localStorage.setItem('av-theme','light');
  }
}

// ════════════════════════════════════════════════
// MODEL / STATUS
// ════════════════════════════════════════════════
function onModelChange(){
  const m=document.getElementById('modelSelect').value;
  fetch(`${API}/api/model`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:m})})
    .then(r=>r.json()).then(d=>{if(d.ok)setStatus('ready',m);});
}

function setStatus(state,label){
  const dot=document.querySelector('.status-dot'),txt=document.getElementById('statusTxt');
  const colors={ready:'#3fb950',error:'#f85149',busy:'#d29922'};
  dot.style.background=colors[state]||colors.ready;
  dot.style.boxShadow=`0 0 5px ${colors[state]||colors.ready}`;
  txt.textContent=label||state;
}

// ════════════════════════════════════════════════
// STATS
// ════════════════════════════════════════════════
async function loadStats(){
  try{
    const r=await fetch(`${API}/api/stats`);
    const d=await r.json();
    if(d.error){setStatus('error','db error');return;}
    document.getElementById('sInst').textContent=fmtNum(d.instances);
    document.getElementById('sPat').textContent=fmtNum(d.patients);
    document.getElementById('sStu').textContent=fmtNum(d.studies);
    document.getElementById('sSer').textContent=fmtNum(d.series);
    document.getElementById('dbPath').textContent=d.db_path||'—';
    document.getElementById('sRptTotal').textContent=fmtNum(d.rpt_total);
    document.getElementById('sRptLinked').textContent=fmtNum(d.rpt_linked);
    if(d.rpt_total>0&&d.studies>0){
      const pct=Math.min(100,Math.round(d.rpt_linked/d.studies*100));
      document.getElementById('rptCoverageWrap').style.display='block';
      document.getElementById('rptCoverageBar').style.width=pct+'%';
      document.getElementById('rptCoverageLabel').textContent=`${pct}% of studies linked  ·  ${fmtNum(d.rpt_unlinked)} missing`;
    }
    if(d.modalities&&d.modalities.length){
      const max=Math.max(...d.modalities.map(m=>m.count));
      document.getElementById('modalityBars').innerHTML=d.modalities.map(m=>`
        <div class="modality-row">
          <span style="min-width:28px;color:var(--accent)">${m.modality}</span>
          <div class="mod-bar-wrap"><div class="mod-bar" style="width:${Math.round(m.count/max*100)}%"></div></div>
          <span style="min-width:40px;text-align:right">${fmtNum(m.count)}</span>
        </div>`).join('');
      document.getElementById('modalitySec').style.display='block';
    }
    setStatus('ready','ready');
  }catch(e){setStatus('error','offline');}
}

async function loadChromaStats(){
  try{
    const r=await fetch(`${API}/api/chroma-stats`);
    const d=await r.json();
    if(d.available){
      document.getElementById('sChromaFrames').textContent=fmtNum(d.count);
      document.getElementById('sChromaSeqs').textContent=d.sequences!=null?fmtNum(d.sequences):'—';
      document.getElementById('chromaStatusLabel').textContent=`Collection: ${d.collection}`;
    } else {
      document.getElementById('sChromaFrames').textContent='—';
      document.getElementById('sChromaSeqs').textContent='—';
      document.getElementById('chromaStatusLabel').textContent=d.error?'Unavailable':'Not ingested';
    }
  }catch(e){
    document.getElementById('chromaStatusLabel').textContent='Unavailable';
  }
}

function fmtNum(n){if(n===undefined||n===null||n==='—')return '—';return Number(n).toLocaleString();}

// ════════════════════════════════════════════════
// SQL SYNTAX HIGHLIGHT
// ════════════════════════════════════════════════
function colorSQL(sql){
  const kws=/\b(SELECT|FROM|WHERE|AND|OR|NOT|IN|LIKE|BETWEEN|ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT|OFFSET|JOIN|LEFT|RIGHT|INNER|OUTER|ON|AS|DISTINCT|COUNT|SUM|AVG|MAX|MIN|CAST|LOWER|UPPER|NULL|IS\s+NOT|IS|CASE|WHEN|THEN|ELSE|END|WITH|UNION|ALL|EXISTS|USING|SUBSTR|INSTR|TRIM|COALESCE|IFNULL)\b/gi;
  return sql.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/'[^']*'/g,s=>`<span class="str">${s}</span>`)
    .replace(/--[^\n]*/g,s=>`<span class="cmt">${s}</span>`)
    .replace(/\b(\d+\.?\d*)\b/g,s=>`<span class="nm">${s}</span>`)
    .replace(kws,s=>`<span class="kw">${s}</span>`);
}

// ════════════════════════════════════════════════
// TABLE RENDERER
// ════════════════════════════════════════════════
function renderTable(rows){
  if(!rows||!rows.length)return '<div style="padding:16px;font-size:12px;color:var(--text3);font-family:var(--mono)">No rows returned.</div>';
  const cols=Object.keys(rows[0]);
  const thead=`<thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead>`;
  const tbody=`<tbody>${rows.map(r=>`<tr>${cols.map(c=>{
    const v=r[c];
    if(v===null||v===undefined)return `<td><span class="v-null">null</span></td>`;
    const s=String(v);
    if(c==='radrpt'||c==='radrpt_excerpt')return `<td><span class="v-report" title="${esc(s)}">${esc(s.length>80?s.slice(0,78)+'…':s)}</span></td>`;
    if(s.toLowerCase().endsWith('.dcm')&&s.startsWith('/')){
      const thumbUrl=API+'/api/thumbnail?path='+encodeURIComponent(s)+'&frame=0';
      const frameUrl=API+'/api/frame?path='+encodeURIComponent(s)+'&frame=0';
      const shortPath=s.length>40?'…'+s.slice(-38):s;
      return `<td><div class="dcm-cell"><img src="${thumbUrl}" class="tbl-thumb" loading="lazy" onclick="openLightbox('${frameUrl}','${esc(s)}')" onerror="this.outerHTML='<span class=\\'tbl-thumb-err\\'>no px</span>'"/><span class="v-path" title="${esc(s)}">${esc(shortPath)}</span></div></td>`;
    }
    if(s.startsWith('/'))return `<td><span class="v-path" title="${esc(s)}">${esc(s.length>50?'…'+s.slice(-48):s)}</span></td>`;
    if(/^\d{8}$/.test(s)&&parseInt(s)>19000101)return `<td><span class="v-num">${s.slice(0,4)}-${s.slice(4,6)}-${s.slice(6,8)}</span></td>`;
    if(s.length>32&&s.includes('.'))return `<td><span class="v-id" title="${esc(s)}">${esc(s.slice(0,28)+'…')}</span></td>`;
    if(!isNaN(s)&&s.trim()!=='')return `<td><span class="v-num">${esc(s)}</span></td>`;
    return `<td>${esc(s.length>60?s.slice(0,58)+'…':s)}</td>`;
  }).join('')}</tr>`).join('')}</tbody>`;
  return `<div class="tbl-scroll"><table class="data-table">${thead}${tbody}</table></div>`;
}

function renderSteps(steps){
  if(!steps||!steps.length)return '<div class="agent-steps"><div class="agent-steps-empty">No tool calls recorded — the agent may have failed before calling sql_query. Check the error message in the Answer tab.</div></div>';
  let html='<div class="agent-steps">';
  steps.forEach(s=>{
    const cls=s.error?'step-err':'step-ok';
    const badge=s.error?'✗ error':'✓ ok';
    html+=`<div class="agent-step-item ${cls}">`;
    html+=`<div class="step-hdr"><span class="step-badge">${badge}</span><span class="step-meta">Step ${s.step} / ${s.max_steps}</span></div>`;
    html+=`<div class="step-sql">${esc(s.sql)}</div>`;
    if(s.error){
      html+=`<div class="step-error-msg">${esc(s.error)}</div>`;
    } else {
      html+=`<div class="step-result">${s.row_count} row${s.row_count!==1?'s':''} returned</div>`;
    }
    html+=`</div>`;
  });
  html+='</div>';
  return html;
}

function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}

// ════════════════════════════════════════════════
// MESSAGE HELPERS
// ════════════════════════════════════════════════
function addMsg(cls,html){
  const es=document.getElementById('emptyState');
  if(es)es.remove();
  const msgs=document.getElementById('messages');
  const d=document.createElement('div');
  d.className=`msg ${cls}`;d.innerHTML=html;
  msgs.appendChild(d);
  if(document.getElementById('autoScrollToggle').checked)msgs.scrollTop=msgs.scrollHeight;
  return d;
}

function addHistory(q,count,elapsed,isImage=false){
  chatHistory.unshift({q,count,elapsed,ts:new Date().toLocaleTimeString(),isImage});
  if(chatHistory.length>30)chatHistory.pop();
  document.getElementById('histScroll').innerHTML=chatHistory.map(h=>`
    <div class="hist-item" onclick="fillQ(${JSON.stringify(h.q)})">
      <div class="hist-q${h.isImage?' img-q':''}">${h.isImage?'📷 ':''}${esc(h.q)}</div>
      <div class="hist-meta">${h.ts} · ${h.count} result${h.count!==1?'s':''} · ${(h.elapsed/1000).toFixed(1)}s</div>
    </div>`).join('');
}

// ════════════════════════════════════════════════
// MAIN SEND ROUTER
// ════════════════════════════════════════════════
async function handleSend(){
  if(isLoading) return;
  const q=inp.value.trim();
  if(attachedImage){
    await handleImageSend(q||'Show me the 5 most visually similar cases to this image.');
  } else {
    if(!q) return;
    await handleTextSend(q);
  }
}

// ════════════════════════════════════════════════
// TEXT QUERY (NL → SQL → synthesis)
// ════════════════════════════════════════════════
async function handleTextSend(question){
  _abortController = new AbortController();
  setLoading(true);
  inp.value=''; inp.style.height='auto';
  addMsg('msg-user',`<div class="bubble bubble-user">${esc(question)}</div>`);
  const thinkEl=addMsg('msg-bot',`<div class="thinking"><div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div><span id="thinkTxt">Initializing agent…</span></div>`);
  const thinkTxt=thinkEl.querySelector('#thinkTxt');
  setStatus('busy','thinking…');
  const think=document.getElementById('thinkToggle').checked;
  const model=document.getElementById('modelSelect').value;
  let sql='',rows=[],rowCount=0,answer='',elapsed=0;
  let agentSteps=[];   // ← NEW: track every tool call
  let allSql=[];       // ← NEW: track every SQL the agent tried
  try{
    const resp=await fetch(`${API}/api/query`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,think,model}),signal:_abortController.signal});

    /* ── NEW: catch non-SSE error responses (503, 400, etc.) ── */
    if(!resp.ok){
      let errMsg=`Server returned HTTP ${resp.status}`;
      try{const errData=await resp.json();errMsg=errData.error||errMsg;}catch{}
      thinkEl.remove();
      addMsg('msg-bot',`<div class="bubble-error">Error: ${esc(errMsg)}</div>`);
      setLoading(false);setStatus('error','error');return;
    }

    const reader=resp.body.getReader(),decoder=new TextDecoder();let buf='';
    while(true){
      const{done,value}=await reader.read();if(done)break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\n\n');buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        let evt;try{evt=JSON.parse(line.slice(6));}catch{continue;}
        switch(evt.event){

          /* ── NEW: agent lifecycle events ── */
          case 'agent_start':
            thinkTxt.textContent=`Agent started (${evt.model}, max ${evt.max_steps} steps)…`;
            break;
          case 'agent_step':
            agentSteps.push(evt);
            if(evt.error){
              thinkTxt.textContent=`Step ${evt.step}/${evt.max_steps}: SQL error → retrying…`;
            } else {
              thinkTxt.textContent=`Step ${evt.step}/${evt.max_steps}: ${evt.row_count} row${evt.row_count!==1?'s':''} returned`;
            }
            break;

          /* ── Original events ── */
          case 'sql_start':    thinkTxt.textContent='Agent is thinking…';break;
          case 'sql_done':     sql=evt.sql;allSql.push(evt.sql);thinkTxt.textContent='Executing query…';break;
          case 'sql_repaired': sql=evt.sql;allSql.push(evt.sql);thinkTxt.textContent=`Refining SQL (attempt ${allSql.length})…`;break;
          case 'exec_start':   thinkTxt.textContent='Running query…';break;
          case 'exec_done':    rows=evt.rows;rowCount=evt.row_count;thinkTxt.textContent='Waiting for agent…';break;
          case 'synth_start':  thinkTxt.textContent='Synthesizing answer…';break;
          case 'answer':       answer=evt.text;break;
          case 'done':         elapsed=evt.elapsed_ms;break;
          case 'error':
            thinkEl.remove();
            addMsg('msg-bot',`<div class="bubble-error">Error: ${esc(evt.message)}</div>`);
            setLoading(false);setStatus('error','error');return;
        }
      }
    }
    thinkEl.remove();
    const id='rc'+Date.now(),elapsedSec=(elapsed/1000).toFixed(1);
    let tabs='',panes='';
    tabs+=`<button class="rc-tab active" onclick="switchRcTab(this,'${id}-ans')">Answer</button>`;
    panes+=`<div class="rc-pane active" id="${id}-ans"><div class="answer-body">${esc(answer)}</div></div>`;
    tabs+=`<button class="rc-tab" onclick="switchRcTab(this,'${id}-sql')">SQL</button>`;
    panes+=`<div class="rc-pane" id="${id}-sql"><div class="sql-block">${colorSQL(sql)}</div></div>`;
    tabs+=`<button class="rc-tab" onclick="switchRcTab(this,'${id}-tbl')">Table <span style="font-size:9px;opacity:.7">${rowCount}</span></button>`;
    panes+=`<div class="rc-pane" id="${id}-tbl">${renderTable(rows)}${rowCount?`<div class="tbl-footer"><span>${rowCount} row${rowCount!==1?'s':''}</span><button class="copy-btn" onclick="copySQL('${id}')">copy SQL</button><span style="margin-left:auto">${elapsedSec}s</span></div>`:''}</div>`;

    /* ── NEW: Steps tab showing every tool call the agent made ── */
    tabs+=`<button class="rc-tab" onclick="switchRcTab(this,'${id}-steps')">Steps <span style="font-size:9px;opacity:.7">${agentSteps.length}</span></button>`;
    panes+=`<div class="rc-pane" id="${id}-steps">${renderSteps(agentSteps)}</div>`;

    addMsg('msg-bot',`<div class="result-card"><div class="rc-tabs">${tabs}<div class="rc-meta"><span>${elapsedSec}s</span></div></div>${panes}</div>`);
    addHistory(question,rowCount,elapsed,false);
    setStatus('ready','ready');
  }catch(e){
    if(e.name==='AbortError'){
      addMsg('msg-bot','<div class="bubble-error">Query stopped.</div>');
      setStatus('ready','ready');
    } else {
      thinkEl.remove();
      addMsg('msg-bot',`<div class="bubble-error">Request failed: ${esc(String(e))}</div>`);
      setStatus('error','error');
    }
  }
  setLoading(false);inp.focus();
}

// ════════════════════════════════════════════════
// IMAGE QUERY (ChromaDB vector search with RAD-DINO)
// ════════════════════════════════════════════════
async function handleImageSend(question){
  if(!attachedImage) return;
  const img=attachedImage;
  clearImage();
  _abortController = new AbortController();
  setLoading(true);
  inp.value=''; inp.style.height='auto';

  addMsg('msg-user',`
    <div class="bubble-with-img">
      <img src="${img.dataUrl}" class="bubble-img-preview" alt="uploaded image">
      <span class="bubble-img-caption">${esc(question)}</span>
    </div>`);

  const thinkEl=addMsg('msg-bot',`<div class="thinking"><div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div><span id="thinkTxt">Decoding image…</span></div>`);
  const thinkTxt=thinkEl.querySelector('#thinkTxt');
  setStatus('busy','searching…');

  let results=[],answer='',elapsed=0;
  const cardId='irc'+Date.now();

  try{
    const resp=await fetch(`${API}/api/image-query`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      signal:_abortController.signal,
      body:JSON.stringify({
        image:    img.base64,
        question: question,
        n_results: 5,
        think:    document.getElementById('thinkToggle').checked,
      })
    });
    const reader=resp.body.getReader(),decoder=new TextDecoder();let buf='';
    while(true){
      const{done,value}=await reader.read();if(done)break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\n\n');buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        let evt;try{evt=JSON.parse(line.slice(6));}catch{continue;}
        switch(evt.event){
          case 'decode_start':  thinkTxt.textContent='Decoding image…';break;
          case 'decode_done':   thinkTxt.textContent=`Querying ChromaDB (${evt.width}×${evt.height} px)…`;break;
          case 'chroma_start':  thinkTxt.textContent='Searching vector database (RAD-DINO)…';break;
          case 'chroma_done':
            results=evt.results;
            thinkTxt.textContent=`Found ${evt.count} cases · enriching…`;
            thinkEl.remove();
            addMsg('msg-bot', renderSimilarCasesCard(results, '(loading…)', 0, cardId));
            break;
          case 'enrich_done':   results=evt.results;break;
          case 'answer':        answer=evt.text;break;
          case 'done':          elapsed=evt.elapsed_ms;break;
          case 'error':
            thinkEl && thinkEl.parentNode && thinkEl.remove();
            addMsg('msg-bot',`<div class="bubble-error">Image search error: ${esc(evt.message)}</div>`);
            setLoading(false);setStatus('error','error');return;
        }
      }
    }
    const ansPane=document.getElementById(`${cardId}-answer`);
    if(ansPane) ansPane.innerHTML=`<div class="answer-body">${esc(answer)}</div>`;
    const timeEl=document.querySelector(`#${cardId} .irc-time`);
    if(timeEl) timeEl.textContent=`${(elapsed/1000).toFixed(1)}s`;
    addHistory(`[Image] ${question}`, results.length, elapsed, true);
    setStatus('ready','ready');
    loadChromaStats();
  }catch(e){
    try{ thinkEl && thinkEl.parentNode && thinkEl.remove(); }catch(_){}
    if(e.name==='AbortError'){
      addMsg('msg-bot','<div class="bubble-error">Query stopped.</div>');
      setStatus('ready','ready');
    } else {
      addMsg('msg-bot',`<div class="bubble-error">Image search failed: ${esc(String(e))}</div>`);
      setStatus('error','error');
    }
  }
  setLoading(false);inp.focus();
}

// ════════════════════════════════════════════════
// SIMILAR CASES CARD RENDERER
// ════════════════════════════════════════════════
function simClass(pct){
  if(pct>=80) return 'high';
  if(pct>=60) return 'med';
  if(pct>=40) return 'low';
  return 'vlow';
}

function fmtDate(d){
  if(!d||String(d).length!==8) return d||'';
  return `${String(d).slice(0,4)}-${String(d).slice(4,6)}-${String(d).slice(6,8)}`;
}

function renderSimilarCasesCard(results, answer, elapsed, cardId){
  const elapsedSec=(elapsed/1000).toFixed(1);

  const casesHtml = results.length ? results.map(r=>{
    const pct     = r.similarity_pct||0, sc=simClass(pct);
    const sopGroups = r.sop_groups||[];
    const totalFrames = sopGroups.reduce((n,sg)=>n+sg.frames.length, 0);

    const chips=[
      r.study_date        ? `<span class="case-chip">${esc(fmtDate(r.study_date))}</span>` : '',
      r.modality          ? `<span class="case-chip mod">${esc(r.modality)}</span>` : '',
      r.series_description? `<span class="case-chip">${esc(String(r.series_description).slice(0,30))}</span>` : '',
      r.patient_sex       ? `<span class="case-chip">${esc(r.patient_sex)}</span>` : '',
      r.patient_age       ? `<span class="case-chip">${esc(r.patient_age)}</span>` : '',
      `<span class="frame-count-badge">${sopGroups.length} SOP · ${totalFrames} frame${totalFrames!==1?'s':''}</span>`,
    ].filter(Boolean).join('');

    const rptHtml = r.radrpt_excerpt
      ? `<div class="case-rpt">${esc(String(r.radrpt_excerpt).slice(0,220))}</div>` : '';

    // One block per SOP UID, each with its own frame strip
    const sopBlocksHtml = sopGroups.map(sg=>{
      const sopSim   = sg.similarity_pct||0;
      const sopSc    = simClass(sopSim);
      const frameStripHtml = sg.frames.map(f=>{
        const hasPath  = f.source_path && f.source_path !== '';
        const fi       = parseInt(f.frame_index)||0;
        const sim      = f.similarity_pct||0;
        const tip      = `SOP: ${sg.sop_uid}\nFrame: ${fi}  Sim: ${sim}%`;
        const lbMeta   = `${esc(r.accession_number||'?')} · Frame ${fi} · ${sim}%`;
        const thumbUrl = hasPath ? `/api/thumbnail?path=${encodeURIComponent(f.source_path)}&frame=${fi}` : '';
        const frameUrl = hasPath ? `/api/frame?path=${encodeURIComponent(f.source_path)}&frame=${fi}` : '';
        return hasPath
          ? `<img class="frame-thumb" src="${thumbUrl}" loading="lazy"
                  title="${tip}" alt="Frame ${fi}"
                  onclick="openLightbox('${frameUrl}','${lbMeta}')">`
          : `<div class="frame-thumb" style="display:flex;align-items:center;justify-content:center;font-size:9px;color:var(--text3)">?</div>`;
      }).join('');

      return `
        <div class="sop-group">
          <div class="sop-header">
            <span class="sop-uid" title="${esc(sg.sop_uid)}">${esc(sg.sop_uid||'—')}</span>
            <span class="sop-sim ${sopSc}">${sopSim}%</span>
          </div>
          <div class="frame-strip">${frameStripHtml}</div>
        </div>`;
    }).join('');

    return `
      <div class="case-item" style="flex-direction:column;align-items:stretch;gap:6px;">
        <div style="display:flex;align-items:center;gap:12px;">
          <div class="case-rank">${r.rank}</div>
          <div class="case-body">
            <div class="case-top">
              <span class="case-acc" title="${esc(r.accession_number||'')}">${esc(String(r.accession_number||'—').slice(0,32))}</span>
              <div class="sim-wrap">
                <div class="sim-bar"><div class="sim-fill ${sc}" style="width:${pct}%"></div></div>
                <span class="case-sim-pct ${sc}">${pct}%</span>
              </div>
            </div>
            ${chips?`<div class="case-chips">${chips}</div>`:''}
            ${rptHtml}
          </div>
        </div>
        ${sopBlocksHtml}
      </div>`;
  }).join('')
  : '<div class="no-results">No similar cases found in ChromaDB.</div>';

  return `
    <div class="img-result-card" id="${cardId}">
      <div class="irc-header">
        <div class="irc-icon">
          <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        </div>
        <span class="irc-title">Similar Cases</span>
        <span class="irc-badge">RAD-DINO · ChromaDB</span>
        <span class="irc-time">${elapsedSec}s</span>
      </div>
      <div class="irc-tabs">
        <button class="irc-tab active" onclick="switchIrcTab(this,'${cardId}-cases')">
          Cases<span class="irc-count">${results.length}</span>
        </button>
        <button class="irc-tab" onclick="switchIrcTab(this,'${cardId}-answer')">Answer</button>
      </div>
      <div class="irc-pane active" id="${cardId}-cases">
        <div class="similar-cases">${casesHtml}</div>
      </div>
      <div class="irc-pane" id="${cardId}-answer">
        <div class="answer-body">${esc(answer)}</div>
      </div>
    </div>`;
}

// ════════════════════════════════════════════════
// LIGHTBOX
// ════════════════════════════════════════════════
function openLightbox(frameUrl, metaText){
  const lb = document.getElementById('lightbox');
  const img = document.getElementById('lightboxImg');
  img.src = '';              // clear first so loading spinner fires
  img.src = frameUrl;
  document.getElementById('lightboxMeta').textContent = metaText || '';
  lb.classList.add('visible');
  document.body.style.overflow = 'hidden';
}

function closeLightbox(){
  document.getElementById('lightbox').classList.remove('visible');
  document.getElementById('lightboxImg').src = '';
  document.body.style.overflow = '';
}

document.addEventListener('keydown', e => {
  if(e.key === 'Escape') closeLightbox();
});

// ════════════════════════════════════════════════
// TAB SWITCHERS
// ════════════════════════════════════════════════
function switchRcTab(btn,paneId){
  const card=btn.closest('.result-card');
  card.querySelectorAll('.rc-tab').forEach(b=>b.classList.remove('active'));
  card.querySelectorAll('.rc-pane').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(paneId).classList.add('active');
}

function switchIrcTab(btn,paneId){
  const card=btn.closest('.img-result-card');
  card.querySelectorAll('.irc-tab').forEach(b=>b.classList.remove('active'));
  card.querySelectorAll('.irc-pane').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(paneId).classList.add('active');
}

function copySQL(id){
  const pane=document.getElementById(id+'-sql');
  const code=pane.querySelector('.sql-block');
  navigator.clipboard.writeText(code.textContent.trim()).then(()=>{
    const btn=document.querySelector(`[onclick="copySQL('${id}')"]`);
    if(btn){btn.textContent='copied!';setTimeout(()=>btn.textContent='copy SQL',1500);}
  });
}
</script>

<!-- LIGHTBOX -->
<div id="lightbox">
  <button class="lb-close" onclick="closeLightbox()">×</button>
  <img id="lightboxImg" src="" alt="">
  <div id="lightboxMeta"></div>
</div>

</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Main entry point: parse CLI arguments, initialize globals, and start Flask server.
    
    Configures the DICOM Query Server with:
    - SQLite database path for DICOM metadata
    - ChromaDB path for image embeddings
    - Ollama model and API endpoint for agentic NL→SQL pipeline
    - Optional: extended thinking mode, custom port, max agent steps
    
    Logs warnings for optional dependencies not installed but continues if critical
    dependencies (flask, langchain-ollama) are present.
    """
    global _db_path, _chromadb_path, _think, _ollama_host, _agent_max_steps

    parser = argparse.ArgumentParser(
        description="DICOM Query Web Server (Agentic NL→SQL via smolagents + Image RAG with RAD-DINO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 dicom_query_server.py
  python3 dicom_query_server.py --db /data/meta.db --port 5050
  python3 dicom_query_server.py --model qwen3:14b --no-think
  python3 dicom_query_server.py --chromadb /data/AngioVision/chromadb
  python3 dicom_query_server.py --ollama-host http://localhost:11434 --agent-max-steps 12
        """,
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB),
        help=f"SQLite database path (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--chromadb",
        type=str,
        default=str(DEFAULT_CHROMADB),
        help=f"ChromaDB persistence directory (default: {DEFAULT_CHROMADB})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to serve on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable Qwen3 thinking mode by default",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=DEFAULT_OLLAMA_HOST,
        help=(
            f"Ollama server base URL, used by the smolagents NL→SQL agent "
            f"via its OpenAI-compatible API (default: {DEFAULT_OLLAMA_HOST})"
        ),
    )
    parser.add_argument(
        "--agent-max-steps",
        type=int,
        default=DEFAULT_AGENT_MAX_STEPS,
        help=(
            f"Max sql_query tool calls the NL→SQL agent may make per "
            f"question before it must answer (default: {DEFAULT_AGENT_MAX_STEPS})"
        ),
    )
    args = parser.parse_args()

    _db_path = Path(args.db)
    _chromadb_path = Path(args.chromadb)
    _think = not args.no_think
    _ollama_host = args.ollama_host.rstrip("/")
    _agent_max_steps = args.agent_max_steps

    if not _db_path.exists():
        log.warning(f"DB not found at {_db_path} — stats will error until DB is reachable")

    if not RADDINO_OK:
        log.warning(
            "RAD-DINO dependencies not installed (torch, transformers). "
            "Install with: pip install torch transformers"
        )

    if not IMAGE_DEPS_OK:
        log.warning(
            "Image RAG dependencies not installed — /api/image-query will return 503. "
            "Install with: pip install chromadb pillow numpy"
        )
    else:
        log.info(f"ChromaDB path  : {_chromadb_path}")

    if not SMOLAGENTS_OK:
        log.warning(
            "smolagents not installed — /api/query (NL→SQL) will return 503. "
            "Install with: pip install 'smolagents[openai]'"
        )
    else:
        log.info(
            f"Agentic NL→SQL : smolagents ToolCallingAgent via Ollama @ {_ollama_host} "
            f"(max_steps={_agent_max_steps})"
        )

    set_model(args.model)

    log.info(f"Starting DICOM Query Server on http://{args.host}:{args.port}")
    log.info(f"  Database   : {_db_path}")
    log.info(f"  Model      : {args.model}")
    log.info(f"  Thinking   : {'ON' if _think else 'OFF'}")
    log.info(f"  Embeddings : RAD-DINO (if available)")
    log.info(f"  Open       : http://localhost:{args.port}")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()