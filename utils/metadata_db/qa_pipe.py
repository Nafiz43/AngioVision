#!/usr/bin/env python3
"""
DICOM Query Web Server
Serves the browser UI and exposes a REST API that bridges:
  - Natural language → SQL (Ollama)
  - SQL → SQLite execution
  - Results → synthesis (Ollama)

NEW ── Image RAG:
  POST /api/image-query accepts a base64-encoded image, queries ChromaDB via
  OpenCLIP embeddings for the top visually similar DICOM sequences (de-duplicated
  by accession number), enriches each hit with SQLite metadata and radiology
  report excerpts, and streams a synthesised narrative answer back to the UI.

  POST /api/chroma-stats returns live ChromaDB collection size.

Tables served:
  dicom_files          — one row per .dcm file (DICOM metadata)
  radiology_reports    — one row per accession number (radrpt text)
  image_ingestion_status — one row per ingested sequence (for ChromaDB stats)

Usage:
    python3 dicom_query_server.py
    python3 dicom_query_server.py --db /path/to/dicom_staging.db --port 5050
    python3 dicom_query_server.py --model qwen3:14b --no-think
    python3 dicom_query_server.py --chromadb /path/to/chromadb
"""

import re
import sys
import json
import time
import sqlite3
import logging
import argparse
import threading
from pathlib import Path
from typing import Optional

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
    from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
    IMAGE_DEPS_OK = True
except ImportError:
    IMAGE_DEPS_OK = False
    np = None
    _chromadb_mod = None
    OpenCLIPEmbeddingFunction = None

# ── DICOM pixel reading — for thumbnail extraction from source files ──────────
try:
    import pydicom as _pydicom_mod
    PYDICOM_OK = True
except ImportError:
    PYDICOM_OK = False
    _pydicom_mod = None

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DB        = Path("/data/Deep_Angiography/AngioVision/dicom_staging.db")
DEFAULT_CHROMADB  = Path("/data/Deep_Angiography/AngioVision/chromadb")
CHROMA_COLLECTION = "dicom_images"
DEFAULT_MODEL     = "qwen3:8b"
DEFAULT_PORT      = 5050
MAX_ROWS_FOR_SYNTHESIS = 200
MAX_RETRIES       = 2
N_SIMILAR_DEFAULT = 5      # results returned to UI
N_SIMILAR_OVERFETCH = 40   # fetched from ChromaDB before de-dup by accession

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
"""

IMAGE_SYNTHESIS_SYSTEM = """
You are a clinical informatics assistant specialising in DICOM angiography imaging.

You receive: the user's question, and the top visually similar cases retrieved from a
ChromaDB vector database using OpenCLIP image embedding similarity.

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

# ── Global state ──────────────────────────────────────────────────────────────
app   = Flask(__name__, static_folder=None)
CORS(app)

_db_path         : Path                  = DEFAULT_DB
_chromadb_path   : Path                  = DEFAULT_CHROMADB
_ollama          : Optional[ChatOllama]  = None
_think           : bool                  = True
_db_stats_cache  : Optional[dict]        = None
_chroma_collection                       = None   # lazy-initialised
_lock            = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════════
# LLM helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_ollama() -> ChatOllama:
    global _ollama
    if _ollama is None:
        _ollama = ChatOllama(model=DEFAULT_MODEL)
    return _ollama


def set_model(model: str):
    global _ollama
    _ollama = ChatOllama(model=model)
    log.info(f"Model set to: {model}")


def llm_call(messages: list[dict], think: bool = True) -> str:
    ollama      = get_ollama()
    lc_messages = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            if not think:
                content = "/no_think\n" + content
            lc_messages.append(HumanMessage(content=content))
    response = ollama.invoke(lc_messages)
    text     = response.content
    text     = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def clean_sql(raw: str) -> str:
    sql = re.sub(r"```(?:sql|sqlite)?\s*", "", raw, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql)
    return sql.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite helpers
# ═══════════════════════════════════════════════════════════════════════════════

def open_db() -> sqlite3.Connection:
    con = sqlite3.connect(str(_db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def run_sql_query(sql: str) -> list[dict]:
    with _lock:
        con = open_db()
        try:
            con.execute("PRAGMA query_only = ON")
            cur  = con.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            con.close()


def get_db_stats() -> dict:
    global _db_stats_cache
    if _db_stats_cache:
        return _db_stats_cache
    with _lock:
        con = open_db()
        try:
            total    = con.execute("SELECT COUNT(*) FROM dicom_files WHERE parse_error IS NULL").fetchone()[0]
            patients = con.execute("SELECT COUNT(DISTINCT patient_id) FROM dicom_files WHERE parse_error IS NULL").fetchone()[0]
            studies  = con.execute("SELECT COUNT(DISTINCT study_instance_uid) FROM dicom_files WHERE parse_error IS NULL").fetchone()[0]
            series   = con.execute("SELECT COUNT(DISTINCT series_instance_uid) FROM dicom_files WHERE parse_error IS NULL").fetchone()[0]
            errors   = con.execute("SELECT COUNT(*) FROM dicom_files WHERE parse_error IS NOT NULL").fetchone()[0]
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
                "instances":   total,
                "patients":    patients,
                "studies":     studies,
                "series":      series,
                "errors":      errors,
                "modalities":  [{"modality": r[0] or "?", "count": r[1]} for r in modalities],
                "db_path":     str(_db_path),
                "rpt_total":   rpt_total,
                "rpt_linked":  rpt_linked,
                "rpt_unlinked": rpt_unlinked,
            }
            return _db_stats_cache
        finally:
            con.close()


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDB / image helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_chroma_collection():
    """Lazy-initialise and return the ChromaDB collection, or None on failure."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    if not IMAGE_DEPS_OK:
        return None
    try:
        client = _chromadb_mod.PersistentClient(path=str(_chromadb_path))
        ef     = OpenCLIPEmbeddingFunction()
        _chroma_collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=ef,
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


def decode_image_to_uint8_rgb(b64_str: str):
    """Decode a base64 image string to a uint8 RGB numpy array (HxWx3)."""
    img_bytes = base64.b64decode(b64_str)
    img = PilImage.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def enrich_results_from_sqlite(results: list[dict]) -> list[dict]:
    """
    Join ChromaDB hit metadata with SQLite to add study_date, series_description,
    patient demographics, and a radiology report excerpt (first 400 chars).
    Non-fatal: returns original results if any SQLite error occurs.
    """
    accessions = list({r["accession_number"] for r in results if r.get("accession_number")})
    if not accessions:
        return results

    enrichment_map: dict[str, dict] = {}
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
                    accessions,
                ).fetchall()
            finally:
                con.close()

        for row in rows:
            acc = row["accession_number"]
            enrichment_map[acc] = {
                "study_date":         row["study_date"]        or "",
                "series_description": row["series_description"] or "",
                "modality":           row["modality"]           or "",
                "patient_age":        row["patient_age"]        or "",
                "patient_sex":        row["patient_sex"]        or "",
                "radrpt_excerpt":     row["radrpt_excerpt"]     or "",
            }
    except Exception as exc:
        log.warning(f"SQLite enrichment failed: {exc}")

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
    Open a DICOM file, extract the frame at *frame_index*, normalise to uint8
    RGB, resize so the longest side is at most thumb_px, and return a
    base64-encoded PNG string.

    Returns None on any failure (missing file, corrupt data, missing deps).
    320 px default keeps each thumbnail under ~30 KB while still looking sharp
    in a full-screen lightbox at typical monitor DPIs.
    """
    if not IMAGE_DEPS_OK or not PYDICOM_OK:
        return None
    try:
        ds          = _pydicom_mod.dcmread(str(source_path), force=False)
        photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        pixels      = ds.pixel_array   # loads pixel data

        # ── Normalise array shape → extract target frame ─────────────────────
        if pixels.ndim == 2:
            frame = pixels                          # single grayscale
        elif pixels.ndim == 3:
            if pixels.shape[2] in (3, 4):
                frame = pixels                      # single RGB/RGBA (H,W,C)
            else:
                fi    = min(max(0, frame_index), pixels.shape[0] - 1)
                frame = pixels[fi]                  # multi-frame grayscale (N,H,W)
        elif pixels.ndim == 4:
            fi    = min(max(0, frame_index), pixels.shape[0] - 1)
            frame = pixels[fi]                      # multi-frame colour (N,H,W,C)
        else:
            return None

        # ── Convert to uint8 RGB ─────────────────────────────────────────────
        if frame.ndim == 2:
            f      = frame.astype(np.float32)
            lo, hi = f.min(), f.max()
            f      = (f - lo) / (hi - lo + 1e-8) * 255.0
            f      = f.astype(np.uint8)
            if "MONOCHROME1" in photometric:
                f = 255 - f         # invert: high value = dark
            rgb = np.stack([f, f, f], axis=-1)
        elif frame.ndim == 3:
            f      = frame.astype(np.float32)
            lo, hi = f.min(), f.max()
            f      = (f - lo) / (hi - lo + 1e-8) * 255.0
            f      = f.astype(np.uint8)
            rgb    = f[:, :, :3]    # drop alpha channel if present
        else:
            return None

        # ── Resize and encode ────────────────────────────────────────────────
        img = PilImage.fromarray(rgb, "RGB")
        img.thumbnail((thumb_px, thumb_px), PilImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as exc:
        log.debug(f"Thumbnail failed [{source_path}:{frame_index}]: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# API routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/stats", methods=["GET"])
def api_stats():
    try:
        return jsonify(get_db_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chroma-stats", methods=["GET"])
def api_chroma_stats():
    """Return ChromaDB collection size and sequence count (not cached)."""
    if not IMAGE_DEPS_OK:
        return jsonify({
            "available": False,
            "error": "Image deps not installed (chromadb, Pillow, open-clip-torch)",
        })
    try:
        col = get_chroma_collection()
        if col is None:
            return jsonify({"available": False, "count": 0, "sequences": 0})

        count = col.count()
        seq_count = None
        try:
            with _lock:
                con = open_db()
                try:
                    seq_count = con.execute(
                        "SELECT COUNT(*) FROM image_ingestion_status WHERE status='completed'"
                    ).fetchone()[0]
                finally:
                    con.close()
        except Exception:
            pass

        return jsonify({
            "available":  True,
            "count":      count,
            "sequences":  seq_count,
            "collection": CHROMA_COLLECTION,
            "path":       str(_chromadb_path),
        })
    except Exception as exc:
        return jsonify({"available": False, "error": str(exc)})


@app.route("/api/query", methods=["POST"])
def api_query():
    """
    POST { "question": "...", "think": true, "model": "qwen3:8b" }
    Returns streaming SSE events for the NL→SQL→synthesis pipeline.
    """
    data     = request.get_json(force=True)
    question = data.get("question", "").strip()
    think    = data.get("think", _think)
    model    = data.get("model", None)

    if not question:
        return jsonify({"error": "question required"}), 400

    if model:
        set_model(model)

    def generate():
        t0 = time.time()

        def emit(obj: dict) -> str:
            return "data: " + json.dumps(obj) + "\n\n"

        # Step 1: SQL generation
        yield emit({"event": "sql_start"})
        try:
            raw_sql = llm_call([
                {"role": "system", "content": SCHEMA_CONTEXT},
                {"role": "user",   "content": f"Convert this question to a SQLite SQL query:\n\n{question}"},
            ], think=think)
            sql = clean_sql(raw_sql)
        except Exception as e:
            yield emit({"event": "error", "message": f"SQL generation failed: {e}"})
            return

        yield emit({"event": "sql_done", "sql": sql})

        # Step 2: Execute SQL (with retry / repair)
        yield emit({"event": "exec_start"})
        rows       = None
        last_error = ""
        for attempt in range(1 + MAX_RETRIES):
            try:
                rows = run_sql_query(sql)
                break
            except Exception as exc:
                last_error = str(exc)
                log.warning(f"SQL exec error (attempt {attempt+1}): {exc}")
                if attempt < MAX_RETRIES:
                    try:
                        repair_raw = llm_call([
                            {"role": "system", "content": ERROR_REPAIR_SYSTEM},
                            {"role": "user",   "content": (
                                f"Original question: {question}\n\n"
                                f"Failed SQL:\n{sql}\n\n"
                                f"SQLite error: {last_error}\n\n"
                                f"Schema:\n{SCHEMA_CONTEXT}"
                            )},
                        ], think=think)
                        sql = clean_sql(repair_raw)
                        yield emit({"event": "sql_repaired", "sql": sql})
                    except Exception as re_exc:
                        log.warning(f"Repair failed: {re_exc}")

        if rows is None:
            yield emit({"event": "error",
                        "message": f"SQL execution failed after {MAX_RETRIES} retries: {last_error}"})
            return

        def serialise(v):
            if v is None:
                return None
            try:
                json.dumps(v)
                return v
            except Exception:
                return str(v)

        clean_rows = [{k: serialise(v) for k, v in row.items()} for row in rows]
        yield emit({"event": "exec_done", "rows": clean_rows, "row_count": len(clean_rows)})

        # Step 3: Synthesis
        yield emit({"event": "synth_start"})
        try:
            display   = clean_rows[:MAX_ROWS_FOR_SYNTHESIS]
            truncated = len(clean_rows) > MAX_ROWS_FOR_SYNTHESIS
            result_str = json.dumps(display, indent=2, default=str)
            if truncated:
                result_str += f"\n\n[... {len(clean_rows) - MAX_ROWS_FOR_SYNTHESIS} additional rows truncated]"

            answer = llm_call([
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user",   "content": (
                    f"Question: {question}\n\n"
                    f"SQL executed:\n{sql}\n\n"
                    f"Results ({len(clean_rows)} total rows):\n{result_str}"
                )},
            ], think=False)
        except Exception as e:
            yield emit({"event": "error", "message": f"Synthesis failed: {e}"})
            return

        elapsed = int((time.time() - t0) * 1000)
        yield emit({"event": "answer", "text": answer})
        yield emit({"event": "done",   "elapsed_ms": elapsed})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/image-query", methods=["POST"])
def api_image_query():
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
      thumbs_start  {total}
      thumb_ready   {rank, thumbnail_b64}   ← one per result, streams progressively
      thumbs_done   {results}
      synth_start   → answer {text} → done {elapsed_ms}
      error {message}

    De-duplicates ChromaDB hits by accession_number so each returned case
    represents a distinct patient study, not multiple frames of the same sequence.
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
                "Run: pip install chromadb open-clip-torch pillow numpy"
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

            # Over-fetch to allow de-duplication by accession number
            fetch_n = min(available, N_SIMILAR_OVERFETCH)

            raw = col.query(
                query_images=[img_array],
                n_results=fetch_n,
                include=["metadatas", "distances"],
            )
            ids       = raw["ids"][0]
            distances = raw["distances"][0]
            metadatas = raw["metadatas"][0]

            # De-duplicate by accession_number — keep highest-similarity frame
            seen_acc: dict[str, dict] = {}
            for id_, dist, meta in zip(ids, distances, metadatas):
                acc = meta.get("accession_number") or "UNKNOWN"
                sim = max(0.0, 1.0 - float(dist))   # cosine dist → similarity
                if acc not in seen_acc or sim > seen_acc[acc]["_sim_raw"]:
                    seen_acc[acc] = {
                        "_sim_raw":      sim,
                        "chroma_id":     id_,
                        "similarity_pct": round(sim * 100, 1),
                        "distance":      round(float(dist), 4),
                        # Spread all ChromaDB metadata fields; coerce None → ""
                        **{k: (v if v is not None else "") for k, v in meta.items()},
                    }

            # Sort by descending similarity, keep top n_results
            top = sorted(seen_acc.values(), key=lambda x: x["_sim_raw"], reverse=True)[:n_results]
            results = []
            for i, r in enumerate(top):
                r.pop("_sim_raw", None)
                r["rank"] = i + 1
                # Ensure integer types survive JSON serialisation cleanly
                for int_key in ("frame_index", "total_frames", "rows", "columns"):
                    if int_key in r and r[int_key] != "":
                        try:
                            r[int_key] = int(r[int_key])
                        except (TypeError, ValueError):
                            pass
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

        # ── Step 4: Load DICOM frame thumbnails ──────────────────────────────
        # Read each source .dcm, extract the matched frame, and stream it back
        # one-by-one so the UI can render images as they arrive.
        yield emit({"event": "thumbs_start", "total": len(enriched)})
        for r in enriched:
            src = r.get("source_path", "")
            fi  = r.get("frame_index", 0)
            try:
                fi_int = int(fi) if fi != "" else 0
            except (TypeError, ValueError):
                fi_int = 0

            thumb = load_dicom_frame_as_b64(str(src), fi_int) if src else None
            r["thumbnail_b64"] = thumb

            # Emit each thumbnail individually — the card updates progressively
            yield emit({
                "event":        "thumb_ready",
                "rank":         r["rank"],
                "thumbnail_b64": thumb,
            })

        yield emit({"event": "thumbs_done", "results": enriched})

        # ── Step 5: LLM synthesis ────────────────────────────────────────────
        yield emit({"event": "synth_start"})
        try:
            cases_text = "\n".join(
                f"#{r['rank']} (similarity {r['similarity_pct']}%): "
                f"Accession={r.get('accession_number','?')}, "
                f"Date={r.get('study_date','?')}, "
                f"Modality={r.get('modality','?')}, "
                f"Frame {r.get('frame_index','?')}/{r.get('total_frames','?')}, "
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
                    f"Top {len(enriched)} visually similar cases retrieved from ChromaDB "
                    f"(OpenCLIP embedding similarity):\n\n{cases_text}"
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


@app.route("/api/model", methods=["POST"])
def api_set_model():
    data  = request.get_json(force=True)
    model = data.get("model", "").strip()
    if not model:
        return jsonify({"error": "model required"}), 400
    set_model(model)
    return jsonify({"ok": True, "model": model})


@app.route("/api/sql", methods=["POST"])
def api_run_sql():
    """Run arbitrary SQL directly (for advanced users)."""
    data = request.get_json(force=True)
    sql  = data.get("sql", "").strip()
    if not sql:
        return jsonify({"error": "sql required"}), 400
    try:
        rows = run_sql_query(sql)
        def serialise(v):
            if v is None: return None
            try: json.dumps(v); return v
            except: return str(v)
        clean_rows = [{k: serialise(v) for k, v in row.items()} for row in rows]
        return jsonify({"rows": clean_rows, "row_count": len(clean_rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/", methods=["GET"])
def index():
    return HTML_PAGE


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

/* Case items */
.similar-cases{padding:4px 0}
.case-item{display:flex;gap:12px;padding:12px 16px;border-bottom:1px solid rgba(48,54,61,.5);transition:background .12s;cursor:default}
.case-item:last-child{border-bottom:none}
.case-item:hover{background:rgba(255,255,255,.02)}
.case-rank{width:26px;height:26px;border-radius:6px;background:var(--violet-bg);border:1px solid rgba(124,58,237,.25);display:flex;align-items:center;justify-content:center;font-size:11px;font-family:var(--mono);font-weight:700;color:var(--violet-light);flex-shrink:0;margin-top:1px}
.case-body{flex:1;min-width:0;display:flex;flex-direction:column;gap:5px}
.case-top{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.case-acc{font-size:12px;font-family:var(--mono);font-weight:600;color:var(--teal);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px}
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
.case-chip.frame{color:var(--violet-light);border-color:rgba(124,58,237,.25);background:var(--violet-bg)}
.case-rpt{font-size:11px;color:var(--report);line-height:1.55;font-style:italic;border-left:2px solid rgba(240,136,62,.3);padding-left:8px;margin-top:1px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical}
.no-results{padding:28px 16px;text-align:center;font-size:12px;color:var(--text3);font-family:var(--mono);line-height:1.8}

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

/* ── Frame thumbnail in case card ── */
.case-img-wrap{width:100px;flex-shrink:0;display:flex;flex-direction:column;align-items:center;gap:5px}
.case-img{
  width:100px;height:100px;
  object-fit:contain;
  border-radius:7px;
  border:1px solid var(--border2);
  background:#000;
  cursor:zoom-in;
  display:block;
  transition:transform .15s,box-shadow .15s;
}
.case-img:hover{transform:scale(1.04);box-shadow:0 0 0 2px var(--violet-light),0 4px 16px rgba(0,0,0,.5)}
.case-img-spinner{
  width:100px;height:100px;
  border-radius:7px;border:1px dashed var(--border2);
  background:var(--bg4);
  display:flex;align-items:center;justify-content:center;
  color:var(--text3);
}
.case-img-spinner .mini-dot{
  width:4px;height:4px;border-radius:50%;background:var(--text3);
  animation:blink 1.4s ease-in-out infinite;
  margin:1px;display:inline-block;
}
.case-img-spinner .mini-dot:nth-child(2){animation-delay:.2s}
.case-img-spinner .mini-dot:nth-child(3){animation-delay:.4s}
.case-frame-label{font-size:9px;font-family:var(--mono);color:var(--text3);text-align:center}
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
    <span class="hdr-sub">NL → SQL · Image RAG</span>
    <div class="hdr-right">
      <div class="status-pill"><span class="status-dot"></span><span id="statusTxt">connecting…</span></div>
      <select class="model-select" id="modelSelect" onchange="onModelChange()">
        <option>qwen3:8b</option>
        <option>qwen3:14b</option>
        <option>qwen3:32b</option>
        <option>qwen3:35b</option>
        <option>llama3.2:latest</option>
      </select>
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
        <div class="empty-sub">Natural language → SQL for metadata &amp; reports. Upload an angiography image to find visually similar cases via ChromaDB vector search.</div>
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
        <div class="img-preview-label">Image attached · vector search enabled</div>
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
      <div class="sb-label">ChromaDB · Vector Store</div>
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
const CREDENTIALS = [
  { user: 'goldman',  pass: 'xK9#mQ2$vL5@pN8!' },
  { user: 'vfilkov',  pass: 'ChangeMe#001!'     },
];
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

const inp     = document.getElementById('nlInput');
const sendBtn = document.getElementById('sendBtn');

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
    if(c==='radrpt')return `<td><span class="v-report" title="${esc(s)}">${esc(s.length>80?s.slice(0,78)+'…':s)}</span></td>`;
    if(s.startsWith('/'))return `<td><span class="v-path" title="${esc(s)}">${esc(s.length>50?'…'+s.slice(-48):s)}</span></td>`;
    if(/^\d{8}$/.test(s)&&parseInt(s)>19000101)return `<td><span class="v-num">${s.slice(0,4)}-${s.slice(4,6)}-${s.slice(6,8)}</span></td>`;
    if(s.length>32&&s.includes('.'))return `<td><span class="v-id" title="${esc(s)}">${esc(s.slice(0,28)+'…')}</span></td>`;
    if(!isNaN(s)&&s.trim()!=='')return `<td><span class="v-num">${esc(s)}</span></td>`;
    return `<td>${esc(s.length>60?s.slice(0,58)+'…':s)}</td>`;
  }).join('')}</tr>`).join('')}</tbody>`;
  return `<div class="tbl-scroll"><table class="data-table">${thead}${tbody}</table></div>`;
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
  isLoading=true; sendBtn.disabled=true;
  inp.value=''; inp.style.height='auto';
  addMsg('msg-user',`<div class="bubble bubble-user">${esc(question)}</div>`);
  const thinkEl=addMsg('msg-bot',`<div class="thinking"><div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div><span id="thinkTxt">Generating SQL…</span></div>`);
  const thinkTxt=thinkEl.querySelector('#thinkTxt');
  setStatus('busy','thinking…');
  const think=document.getElementById('thinkToggle').checked;
  const model=document.getElementById('modelSelect').value;
  let sql='',rows=[],rowCount=0,answer='',elapsed=0;
  try{
    const resp=await fetch(`${API}/api/query`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,think,model})});
    const reader=resp.body.getReader(),decoder=new TextDecoder();let buf='';
    while(true){
      const{done,value}=await reader.read();if(done)break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\n\n');buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        let evt;try{evt=JSON.parse(line.slice(6));}catch{continue;}
        switch(evt.event){
          case 'sql_start':    thinkTxt.textContent='Generating SQL…';break;
          case 'sql_done':     sql=evt.sql;thinkTxt.textContent='Executing query…';break;
          case 'sql_repaired': sql=evt.sql;thinkTxt.textContent='Repairing SQL…';break;
          case 'exec_start':   thinkTxt.textContent='Running query…';break;
          case 'exec_done':    rows=evt.rows;rowCount=evt.row_count;thinkTxt.textContent='Synthesising…';break;
          case 'synth_start':  thinkTxt.textContent='Synthesising answer…';break;
          case 'answer':       answer=evt.text;break;
          case 'done':         elapsed=evt.elapsed_ms;break;
          case 'error':
            thinkEl.remove();
            addMsg('msg-bot',`<div class="bubble-error">Error: ${esc(evt.message)}</div>`);
            isLoading=false;sendBtn.disabled=false;setStatus('error','error');return;
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
    addMsg('msg-bot',`<div class="result-card"><div class="rc-tabs">${tabs}<div class="rc-meta"><span>${elapsedSec}s</span></div></div>${panes}</div>`);
    addHistory(question,rowCount,elapsed,false);
    setStatus('ready','ready');
  }catch(e){
    thinkEl.remove();
    addMsg('msg-bot',`<div class="bubble-error">Request failed: ${esc(String(e))}</div>`);
    setStatus('error','error');
  }
  isLoading=false;sendBtn.disabled=false;inp.focus();
}

// ════════════════════════════════════════════════
// IMAGE QUERY (ChromaDB vector search)
// ════════════════════════════════════════════════
async function handleImageSend(question){
  if(!attachedImage) return;
  const img=attachedImage;
  clearImage();
  isLoading=true; sendBtn.disabled=true;
  inp.value=''; inp.style.height='auto';

  // Show user message with image thumbnail
  addMsg('msg-user',`
    <div class="bubble-with-img">
      <img src="${img.dataUrl}" class="bubble-img-preview" alt="uploaded image">
      <span class="bubble-img-caption">${esc(question)}</span>
    </div>`);

  const thinkEl=addMsg('msg-bot',`<div class="thinking"><div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div><span id="thinkTxt">Decoding image…</span></div>`);
  const thinkTxt=thinkEl.querySelector('#thinkTxt');
  setStatus('busy','searching…');

  // Track the card DOM id so we can inject thumbnails progressively
  let cardId=null;
  let results=[],answer='',elapsed=0;

  try{
    const resp=await fetch(`${API}/api/image-query`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
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
          case 'chroma_start':  thinkTxt.textContent='Searching vector database…';break;
          case 'chroma_done':   results=evt.results;thinkTxt.textContent=`Found ${evt.count} candidates · enriching…`;break;
          case 'enrich_start':  thinkTxt.textContent='Enriching from DICOM database…';break;
          case 'enrich_done':   results=evt.results;thinkTxt.textContent='Loading frame images…';break;
          case 'thumbs_start':
            // Render card now (with spinner placeholders) so thumbnails appear in-place
            thinkEl.remove();
            const cardHtml=renderSimilarCasesCard(results,'(loading…)',0);
            const cardEl=addMsg('msg-bot', cardHtml);
            // Extract the generated card ID from the rendered DOM
            cardId=cardEl.querySelector('.img-result-card')?.id||null;
            thinkTxt && (thinkTxt.textContent='Loading frame images…');
            break;
          case 'thumb_ready':
            // Replace spinner for this rank with the actual image
            if(cardId && evt.thumbnail_b64){
              const slotId=`${cardId}-thumb-${evt.rank}`;
              const slot=document.getElementById(slotId);
              if(slot){
                const r=results.find(x=>x.rank===evt.rank)||{};
                const accStr=esc(r.accession_number||'?');
                const frameStr=`Frame ${r.frame_index??'?'} / ${r.total_frames??'?'}`;
                const metaStr=`${accStr} · ${frameStr} · ${esc(r.modality||'')}`;
                slot.outerHTML=`
                  <img class="case-img" id="${slotId}"
                       src="data:image/png;base64,${evt.thumbnail_b64}"
                       alt="Frame ${r.frame_index??''}"
                       title="${metaStr}"
                       onclick="openLightbox('${evt.thumbnail_b64}','${metaStr}')">`;
              }
            }
            break;
          case 'thumbs_done':
            results=evt.results;
            thinkTxt && (thinkTxt.textContent='Synthesising answer…');
            break;
          case 'synth_start':  thinkTxt && (thinkTxt.textContent='Synthesising answer…');break;
          case 'answer':       answer=evt.text;break;
          case 'done':         elapsed=evt.elapsed_ms;break;
          case 'error':
            if(!cardId){
              thinkEl && thinkEl.parentNode && thinkEl.remove();
            }
            addMsg('msg-bot',`<div class="bubble-error">Image search error: ${esc(evt.message)}</div>`);
            isLoading=false;sendBtn.disabled=false;setStatus('error','error');return;
        }
      }
    }

    // If card was already rendered (thumbs_start path), update the Answer tab
    if(cardId && answer){
      const ansPane=document.getElementById(`${cardId}-answer`);
      if(ansPane) ansPane.innerHTML=`<div class="answer-body">${esc(answer)}</div>`;
      // Update elapsed time in header
      const timeEl=document.querySelector(`#${cardId} .irc-time`);
      if(timeEl) timeEl.textContent=`${(elapsed/1000).toFixed(1)}s`;
    } else if(!cardId){
      // Fallback: card wasn't rendered yet (e.g. thumbs_start never fired)
      addMsg('msg-bot', renderSimilarCasesCard(results, answer, elapsed));
    }

    addHistory(`[Image] ${question}`, results.length, elapsed, true);
    setStatus('ready','ready');
    loadChromaStats();
  }catch(e){
    try{ thinkEl && thinkEl.parentNode && thinkEl.remove(); }catch(_){}
    addMsg('msg-bot',`<div class="bubble-error">Image search failed: ${esc(String(e))}<br><small style="opacity:.7">Is the server running? Are ChromaDB + pydicom installed?</small></div>`);
    setStatus('error','error');
  }
  isLoading=false;sendBtn.disabled=false;inp.focus();
}

// ════════════════════════════════════════════════
// LIGHTBOX
// ════════════════════════════════════════════════
function openLightbox(b64, metaText){
  const lb=document.getElementById('lightbox');
  document.getElementById('lightboxImg').src='data:image/png;base64,'+b64;
  document.getElementById('lightboxMeta').textContent=metaText||'';
  lb.style.display='flex';
  // Prevent scroll bleed
  document.body.style.overflow='hidden';
}

function closeLightbox(){
  document.getElementById('lightbox').style.display='none';
  document.getElementById('lightboxImg').src='';
  document.body.style.overflow='';
}

document.addEventListener('keydown',e=>{if(e.key==='Escape')closeLightbox();});

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

function renderSimilarCasesCard(results, answer, elapsed){
  const elapsedSec=(elapsed/1000).toFixed(1);
  const cardId='irc'+Date.now();

  const casesHtml = results.length ? results.map(r=>{
    const pct=r.similarity_pct||0, sc=simClass(pct);
    const chips=[
      r.study_date  ? `<span class="case-chip">${esc(fmtDate(r.study_date))}</span>` : '',
      r.modality    ? `<span class="case-chip mod">${esc(r.modality)}</span>` : '',
      (r.total_frames!=null&&r.total_frames!=='')
        ? `<span class="case-chip frame">Frame ${r.frame_index??'?'} / ${r.total_frames}</span>` : '',
      r.series_description ? `<span class="case-chip">${esc(String(r.series_description).slice(0,30))}</span>` : '',
      r.patient_sex ? `<span class="case-chip">${esc(r.patient_sex)}</span>` : '',
      r.patient_age ? `<span class="case-chip">${esc(r.patient_age)}</span>` : '',
    ].filter(Boolean).join('');

    const rptHtml=r.radrpt_excerpt
      ? `<div class="case-rpt">${esc(String(r.radrpt_excerpt).slice(0,220))}</div>`:'';

    // Spinner placeholder — JS will replace with real image via thumb_ready
    const imgSlotId=`${cardId}-thumb-${r.rank}`;
    const spinnerHtml=`
      <div class="case-img-wrap">
        <div class="case-img-spinner" id="${imgSlotId}">
          <span class="mini-dot"></span><span class="mini-dot"></span><span class="mini-dot"></span>
        </div>
        <div class="case-frame-label">frame ${r.frame_index??'?'}</div>
      </div>`;

    return `
      <div class="case-item">
        <div class="case-rank">${r.rank}</div>
        ${spinnerHtml}
        <div class="case-body">
          <div class="case-top">
            <span class="case-acc" title="${esc(r.accession_number||'')}">${esc(String(r.accession_number||'—').slice(0,28))}</span>
            <div class="sim-wrap">
              <div class="sim-bar"><div class="sim-fill ${sc}" style="width:${pct}%"></div></div>
              <span class="case-sim-pct ${sc}">${pct}%</span>
            </div>
          </div>
          ${chips?`<div class="case-chips">${chips}</div>`:''}
          ${rptHtml}
        </div>
      </div>`;
  }).join('')
  : '<div class="no-results">No similar cases found in ChromaDB.<br>Run image ingestion first:<br>python dicom_ingest_sqlite.py --images-only</div>';

  return `
    <div class="img-result-card" id="${cardId}">
      <div class="irc-header">
        <div class="irc-icon">
          <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        </div>
        <span class="irc-title">Similar Cases</span>
        <span class="irc-badge">ChromaDB · OpenCLIP</span>
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
<!-- LIGHTBOX MODAL -->
<div id="lightbox" style="display:none;position:fixed;inset:0;z-index:9500;background:rgba(0,0,0,.88);display:none;align-items:center;justify-content:center;cursor:zoom-out;animation:fadeIn .15s ease" onclick="closeLightbox()">
  <button style="position:absolute;top:18px;right:22px;background:none;border:none;color:rgba(255,255,255,.7);font-size:28px;cursor:pointer;line-height:1;padding:4px" onclick="closeLightbox()">×</button>
  <div style="display:flex;flex-direction:column;align-items:center;gap:12px" onclick="event.stopPropagation()">
    <img id="lightboxImg" src="" alt="" style="max-width:90vw;max-height:80vh;object-fit:contain;border-radius:8px;box-shadow:0 0 60px rgba(0,0,0,.9)">
    <div id="lightboxMeta" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(255,255,255,.5);text-align:center;line-height:1.7"></div>
  </div>
</div>

</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global _db_path, _chromadb_path, _think

    parser = argparse.ArgumentParser(
        description="DICOM Query Web Server (NL→SQL + Image RAG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 dicom_query_server.py
  python3 dicom_query_server.py --db /data/meta.db --port 5050
  python3 dicom_query_server.py --model qwen3:14b --no-think
  python3 dicom_query_server.py --chromadb /data/AngioVision/chromadb
        """,
    )
    parser.add_argument("--db",       type=str, default=str(DEFAULT_DB),
                        help=f"SQLite database path (default: {DEFAULT_DB})")
    parser.add_argument("--chromadb", type=str, default=str(DEFAULT_CHROMADB),
                        help=f"ChromaDB persistence directory (default: {DEFAULT_CHROMADB})")
    parser.add_argument("--port",     type=int, default=DEFAULT_PORT,
                        help=f"Port to serve on (default: {DEFAULT_PORT})")
    parser.add_argument("--host",     type=str, default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--model",    type=str, default=DEFAULT_MODEL,
                        help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable Qwen3 thinking mode by default")
    args = parser.parse_args()

    _db_path       = Path(args.db)
    _chromadb_path = Path(args.chromadb)
    _think         = not args.no_think

    if not _db_path.exists():
        log.warning(f"DB not found at {_db_path} — stats will error until DB is reachable")

    if not IMAGE_DEPS_OK:
        log.warning(
            "Image RAG dependencies not installed — /api/image-query will return 503. "
            "Install with: pip install chromadb open-clip-torch pillow numpy"
        )
    else:
        log.info(f"ChromaDB path  : {_chromadb_path}")

    set_model(args.model)

    log.info(f"Starting DICOM Query Server on http://{args.host}:{args.port}")
    log.info(f"  Database : {_db_path}")
    log.info(f"  Model    : {args.model}")
    log.info(f"  Thinking : {'ON' if _think else 'OFF'}")
    log.info(f"  Open     : http://localhost:{args.port}")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()