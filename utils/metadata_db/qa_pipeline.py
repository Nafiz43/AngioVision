#!/usr/bin/env python3
"""
DICOM Query Web Server
Serves the browser UI and exposes a REST API that bridges:
  - Natural language → SQL (Ollama)
  - SQL → SQLite execution
  - Results → synthesis (Ollama)

Tables served:
  dicom_files          — one row per .dcm file (DICOM metadata)
  radiology_reports    — one row per accession number (radrpt text)

Usage:
    python3 dicom_query_server.py
    python3 dicom_query_server.py --db /path/to/dicom_staging.db --port 5050
    python3 dicom_query_server.py --model qwen3:14b --no-think
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

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DB    = Path("/data/Deep_Angiography/AngioVision/dicom_staging.db")
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_PORT  = 5050
MAX_ROWS_FOR_SYNTHESIS = 200
MAX_RETRIES   = 2

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
  radrpt             TEXT               — full free-text radiology report (may be thousands of characters)
  source_csv         TEXT               — path to the source CSV file
  csv_inserted_at    TEXT               — UTC ISO8601 timestamp when row was ingested

### CRITICAL: radrpt is a free-text narrative field in radiology_reports, NOT in dicom_files.
  Do NOT search dicom_files columns (e.g. referring_physician, study_description) for report content.
  Always search r.radrpt when the user asks about anything written IN a report.

### Typical structure of a radrpt value (interventional radiology):
  The report text contains labelled sections separated by keywords. Common patterns include:

  PROCEDURE:       list of performed steps (e.g. "Femoral artery access", "Celiac artery selection")
  DATE:            date and time of the procedure
  INDICATION:      clinical indication / diagnosis
  PHYSICIANS:      attending physician name(s) — also appears as "Attending:" sub-field
  FLUOROSCOPY TIME: total fluoroscopy time in minutes (numeric value in text)
  FLUOROSCOPY DOSE: radiation dose (numeric value, units vary — mGy, µGy·m², cGy·cm²)
  CONTRAST:        contrast agent used and volume (e.g. "130mL omni 300")
  TECHNIQUE/FINDINGS: detailed narrative of the procedure
  COMPLICATION:    any complications noted
  IMPRESSION:      summary of key findings
  PLAN:            follow-up plan
  SIGNED BY / Electronically Signed By:
                   the physician who signed off the final report
                   (e.g. "Final Report Electronically Signed By: Smith on 11/1/2018")
                   This is the authorising/signing physician — distinct from referring_physician.

### Key text-search mappings (user intent → correct SQL pattern):
  "signed by <name>"         → LOWER(r.radrpt) LIKE '%signed by%<lowercase name>%'
  "electronically signed by" → LOWER(r.radrpt) LIKE '%electronically signed by%<name>%'
  "attending physician"      → LOWER(r.radrpt) LIKE '%attending%<name>%'
                               OR LOWER(r.radrpt) LIKE '%physicians%<name>%'
  "fluoroscopy dose"         → extract numeric value after 'FLUOROSCOPY DOSE:' in radrpt
  "fluoroscopy time"         → extract numeric value after 'FLUOROSCOPY TIME:' in radrpt
  "contrast volume"          → LOWER(r.radrpt) LIKE '%contrast%'
  "indication / diagnosis"   → LOWER(r.radrpt) LIKE '%indication%<term>%'
  "complication"             → LOWER(r.radrpt) LIKE '%complication%'
  "impression"               → LOWER(r.radrpt) LIKE '%impression%<term>%'
  "procedure type"           → LOWER(r.radrpt) LIKE '%procedure%<term>%'

### Joining the two tables:
  SELECT d.accession_number, d.series_description, d.frame_count,
         d.modality, d.study_date, r.radrpt
  FROM   dicom_files d
  JOIN   radiology_reports r USING (accession_number)
  WHERE  d.parse_error IS NULL
  LIMIT  5;

### Report text search examples:

  -- Reports signed by a specific doctor (search radrpt, NOT referring_physician):
  SELECT DISTINCT r.accession_number, d.study_date, r.radrpt
  FROM   radiology_reports r
  JOIN   dicom_files d USING (accession_number)
  WHERE  LOWER(r.radrpt) LIKE '%signed by%moreno%'
    AND  d.parse_error IS NULL;

  -- Reports mentioning stenosis:
  SELECT DISTINCT r.accession_number, d.study_date
  FROM   radiology_reports r
  JOIN   dicom_files d USING (accession_number)
  WHERE  LOWER(r.radrpt) LIKE '%stenosis%'
    AND  d.parse_error IS NULL;

  -- Average DICOM dose_product for studies signed by a specific doctor:
  -- (dose_product is in dicom_files; signing doctor is in radrpt text)
  SELECT AVG(CAST(d.dose_product AS REAL)) AS avg_dose
  FROM   dicom_files d
  JOIN   radiology_reports r USING (accession_number)
  WHERE  d.parse_error IS NULL
    AND  LOWER(r.radrpt) LIKE '%signed by%smith%'
    AND  d.dose_product IS NOT NULL
    AND  d.dose_product != '';

  -- Fluoroscopy dose from report text (numeric extraction):
  -- Use SUBSTR + INSTR to extract value after 'FLUOROSCOPY DOSE:' label
  SELECT r.accession_number,
         TRIM(SUBSTR(r.radrpt,
              INSTR(UPPER(r.radrpt), 'FLUOROSCOPY DOSE:') + 17,
              20)) AS fluoro_dose_raw
  FROM   radiology_reports r
  JOIN   dicom_files d USING (accession_number)
  WHERE  UPPER(r.radrpt) LIKE '%FLUOROSCOPY DOSE:%'
    AND  d.parse_error IS NULL;

  -- Coverage check — accessions in DICOM but with no report:
  SELECT DISTINCT d.accession_number
  FROM   dicom_files d
  LEFT JOIN radiology_reports r USING (accession_number)
  WHERE  d.parse_error IS NULL
    AND  r.accession_number IS NULL
    AND  d.accession_number NOT LIKE 'MISSING_%';

---

## SQL Rules

1. Always add `WHERE parse_error IS NULL` (on dicom_files) unless the question is about errors.
2. All numeric columns in dicom_files are TEXT — cast explicitly:
     CAST(frame_count AS INTEGER), CAST(kvp AS REAL), CAST(dose_product AS REAL)
3. Text search on radrpt: always use LOWER() + LIKE with % wildcards:
     WHERE LOWER(r.radrpt) LIKE '%signed by%smith%'
     WHERE LOWER(r.radrpt) LIKE '%stenosis%'
4. NEVER search dicom_files columns for physician names that appear in report text.
   "Signed By", "Attending", "PHYSICIANS:" are inside radrpt, not in referring_physician.
5. Date range filtering on study_date (YYYYMMDD string):
     WHERE d.study_date BETWEEN '20090101' AND '20121231'
6. Use DISTINCT when counting or listing unique accession numbers or studies.
7. Default LIMIT 25 unless the user asks for all or the query is an aggregation.
8. DSA series: LOWER(d.series_description) LIKE '%dsa%'
9. accession_number LIKE 'MISSING_%' means DICOM had no AccessionNumber tag.
10. When joining radiology_reports use JOIN ... USING (accession_number).
11. For cross-table queries (e.g. dose by signing doctor): JOIN both tables, filter on
    radrpt text for the doctor, aggregate on dicom_files numeric columns.
12. Output ONLY the SQL query — no explanation, no markdown fences, no preamble.
"""

SYNTHESIS_SYSTEM = """
You are a clinical informatics assistant helping radiologists and researchers
query a DICOM imaging database.

You receive: the question, the SQL executed, and the raw results (JSON).

The database has two tables:
- dicom_files: DICOM metadata (one row per .dcm file)
- radiology_reports: free-text radiology reports (one row per accession number)

Produce a clear, concise, human-readable answer in PLAIN TEXT only.
- Do NOT use markdown tables (no | pipes, no --- separators)
- Do NOT use markdown headers (no ## or **bold**)
- Use plain dashes (-) for lists when listing items
- Summarise patterns when many results; enumerate each item when ≤ 10 rows
- Highlight clinically or technically interesting findings
- If results reference radiology report text (radrpt), summarise key findings mentioned
- If results are empty, say so and suggest why
- Do NOT re-state the SQL; be direct and informative
"""

ERROR_REPAIR_SYSTEM = """
You are an expert SQLite query debugger for a DICOM database.
You receive: the original question, the failed SQL, the SQLite error, and schema context.
Produce a corrected SQL query that fixes the error.
Output ONLY the corrected SQL — no explanation, no markdown fences.

Key rules:
- All columns in dicom_files are TEXT — always CAST numeric values before comparison
- Always include parse_error IS NULL on dicom_files unless asking about errors
- radiology_reports joins to dicom_files on accession_number (use USING or ON clause)
- radrpt is a TEXT column in radiology_reports, NOT in dicom_files
- NEVER search referring_physician or any dicom_files column for physician names that
  appear in report text — "Signed By", "Attending", "PHYSICIANS:" are inside radrpt
- For "signed by <name>" queries: WHERE LOWER(r.radrpt) LIKE '%signed by%<name>%'
- For cross-table aggregations (e.g. avg dose by signing doctor): JOIN both tables,
  filter on radrpt LIKE for the doctor, AVG(CAST(d.dose_product AS REAL)) from dicom_files
"""

# ── Global state ──────────────────────────────────────────────────────────────
app   = Flask(__name__, static_folder=None)
CORS(app)

_db_path  : Path          = DEFAULT_DB
_ollama   : Optional[ChatOllama] = None
_think    : bool          = True
_db_stats_cache : Optional[dict] = None
_lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────
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
    ollama = get_ollama()
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
    text = response.content
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def clean_sql(raw: str) -> str:
    sql = re.sub(r"```(?:sql|sqlite)?\s*", "", raw, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql)
    return sql.strip()


def open_db() -> sqlite3.Connection:
    con = sqlite3.connect(str(_db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def run_sql_query(sql: str) -> list[dict]:
    with _lock:
        con = open_db()
        try:
            con.execute("PRAGMA query_only = ON")
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            return rows
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
                "SELECT modality, COUNT(*) as cnt FROM dicom_files WHERE parse_error IS NULL GROUP BY modality ORDER BY cnt DESC LIMIT 5"
            ).fetchall()

            # radiology_reports stats
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


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def api_stats():
    try:
        return jsonify(get_db_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def api_query():
    """
    POST { "question": "...", "think": true, "model": "qwen3:8b" }
    Returns streaming JSON events:
      { "event": "sql_start" }
      { "event": "sql_done",  "sql": "..." }
      { "event": "exec_start" }
      { "event": "exec_done", "rows": [...], "row_count": N }
      { "event": "synth_start" }
      { "event": "answer",    "text": "..." }
      { "event": "done",      "elapsed_ms": N }
      { "event": "error",     "message": "..." }
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

        # Step 2: Execute SQL (with retry)
        yield emit({"event": "exec_start"})
        rows = None
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
            yield emit({"event": "error", "message": f"SQL execution failed after {MAX_RETRIES} retries: {last_error}"})
            return

        # Serialise rows (handle non-JSON-serialisable types)
        def serialise(v):
            if v is None:
                return None
            try:
                json.dumps(v)
                return v
            except (TypeError, ValueError, OverflowError):
                return str(v)

        clean_rows = [{k: serialise(v) for k, v in row.items()} for row in rows]
        yield emit({"event": "exec_done", "rows": clean_rows, "row_count": len(clean_rows)})

        # Step 3: Synthesis
        yield emit({"event": "synth_start"})
        try:
            display = clean_rows[:MAX_ROWS_FOR_SYNTHESIS]
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
        yield emit({"event": "done", "elapsed_ms": elapsed})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
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
            except (TypeError, ValueError, OverflowError): return str(v)
        clean_rows = [{k: serialise(v) for k, v in row.items()} for row in rows]
        return jsonify({"rows": clean_rows, "row_count": len(clean_rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/", methods=["GET"])
def index():
    """Serve the single-page UI."""
    return HTML_PAGE


# ── HTML UI ───────────────────────────────────────────────────────────────────
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
  --mono:'JetBrains Mono',monospace;
  --sans:'Syne',sans-serif;
}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:var(--sans)}

/* ══════════════════════════════════════════════════
   LOGIN OVERLAY
══════════════════════════════════════════════════ */
#loginOverlay{
  position:fixed;inset:0;z-index:9999;
  background:var(--bg);
  display:flex;align-items:center;justify-content:center;
  transition:opacity .4s ease, visibility .4s ease;
}
#loginOverlay.hidden{
  opacity:0;visibility:hidden;pointer-events:none;
}

/* Subtle animated grid background */
#loginOverlay::before{
  content:'';
  position:absolute;inset:0;
  background-image:
    linear-gradient(rgba(88,166,255,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(88,166,255,.04) 1px, transparent 1px);
  background-size:40px 40px;
  animation:gridDrift 20s linear infinite;
}
@keyframes gridDrift{from{background-position:0 0}to{background-position:40px 40px}}

/* Glow orb behind card */
#loginOverlay::after{
  content:'';
  position:absolute;
  width:500px;height:500px;
  background:radial-gradient(circle, rgba(31,111,235,.15) 0%, transparent 70%);
  border-radius:50%;
  pointer-events:none;
}

.login-card{
  position:relative;z-index:1;
  width:380px;
  background:var(--bg2);
  border:1px solid var(--border2);
  border-radius:16px;
  padding:36px 32px 32px;
  box-shadow:0 0 0 1px rgba(88,166,255,.08), 0 24px 60px rgba(0,0,0,.6);
  animation:cardIn .5s cubic-bezier(.22,1,.36,1) both;
}
@keyframes cardIn{
  from{opacity:0;transform:translateY(20px) scale(.97)}
  to  {opacity:1;transform:translateY(0)   scale(1)}
}

/* Logo mark */
.login-logo{
  width:48px;height:48px;
  background:linear-gradient(135deg,var(--accent2),#0a3d7a);
  border-radius:12px;
  display:flex;align-items:center;justify-content:center;
  margin:0 auto 20px;
  box-shadow:0 0 20px rgba(31,111,235,.4);
}
.login-logo svg{width:22px;height:22px;stroke:#fff;fill:none;stroke-width:2}

.login-title{
  text-align:center;
  font-size:19px;font-weight:800;letter-spacing:-.4px;
  margin-bottom:4px;
}
.login-sub{
  text-align:center;
  font-size:11px;font-family:var(--mono);
  color:var(--text3);letter-spacing:.4px;
  margin-bottom:28px;
}

.login-field{margin-bottom:16px}
.login-label{
  display:block;font-size:10px;font-family:var(--mono);
  color:var(--text3);text-transform:uppercase;letter-spacing:.6px;
  margin-bottom:7px;
}
.login-input{
  width:100%;
  background:var(--bg3);
  border:1px solid var(--border);
  border-radius:8px;
  padding:10px 14px;
  font-size:13px;font-family:var(--mono);
  color:var(--text);outline:none;
  transition:border-color .2s, box-shadow .2s;
}
.login-input:focus{
  border-color:var(--accent2);
  box-shadow:0 0 0 3px rgba(31,111,235,.18);
}
.login-input::placeholder{color:var(--text3)}

/* Password wrapper with show/hide toggle */
.pw-wrap{position:relative}
.pw-wrap .login-input{padding-right:40px}
.pw-toggle{
  position:absolute;right:12px;top:50%;transform:translateY(-50%);
  background:none;border:none;cursor:pointer;padding:2px;
  color:var(--text3);transition:color .15s;line-height:0;
}
.pw-toggle:hover{color:var(--text2)}
.pw-toggle svg{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:2}

/* Error message */
.login-error{
  display:none;
  background:rgba(248,81,73,.1);
  border:1px solid rgba(248,81,73,.3);
  border-radius:7px;
  padding:9px 12px;
  font-size:11px;font-family:var(--mono);
  color:#ffb3b3;
  margin-bottom:14px;
  text-align:center;
}
.login-error.visible{display:block;animation:errIn .2s ease}
@keyframes errIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}

/* Submit button */
.login-btn{
  width:100%;
  padding:11px;
  background:var(--accent2);
  border:none;border-radius:8px;
  font-size:13px;font-family:var(--sans);font-weight:700;
  color:#fff;cursor:pointer;
  transition:background .2s, transform .1s, box-shadow .2s;
  letter-spacing:.2px;
  margin-top:4px;
}
.login-btn:hover{background:var(--accent);box-shadow:0 4px 16px rgba(88,166,255,.25)}
.login-btn:active{transform:scale(.98)}
.login-btn:disabled{background:var(--bg4);color:var(--text3);cursor:not-allowed;box-shadow:none}

/* Shake animation for wrong credentials */
@keyframes shake{
  0%,100%{transform:translateX(0)}
  15%    {transform:translateX(-7px)}
  30%    {transform:translateX( 7px)}
  45%    {transform:translateX(-5px)}
  60%    {transform:translateX( 5px)}
  75%    {transform:translateX(-3px)}
  90%    {transform:translateX( 3px)}
}
.login-card.shake{animation:shake .45s ease}

/* Divider / footer */
.login-footer{
  margin-top:20px;padding-top:16px;
  border-top:1px solid var(--border);
  font-size:10px;font-family:var(--mono);
  color:var(--text3);text-align:center;
  line-height:1.6;
}
.login-footer span{color:var(--accent);font-weight:600}

/* Attempt counter */
.attempt-bar{
  height:2px;background:var(--bg4);border-radius:1px;margin-top:10px;overflow:hidden;
}
.attempt-fill{
  height:2px;background:var(--red);border-radius:1px;
  width:0%;transition:width .3s ease;
}

/* ══════════════════════════════════════════════════
   MAIN APP  (hidden until login)
══════════════════════════════════════════════════ */
#appWrapper{
  height:100%;
  opacity:0;
  transition:opacity .4s ease .1s;
  pointer-events:none;
}
#appWrapper.visible{
  opacity:1;pointer-events:auto;
}

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

/* Logout button */
.logout-btn{
  background:none;border:1px solid var(--border);
  border-radius:6px;padding:4px 10px;
  font-size:10px;font-family:var(--mono);
  color:var(--text3);cursor:pointer;
  transition:all .15s;letter-spacing:.3px;
  text-transform:uppercase;
}
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

/* ── Result card ── */
.result-card{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.rc-tabs{display:flex;border-bottom:1px solid var(--border);background:var(--bg3)}
.rc-tab{padding:9px 16px;font-size:11px;font-family:var(--mono);background:none;border:none;color:var(--text3);cursor:pointer;position:relative;transition:color .15s;letter-spacing:.3px;text-transform:uppercase}
.rc-tab:hover{color:var(--text2)}
.rc-tab.active{color:var(--accent)}
.rc-tab.active::after{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;background:var(--accent)}
.rc-meta{margin-left:auto;display:flex;align-items:center;gap:8px;padding-right:12px;font-size:10px;font-family:var(--mono);color:var(--text3)}
.rc-pane{display:none}.rc-pane.active{display:block}

/* ── Answer pane ── */
.answer-body{padding:14px 16px;font-size:13.5px;line-height:1.75;color:var(--text);white-space:pre-wrap;font-family:var(--sans)}

/* ── SQL pane ── */
.sql-block{padding:14px 16px;font-family:var(--mono);font-size:12px;line-height:1.8;overflow-x:auto;color:#c9d1d9;white-space:pre}
.kw{color:#ff7b72;font-weight:600}
.fn{color:#d2a8ff}
.str{color:#a5d6ff}
.nm{color:#79c0ff}
.cmt{color:var(--text3)}

/* ── Table pane ── */
.tbl-scroll{overflow-x:auto;max-height:400px;overflow-y:auto}
.tbl-scroll::-webkit-scrollbar{height:5px;width:5px}.tbl-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.data-table{border-collapse:collapse;width:max-content;min-width:100%;font-family:var(--mono);font-size:11.5px}
.data-table thead{position:sticky;top:0;z-index:2}
.data-table th{background:var(--bg4);padding:8px 14px;text-align:left;color:var(--text2);white-space:nowrap;border-bottom:1px solid var(--border2);font-size:10px;font-weight:600;letter-spacing:.5px;text-transform:uppercase}
.data-table td{padding:7px 14px;border-bottom:1px solid rgba(48,54,61,.5);white-space:nowrap;color:var(--text);vertical-align:middle;max-width:300px;overflow:hidden;text-overflow:ellipsis}
.data-table tr:last-child td{border-bottom:none}
.data-table tr:hover td{background:rgba(255,255,255,.03)}
.data-table tr:nth-child(even) td{background:rgba(255,255,255,.015)}
.v-null{color:var(--text3);font-style:italic}
.v-num{color:#79c0ff}
.v-path{color:var(--purple);font-size:10.5px}
.v-id{color:var(--teal);font-size:10.5px}
.v-report{color:var(--report);font-size:10.5px;font-style:italic}
.tbl-footer{padding:8px 14px;font-size:10px;font-family:var(--mono);color:var(--text3);border-top:1px solid var(--border);background:var(--bg3);display:flex;align-items:center;gap:8px}

/* ── Chip groups ── */
.chips-wrap{padding:4px 20px 8px;display:flex;flex-direction:column;gap:4px}
.chip-group{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.chip-group-label{font-size:9px;font-family:var(--mono);color:var(--text3);text-transform:uppercase;letter-spacing:.6px;padding-right:2px;white-space:nowrap}
.chip{padding:5px 12px;background:var(--bg3);border:1px solid var(--border);border-radius:100px;font-size:11.5px;font-family:var(--mono);color:var(--text2);cursor:pointer;transition:all .15s;user-select:none}
.chip:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-bg)}
.chip.report{border-color:rgba(240,136,62,.3);color:var(--report)}
.chip.report:hover{border-color:var(--report);background:var(--report-bg)}

/* ── Thinking animation ── */
.thinking{display:flex;align-items:center;gap:8px;padding:10px 14px;background:var(--bg3);border:1px solid var(--border);border-radius:10px;font-size:12px;color:var(--text2);font-family:var(--mono)}
.dots{display:flex;gap:3px}
.dot{width:4px;height:4px;border-radius:50%;background:var(--accent);animation:blink 1.4s ease-in-out infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.2;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}

/* ── Empty state ── */
.empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;text-align:center;padding:40px}
.empty-icon{width:56px;height:56px;background:var(--bg3);border:1px solid var(--border);border-radius:14px;display:flex;align-items:center;justify-content:center}
.empty-title{font-size:20px;font-weight:800}
.empty-sub{font-size:13px;color:var(--text2);max-width:320px;line-height:1.6}

/* ── Input bar ── */
.input-bar{padding:14px 20px;border-top:1px solid var(--border);background:var(--bg2);display:flex;gap:8px;align-items:flex-end}
.input-wrap{flex:1;background:var(--bg3);border:1px solid var(--border);border-radius:10px;display:flex;align-items:flex-end;padding:0 12px;transition:border-color .2s}
.input-wrap:focus-within{border-color:var(--accent)}
textarea.nl-input{flex:1;background:none;border:none;outline:none;color:var(--text);font-family:var(--sans);font-size:13.5px;padding:11px 0;resize:none;line-height:1.55;max-height:130px;overflow-y:auto}
textarea.nl-input::placeholder{color:var(--text3)}
.send-btn{width:40px;height:40px;border-radius:9px;border:none;background:var(--accent2);cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background .2s}
.send-btn:hover:not(:disabled){background:var(--accent)}
.send-btn:disabled{background:var(--bg4);cursor:not-allowed}
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
.hist-meta{font-family:var(--mono);font-size:9.5px;color:var(--text3)}
.modality-row{display:flex;align-items:center;gap:8px;font-size:11px;font-family:var(--mono);padding:3px 0;color:var(--text2)}
.mod-bar-wrap{flex:1;background:var(--bg4);border-radius:3px;height:5px;overflow:hidden}
.mod-bar{height:5px;background:var(--accent2);border-radius:3px;transition:width .5s ease}
.copy-btn{font-size:10px;font-family:var(--mono);background:none;border:1px solid var(--border);color:var(--text3);border-radius:4px;padding:2px 7px;cursor:pointer;transition:all .15s;margin-left:auto}
.copy-btn:hover{border-color:var(--accent);color:var(--accent)}
</style>
</head>
<body>

<!-- ══════════════════════════════════════════════
     LOGIN OVERLAY
══════════════════════════════════════════════ -->
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
      <input class="login-input" id="loginUser" type="text"
             placeholder="Enter username" autocomplete="username" spellcheck="false">
    </div>

    <div class="login-field">
      <label class="login-label" for="loginPass">Password</label>
      <div class="pw-wrap">
        <input class="login-input" id="loginPass" type="password"
               placeholder="Enter password" autocomplete="current-password">
        <button class="pw-toggle" id="pwToggle" type="button" title="Show / hide password"
                onclick="togglePwVisibility()">
          <!-- Eye icon (shown when password is hidden) -->
          <svg id="eyeOpen" viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
          <!-- Eye-off icon (shown when password is visible) -->
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

<!-- ══════════════════════════════════════════════
     MAIN APP
══════════════════════════════════════════════ -->
<div id="appWrapper">
<div class="layout">

  <!-- Header -->
  <header class="header">
    <div class="logo">
      <svg viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h10"/><circle cx="19" cy="17" r="3"/></svg>
    </div>
    <span class="hdr-title">AngioVision · Query Engine</span>
    <div class="hdr-sep"></div>
    <span class="hdr-sub">NL → SQL → SQLite</span>
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

  <!-- Chat -->
  <main class="chat-col">
    <div class="messages" id="messages">
      <div class="empty" id="emptyState">
        <div class="empty-icon">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="1.8"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>
        </div>
        <div class="empty-title">Ask about your DICOM data</div>
        <div class="empty-sub">Natural language → SQL → results. Queries span DICOM metadata and radiology reports. Try the examples below or type your own question.</div>
      </div>
    </div>

    <div class="chips-wrap">
      <div class="chip-group">
        <span class="chip-group-label">DICOM</span>
        <span class="chip" onclick="fillQ('How many unique patients are in the database?')">Patient count</span>
        <span class="chip" onclick="fillQ('How many studies, series, and instances do we have in total?')">Dataset overview</span>
        <span class="chip" onclick="fillQ('List all DSA series with more than 20 frames, showing accession number and source path.')">DSA > 20 frames</span>
        <span class="chip" onclick="fillQ('What modalities are present and how many instances does each have?')">Modalities</span>
        <span class="chip" onclick="fillQ('How many studies were performed each year? Order by year.')">Studies by year</span>
        <span class="chip" onclick="fillQ('Which instances have the highest dose product? Show top 10.')">Top dose</span>
        <span class="chip" onclick="fillQ('Find all instances where KVP was above 80.')">KVP > 80</span>
        <span class="chip" onclick="fillQ('How many accession numbers are synthetic (missing from DICOM)?')">Missing accessions</span>
      </div>
      <div class="chip-group">
        <span class="chip-group-label">Reports</span>
        <span class="chip report" onclick="fillQ('How many accession numbers have a linked radiology report?')">Report coverage</span>
        <span class="chip report" onclick="fillQ('Show 5 example radiology reports with their accession number and study date.')">Sample reports</span>
        <span class="chip report" onclick="fillQ('Find all reports electronically signed by [Doctor Name]. Show accession number, study date, and the report text.')">Signed by doctor</span>
        <span class="chip report" onclick="fillQ('Find all studies whose radiology report mentions stenosis. Show accession number, study date, and a snippet of the report.')">Stenosis mentions</span>
        <span class="chip report" onclick="fillQ('Find all studies whose radiology report mentions occlusion.')">Occlusion mentions</span>
        <span class="chip report" onclick="fillQ('What is the average DICOM dose product for studies whose radiology report was signed by a specific doctor?')">Avg dose by doctor</span>
        <span class="chip report" onclick="fillQ('Which accession numbers have DICOM data but no linked radiology report?')">Missing reports</span>
        <span class="chip report" onclick="fillQ('Show 5 studies with their modality, frame count, and first 300 characters of the linked radiology report.')">Metadata + report</span>
      </div>
    </div>

    <div class="input-bar">
      <div class="input-wrap">
        <textarea class="nl-input" id="nlInput" rows="1" placeholder="Ask about DICOM metadata, radiology reports, or both…"></textarea>
      </div>
      <button class="send-btn" id="sendBtn" onclick="handleSend()" title="Send (Enter)">
        <svg viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
      </button>
    </div>
  </main>

  <!-- Sidebar -->
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
    <div class="sb-sec" id="modalitySec" style="display:none">
      <div class="sb-label">Modalities</div>
      <div id="modalityBars"></div>
    </div>
    <div class="sb-sec">
      <div class="sb-label">Options</div>
      <div class="opt-row">
        <span class="opt-lab">Thinking mode</span>
        <label class="toggle"><input type="checkbox" id="thinkToggle" checked><span class="tslider"></span></label>
      </div>
      <div class="opt-row">
        <span class="opt-lab">Auto-scroll</span>
        <label class="toggle"><input type="checkbox" id="autoScrollToggle" checked><span class="tslider"></span></label>
      </div>
    </div>
    <div class="sb-hdr" style="border-top:1px solid var(--border)">History</div>
    <div class="hist-scroll" id="histScroll"></div>
  </aside>

</div>
</div><!-- /#appWrapper -->

<script>
// ════════════════════════════════════════════════
// LOGIN LOGIC
// ════════════════════════════════════════════════
const CREDENTIALS = [
  { user: 'goldman',  pass: 'xK9#mQ2$vL5@pN8!' },
  { user: 'vfilkov',    pass: 'ChangeMe#001!'     },
  // Add more entries here as needed:
  // { user: 'username', pass: 'password' },
];
const MAX_ATTEMPTS = 5;
let loginAttempts = 0;
let lockoutTimer  = null;

function togglePwVisibility(){
  const inp  = document.getElementById('loginPass');
  const open = document.getElementById('eyeOpen');
  const off  = document.getElementById('eyeOff');
  if(inp.type === 'password'){
    inp.type = 'text';
    open.style.display = 'none';
    off.style.display  = 'block';
  } else {
    inp.type = 'password';
    open.style.display = 'block';
    off.style.display  = 'none';
  }
}

function handleLogin(){
  const u = document.getElementById('loginUser').value.trim();
  const p = document.getElementById('loginPass').value;
  const btn   = document.getElementById('loginBtn');
  const err   = document.getElementById('loginError');
  const card  = document.getElementById('loginCard');
  const fill  = document.getElementById('attemptFill');

  if(btn.disabled) return;

  const match = CREDENTIALS.find(c => c.user === u && c.pass === p);
  if(match){
    // ── Success ──
    err.classList.remove('visible');
    document.getElementById('sessionUser').textContent = u;

    // Fade out overlay, reveal app
    document.getElementById('loginOverlay').classList.add('hidden');
    const app = document.getElementById('appWrapper');
    app.classList.add('visible');

    // Start loading DB stats now that we're in
    loadStats();
    document.getElementById('nlInput').focus();

  } else {
    // ── Failure ──
    loginAttempts++;
    const pct = Math.min(100, (loginAttempts / MAX_ATTEMPTS) * 100);
    fill.style.width = pct + '%';

    err.classList.remove('visible');
    void err.offsetWidth; // reflow to re-trigger animation
    err.classList.add('visible');

    card.classList.remove('shake');
    void card.offsetWidth;
    card.classList.add('shake');

    document.getElementById('loginPass').value = '';
    document.getElementById('loginPass').focus();

    // Lockout after MAX_ATTEMPTS
    if(loginAttempts >= MAX_ATTEMPTS){
      btn.disabled = true;
      let secs = 30;
      err.textContent = `Too many attempts. Try again in ${secs}s.`;
      lockoutTimer = setInterval(()=>{
        secs--;
        err.textContent = `Too many attempts. Try again in ${secs}s.`;
        if(secs <= 0){
          clearInterval(lockoutTimer);
          loginAttempts = 0;
          fill.style.width = '0%';
          btn.disabled = false;
          err.textContent = 'Invalid username or password.';
          err.classList.remove('visible');
          document.getElementById('loginUser').focus();
        }
      }, 1000);
    }
  }
}

function handleLogout(){
  // Clear session and show login again
  document.getElementById('appWrapper').classList.remove('visible');
  const overlay = document.getElementById('loginOverlay');
  overlay.classList.remove('hidden');
  document.getElementById('loginUser').value = '';
  document.getElementById('loginPass').value = '';
  document.getElementById('loginError').classList.remove('visible');
  document.getElementById('sessionUser').textContent = '—';
  document.getElementById('loginUser').focus();
}

// Allow Enter key to submit login
document.addEventListener('DOMContentLoaded', ()=>{
  ['loginUser','loginPass'].forEach(id=>{
    document.getElementById(id).addEventListener('keydown', e=>{
      if(e.key === 'Enter') handleLogin();
    });
  });
  // Focus username on load
  document.getElementById('loginUser').focus();
});

// ════════════════════════════════════════════════
// MAIN APP LOGIC
// ════════════════════════════════════════════════
const API = '';
let history = [];
let isLoading = false;

// ── Input auto-resize ──
const inp = document.getElementById('nlInput');
inp.addEventListener('input', () => { inp.style.height='auto'; inp.style.height=Math.min(inp.scrollHeight,130)+'px'; });
inp.addEventListener('keydown', e => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();handleSend();} });
function fillQ(q){inp.value=q;inp.dispatchEvent(new Event('input'));inp.focus();}

// ── Model change ──
function onModelChange(){
  const m = document.getElementById('modelSelect').value;
  fetch(`${API}/api/model`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:m})})
    .then(r=>r.json()).then(d=>{ if(d.ok) setStatus('ready',m); });
}

// ── Status ──
function setStatus(state, label){
  const dot = document.querySelector('.status-dot');
  const txt = document.getElementById('statusTxt');
  const colors = {ready:getComputedStyle(document.documentElement).getPropertyValue('--green').trim()||'#3fb950', error:'#f85149', busy:'#d29922'};
  dot.style.background = colors[state]||colors.ready;
  dot.style.boxShadow = `0 0 5px ${colors[state]||colors.ready}`;
  txt.textContent = label||state;
}

// ── Fetch stats (called after login) ──
async function loadStats(){
  try{
    const r = await fetch(`${API}/api/stats`);
    const d = await r.json();
    if(d.error) { setStatus('error','db error'); return; }
    document.getElementById('sInst').textContent = fmtNum(d.instances);
    document.getElementById('sPat').textContent  = fmtNum(d.patients);
    document.getElementById('sStu').textContent  = fmtNum(d.studies);
    document.getElementById('sSer').textContent  = fmtNum(d.series);
    document.getElementById('dbPath').textContent = d.db_path||'—';

    // Report stats
    document.getElementById('sRptTotal').textContent  = fmtNum(d.rpt_total);
    document.getElementById('sRptLinked').textContent = fmtNum(d.rpt_linked);
    if(d.rpt_total > 0 && d.studies > 0){
      const pct = Math.min(100, Math.round(d.rpt_linked / d.studies * 100));
      const wrap = document.getElementById('rptCoverageWrap');
      wrap.style.display = 'block';
      document.getElementById('rptCoverageBar').style.width = pct + '%';
      document.getElementById('rptCoverageLabel').textContent =
        `${pct}% of studies have a linked report  ·  ${fmtNum(d.rpt_unlinked)} missing`;
    }

    if(d.modalities && d.modalities.length){
      const sec = document.getElementById('modalitySec');
      const bars = document.getElementById('modalityBars');
      const max = Math.max(...d.modalities.map(m=>m.count));
      bars.innerHTML = d.modalities.map(m=>`
        <div class="modality-row">
          <span style="min-width:28px;color:var(--accent)">${m.modality}</span>
          <div class="mod-bar-wrap"><div class="mod-bar" style="width:${Math.round(m.count/max*100)}%"></div></div>
          <span style="min-width:40px;text-align:right">${fmtNum(m.count)}</span>
        </div>`).join('');
      sec.style.display='block';
    }
    setStatus('ready','ready');
  } catch(e){ setStatus('error','offline'); }
}
function fmtNum(n){ if(n===undefined||n===null||n==='—') return '—'; return Number(n).toLocaleString(); }

// ── SQL syntax highlight ──
function colorSQL(sql){
  const kws = /\b(SELECT|FROM|WHERE|AND|OR|NOT|IN|LIKE|BETWEEN|ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT|OFFSET|JOIN|LEFT|RIGHT|INNER|OUTER|ON|AS|DISTINCT|COUNT|SUM|AVG|MAX|MIN|CAST|LOWER|UPPER|NULL|IS\s+NOT|IS|CASE|WHEN|THEN|ELSE|END|WITH|UNION|ALL|EXISTS|INSERT|UPDATE|DELETE|SET|VALUES|CREATE|DROP|TABLE|INDEX|INTO|PRAGMA|REPLACE|SUBSTRING|LENGTH|TRIM|DATE|STRFTIME|COALESCE|IFNULL|USING)\b/gi;
  return sql
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/'[^']*'/g, s=>`<span class="str">${s}</span>`)
    .replace(/--[^\n]*/g, s=>`<span class="cmt">${s}</span>`)
    .replace(/\b(\d+\.?\d*)\b/g, s=>`<span class="nm">${s}</span>`)
    .replace(kws, s=>`<span class="kw">${s}</span>`);
}

// ── Table renderer ──
function renderTable(rows){
  if(!rows||!rows.length) return '<div style="padding:16px;font-size:12px;color:var(--text3);font-family:var(--mono)">No rows returned.</div>';
  const cols = Object.keys(rows[0]);
  const thead = `<thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead>`;
  const tbody = `<tbody>${rows.map(r=>`<tr>${cols.map(c=>{
    const v = r[c];
    if(v===null||v===undefined) return `<td><span class="v-null">null</span></td>`;
    const s = String(v);
    if(c==='radrpt') return `<td><span class="v-report" title="${esc(s)}">${esc(s.length>80?s.slice(0,78)+'…':s)}</span></td>`;
    if(s.startsWith('/')) return `<td><span class="v-path" title="${esc(s)}">${esc(s.length>50?'…'+s.slice(-48):s)}</span></td>`;
    if(/^\d{8}$/.test(s)&&parseInt(s)>19000101) return `<td><span class="v-num">${s.slice(0,4)}-${s.slice(4,6)}-${s.slice(6,8)}</span></td>`;
    if(s.length>32&&s.includes('.')) return `<td><span class="v-id" title="${esc(s)}">${esc(s.slice(0,28)+'…')}</span></td>`;
    if(!isNaN(s)&&s.trim()!=='') return `<td><span class="v-num">${esc(s)}</span></td>`;
    return `<td>${esc(s.length>60?s.slice(0,58)+'…':s)}</td>`;
  }).join('')}</tr>`).join('')}</tbody>`;
  return `<div class="tbl-scroll"><table class="data-table">${thead}${tbody}</table></div>`;
}
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

// ── Message helpers ──
function addMsg(cls, html){
  const es = document.getElementById('emptyState');
  if(es) es.remove();
  const msgs = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = `msg ${cls}`;
  d.innerHTML = html;
  msgs.appendChild(d);
  if(document.getElementById('autoScrollToggle').checked)
    msgs.scrollTop = msgs.scrollHeight;
  return d;
}

// ── Main send ──
async function handleSend(){
  const q = inp.value.trim();
  if(!q||isLoading) return;
  isLoading = true;
  sendBtn.disabled = true;
  inp.value=''; inp.style.height='auto';

  addMsg('msg-user', `<div class="bubble bubble-user">${esc(q)}</div>`);

  const thinkEl = addMsg('msg-bot',
    `<div class="thinking"><div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div><span id="thinkTxt">Generating SQL…</span></div>`);
  const thinkTxt = thinkEl.querySelector('#thinkTxt');

  setStatus('busy','thinking…');

  const think   = document.getElementById('thinkToggle').checked;
  const model   = document.getElementById('modelSelect').value;
  let sql='', rows=[], rowCount=0, answer='', elapsed=0;

  try {
    const resp = await fetch(`${API}/api/query`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:q, think, model})
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while(true){
      const {done, value} = await reader.read();
      if(done) break;
      buf += decoder.decode(value, {stream:true});
      const lines = buf.split('\n\n');
      buf = lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: ')) continue;
        let evt;
        try{ evt = JSON.parse(line.slice(6)); } catch{ continue; }
        switch(evt.event){
          case 'sql_start':    thinkTxt.textContent='Generating SQL…'; break;
          case 'sql_done':     sql=evt.sql; thinkTxt.textContent='Executing query…'; break;
          case 'sql_repaired': sql=evt.sql; thinkTxt.textContent='Repairing SQL…'; break;
          case 'exec_start':   thinkTxt.textContent='Running query…'; break;
          case 'exec_done':    rows=evt.rows; rowCount=evt.row_count; thinkTxt.textContent='Synthesising answer…'; break;
          case 'synth_start':  thinkTxt.textContent='Synthesising answer…'; break;
          case 'answer':       answer=evt.text; break;
          case 'done':         elapsed=evt.elapsed_ms; break;
          case 'error':
            thinkEl.remove();
            addMsg('msg-bot',`<div class="bubble-error">Error: ${esc(evt.message)}</div>`);
            isLoading=false; sendBtn.disabled=false; setStatus('error','error'); return;
        }
      }
    }

    thinkEl.remove();

    const id = 'rc'+Date.now();
    const elapsedSec = (elapsed/1000).toFixed(1);
    let tabs='', panes='';

    tabs += `<button class="rc-tab active" onclick="switchTab(this,'${id}-ans')">Answer</button>`;
    panes += `<div class="rc-pane active" id="${id}-ans"><div class="answer-body">${esc(answer)}</div></div>`;

    tabs += `<button class="rc-tab" onclick="switchTab(this,'${id}-sql')">SQL</button>`;
    panes += `<div class="rc-pane" id="${id}-sql"><div class="sql-block">${colorSQL(sql)}</div></div>`;

    tabs += `<button class="rc-tab" onclick="switchTab(this,'${id}-tbl')">Table <span style="font-size:9px;opacity:.7">${rowCount}</span></button>`;
    panes += `<div class="rc-pane" id="${id}-tbl">${renderTable(rows)}${rowCount?`<div class="tbl-footer"><span>${rowCount} row${rowCount!==1?'s':''}</span><button class="copy-btn" onclick="copySQL('${id}')">copy SQL</button><span style="margin-left:auto">${elapsedSec}s</span></div>`:''}</div>`;

    addMsg('msg-bot',`<div class="result-card"><div class="rc-tabs">${tabs}<div class="rc-meta"><span>${elapsedSec}s</span></div></div>${panes}</div>`);

    addHistory(q, rowCount, elapsed);
    setStatus('ready','ready');

  } catch(e){
    thinkEl.remove();
    addMsg('msg-bot',`<div class="bubble-error">Request failed: ${esc(String(e))}<br><small style="opacity:.7">Is the server running on port 5050?</small></div>`);
    setStatus('error','error');
  }

  isLoading=false; sendBtn.disabled=false; inp.focus();
}

function switchTab(btn, paneId){
  const card = btn.closest('.result-card');
  card.querySelectorAll('.rc-tab').forEach(b=>b.classList.remove('active'));
  card.querySelectorAll('.rc-pane').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(paneId).classList.add('active');
}

function copySQL(id){
  const pane = document.getElementById(id+'-sql');
  const code = pane.querySelector('.sql-block');
  navigator.clipboard.writeText(code.textContent.trim()).then(()=>{
    const btn = document.querySelector(`[onclick="copySQL('${id}')"]`);
    if(btn){btn.textContent='copied!';setTimeout(()=>btn.textContent='copy SQL',1500);}
  });
}

function addHistory(q, rowCount, elapsed){
  history.unshift({q, rowCount, elapsed, ts: new Date().toLocaleTimeString()});
  if(history.length>30) history.pop();
  const hl = document.getElementById('histScroll');
  hl.innerHTML = history.map(h=>`
    <div class="hist-item" onclick="fillQ(${JSON.stringify(h.q)})">
      <div class="hist-q">${esc(h.q)}</div>
      <div class="hist-meta">${h.ts} &nbsp;·&nbsp; ${h.rowCount} row${h.rowCount!==1?'s':''} &nbsp;·&nbsp; ${(h.elapsed/1000).toFixed(1)}s</div>
    </div>`).join('');
}
</script>
</body>
</html>
"""

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    global _db_path, _think

    parser = argparse.ArgumentParser(
        description="DICOM Query Web Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 dicom_query_server.py
  python3 dicom_query_server.py --db /data/meta.db --port 5050
  python3 dicom_query_server.py --model qwen3:14b --no-think
        """,
    )
    parser.add_argument("--db",      type=str, default=str(DEFAULT_DB),
                        help=f"Path to SQLite database (default: {DEFAULT_DB})")
    parser.add_argument("--port",    type=int, default=DEFAULT_PORT,
                        help=f"Port to serve on (default: {DEFAULT_PORT})")
    parser.add_argument("--host",    type=str, default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--model",   type=str, default=DEFAULT_MODEL,
                        help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable Qwen3 thinking mode by default")
    args = parser.parse_args()

    _db_path = Path(args.db)
    _think   = not args.no_think

    if not _db_path.exists():
        log.warning(f"DB not found at {_db_path} — stats endpoint will error until DB is reachable")

    set_model(args.model)

    log.info(f"Starting DICOM Query Server on http://{args.host}:{args.port}")
    log.info(f"  Database : {_db_path}")
    log.info(f"  Model    : {args.model}")
    log.info(f"  Thinking : {'ON' if _think else 'OFF'}")
    log.info(f"  Open     : http://localhost:{args.port}")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()