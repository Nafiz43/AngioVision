#!/usr/bin/env python3
"""
DICOM Neo4j Query Pipeline
Natural language → Cypher → Neo4j → LLM synthesis

Flow:
  User query
    → Qwen3:35b (text-to-Cypher)
    → Neo4j execution
    → Qwen3:35b (result synthesis)
    → Human-readable answer

Usage:
    python3 qa_pipeline.py                    # interactive REPL
    python3 qa_pipeline.py -q "your question" # single shot
    python3 qa_pipeline.py --no-think         # disable Qwen3 thinking mode
    python3 qa_pipeline.py --examples         # print example questions

Requirements:
    pip install neo4j requests
    Ollama running with: ollama pull qwen3:35b
"""

import re
import sys
import json
import logging
import argparse
import requests
from typing import Optional

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: pip install neo4j")
    sys.exit(1)

try:
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
except ImportError:
    NEO4J_URI      = "bolt://localhost:7687"
    NEO4J_USER     = "neo4j"
    NEO4J_PASSWORD = "neo4j"
    NEO4J_DATABASE = "neo4j"

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen3:35b"

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Graph schema context (injected into every Cypher-gen prompt) ──────────────
SCHEMA_CONTEXT = """
You are an expert Neo4j Cypher query generator for a DICOM medical imaging database.

## Graph Schema

Nodes and their key properties:

(:AccessionNumber {value})
    — Primary key anchor. Each accession links to one Study.

(:Patient {patient_id, patient_name, patient_sex, patient_age,
           pregnancy_status, identity_removed, deidentification})

(:Study {study_instance_uid, accession_number, study_date, study_time,
         study_description, referring_physician,
         requested_procedure_desc, performed_step_date,
         performed_step_time, performed_step_desc})

(:Series {series_instance_uid, series_number, series_date, series_time,
          series_description, modality, protocol_name, acquisition_number})

(:Instance {sop_instance_uid, sop_class_uid, instance_number,
            image_type, acquisition_date, acquisition_time,
            content_date, content_time,
            rows, columns, bits_allocated, bits_stored, high_bit,
            samples_per_pixel, pixel_representation,
            photometric_interpretation,
            number_of_frames, frame_count, frame_time, cine_rate,
            images_in_acquisition, representative_frame_number,
            start_trim, stop_trim, recommended_display_frame_rate,
            kvp, exposure_time, xray_tube_current, avg_pulse_width,
            radiation_setting, radiation_mode, dose_product,
            distance_source_to_detector, distance_source_to_patient,
            est_magnification_factor, intensifier_size,
            imager_pixel_spacing, focal_spots,
            positioner_motion, positioner_primary_angle, positioner_secondary_angle,
            patient_position,
            window_center, window_width, voi_lut_function,
            lossy_image_compression, longitudinal_temporal_info,
            pixel_intensity_relationship,
            contrast_bolus_agent, contrast_bolus_ingredient,
            manufacturer, manufacturer_model, station_name,
            software_versions, device_serial_number,
            detector_id, detector_description,
            specific_character_set, source_file, source_path})

## Relationships

(Patient)-[:UNDERWENT]->(Study)
(AccessionNumber)-[:BELONGS_TO]->(Study)
(Study)-[:HAS_SERIES]->(Series)
(Series)-[:HAS_INSTANCE]->(Instance)

## Query Rules

1. Always start traversal from (:AccessionNumber) or (:Patient) unless aggregating globally.
2. For date comparisons, study_date / acquisition_date are stored as strings in YYYYMMDD format.
3. For numeric comparisons (frame_count, kvp, rows, etc.) use toInteger() or toFloat() if needed.
4. Prefer LIMIT clauses — default to LIMIT 25 unless the user asks for counts/aggregates.
5. Use toLower() for case-insensitive text matching.
6. Return only what is asked — do not over-fetch.
7. source_path gives the full filesystem path to the .dcm file.
8. modality 'XA' = X-ray Angiography; DSA = Digital Subtraction Angiography (usually in series_description).
9. Output ONLY the Cypher query — no explanation, no markdown fences, no preamble.
"""

SYNTHESIS_SYSTEM = """
You are a clinical informatics assistant helping radiologists and researchers query a DICOM database.
You receive:
  1. The original natural language question
  2. The Cypher query that was executed
  3. The raw results from Neo4j (as JSON)

Your job: produce a clear, concise, human-readable answer.
- Summarise patterns and counts when there are many results.
- If the result set is small (≤10 rows), present each item clearly.
- Highlight clinically or technically interesting findings.
- If results are empty, say so and suggest why the query may have returned nothing.
- Do NOT re-state the Cypher query unless the user would benefit from seeing it.
- Be direct and informative. Use markdown tables when they aid readability.
"""

# ── Ollama helper ─────────────────────────────────────────────────────────────
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

OLLAMA_MODEL = "qwen3.6:35b"

ollama = ChatOllama(
    model=OLLAMA_MODEL)

def ollama_chat(messages: list[dict], think: bool = True) -> str:
    """Call Ollama using LangChain invoke()."""

    lc_messages = []

    for m in messages:
        role = m["role"]
        content = m["content"]

        if role == "system":
            lc_messages.append(SystemMessage(content=content))

        elif role == "user":
            if not think:
                content = "/no_think\n" + content

            lc_messages.append(HumanMessage(content=content))

    try:
        response = ollama.invoke(
            lc_messages,
        )

        content = response.content

        # Strip <think>...</think> blocks
        content = re.sub(
            r"<think>.*?</think>",
            "",
            content,
            flags=re.DOTALL,
        ).strip()

        return content

    except Exception as e:
        raise RuntimeError(f"Ollama invoke failed: {e}")


# ── Cypher generation ─────────────────────────────────────────────────────────
def generate_cypher(user_query: str, think: bool = True) -> str:
    """Convert natural language to Cypher via Qwen3."""
    log.info("Generating Cypher from query...")
    messages = [
        {"role": "system", "content": SCHEMA_CONTEXT},
        {"role": "user",   "content": f"Convert this question to a Cypher query:\n\n{user_query}"},
    ]
    raw = ollama_chat(messages, think=think)

    # Strip any accidental markdown fences
    cypher = re.sub(r"```(?:cypher|sql)?\s*", "", raw)
    cypher = re.sub(r"```", "", cypher).strip()

    log.info(f"Generated Cypher:\n{cypher}")
    return cypher


# ── Neo4j execution ───────────────────────────────────────────────────────────
def run_cypher(cypher: str, driver) -> list[dict]:
    """Execute a Cypher query and return results as a list of dicts."""
    with driver.session(database=NEO4J_DATABASE) as session:
        result  = session.run(cypher)
        records = []
        for rec in result:
            row = {}
            for key in rec.keys():
                val = rec[key]
                if hasattr(val, "_properties"):   # Node
                    row[key] = dict(val._properties)
                elif hasattr(val, "items"):        # dict-like
                    row[key] = dict(val)
                else:
                    row[key] = val
            records.append(row)
    log.info(f"Neo4j returned {len(records)} record(s).")
    return records


# ── Result synthesis ──────────────────────────────────────────────────────────
def synthesize_answer(user_query: str, cypher: str,
                      results: list[dict], think: bool = True) -> str:
    """Summarise Neo4j results into a human-readable answer via Qwen3."""
    log.info("Synthesizing answer from results...")

    display_results = results[:100]
    truncated       = len(results) > 100
    result_str      = json.dumps(display_results, indent=2, default=str)
    if truncated:
        result_str += f"\n\n[... {len(results) - 100} additional rows truncated]"

    user_content = f"""
Question: {user_query}

Cypher executed:
{cypher}

Neo4j results ({len(results)} total rows):
{result_str}
""".strip()

    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM},
        {"role": "user",   "content": user_content},
    ]
    return ollama_chat(messages, think=think)


# ── Full pipeline ─────────────────────────────────────────────────────────────
def query_pipeline(user_query: str, driver,
                   think: bool = True,
                   verbose: bool = False) -> tuple[str, str]:
    """
    End-to-end pipeline:
      user_query → Cypher → Neo4j → synthesis → (answer, cypher)
    Returns (answer, cypher) tuple so callers can display the Cypher if needed.
    """
    # Step 1: Text → Cypher
    try:
        cypher = generate_cypher(user_query, think=think)
    except RuntimeError as e:
        return f"[Cypher generation failed] {e}", ""

    if verbose:
        print(f"\n{'─'*60}")
        print("GENERATED CYPHER:")
        print(cypher)
        print('─'*60)

    # Step 2: Execute Cypher
    try:
        results = run_cypher(cypher, driver)
    except Exception as e:
        log.warning(f"Cypher execution error: {e}")
        return (
            f"The generated Cypher query failed to execute.\n\n"
            f"**Cypher:**\n```\n{cypher}\n```\n\n"
            f"**Error:** {e}\n\n"
            f"Try rephrasing your question or check the schema."
        ), cypher

    if verbose:
        print(f"\nNeo4j returned {len(results)} row(s).")

    # Step 3: Synthesise answer
    try:
        answer = synthesize_answer(user_query, cypher, results, think=think)
    except RuntimeError as e:
        return f"[Synthesis failed] {e}", cypher

    return answer, cypher


# ── Interactive REPL ──────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        DICOM Knowledge Graph Query Pipeline                  ║
║        Model : {model:<46}║
║        Neo4j : {db:<46}║
╚══════════════════════════════════════════════════════════════╝
Type your question in plain English.  Commands:
  :quit / :q   — exit
  :cypher      — show last generated Cypher
  :verbose     — toggle verbose mode (show Cypher each time)
  :think       — toggle Qwen3 thinking mode (on by default)
  :help        — show example questions
"""

EXAMPLE_QUESTIONS = """
EXAMPLE QUESTIONS YOU CAN ASK:
─────────────────────────────────────────────────────────────
EXPLORATION & COUNTS
  1.  How many unique patients are in the database?
  2.  How many studies, series, and instances do we have in total?
  3.  What modalities are present and how many instances does each have?
  4.  List all distinct study descriptions in the database.
  5.  Which manufacturers' equipment was used, and how many series each?

ACCESSION / PATIENT LOOKUP
  6.  What images are associated with accession number 0BrnGBKrkm?
  7.  Show me all series for patient ID eYUFraGrHn.
  8.  Find all studies with study description containing 'EMBOLIZATION'.
  9.  Which patients had more than 5 series in a single study?

DSA / ANGIOGRAPHY SPECIFIC
  10. List all DSA series with more than 20 frames, showing their source path.
  11. Which series have 'DSA' in their description? Group by protocol name.
  12. Find all instances where contrast agent is IODINE.
  13. Show XA modality instances acquired in 2009, sorted by acquisition date.
  14. What is the average frame count across all DSA series?

ACQUISITION PARAMETERS
  15. Find instances where KVP was above 80.
  16. List instances with radiation mode PULSED and exposure time over 1000ms.
  17. Which instances have the highest dose product? Show top 10.
  18. Find all instances with positioner primary angle of 0 degrees.
  19. Show me instances with image resolution 1024x1024.

EQUIPMENT & STATION
  20. Which station names appear in the database and how many series each?
  21. List all distinct Siemens model names used in acquisitions.
  22. Find all instances captured on station IR1625ZEEGO.
  23. How many series were acquired using the AXIOM-Artis system?

TEMPORAL
  24. How many studies were performed in each year? Order by year.
  25. Find all acquisitions that happened between 2009 and 2012.
  26. Which patient had the earliest study date in the dataset?

FILE PATHS
  27. Give me the source_path for all instances in accession 0BrnGBKrkm.
  28. How many .dcm files are stored per series for the study
      with description 'IR SPLENIC EMBOLIZATION'?
─────────────────────────────────────────────────────────────
"""


def repl(driver, think: bool = True, verbose: bool = False):
    print(BANNER.format(model=OLLAMA_MODEL, db=NEO4J_URI))
    last_cypher = ""

    while True:
        try:
            user_input = input("\n❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input.lower() in (":quit", ":q", "exit", "quit"):
            print("Goodbye.")
            break

        elif user_input.lower() == ":cypher":
            if last_cypher:
                print(f"\n{last_cypher}")
            else:
                print("No query run yet.")
            continue

        elif user_input.lower() == ":verbose":
            verbose = not verbose
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
            continue

        elif user_input.lower() == ":think":
            think = not think
            print(f"Thinking mode: {'ON' if think else 'OFF'}")
            continue

        elif user_input.lower() == ":help":
            print(EXAMPLE_QUESTIONS)
            continue

        # ── Run pipeline ──────────────────────────────────────────────────────
        print("\n⏳ Processing...\n")
        answer, last_cypher = query_pipeline(
            user_input, driver, think=think, verbose=verbose
        )

        print("\n" + "─"*60)
        print("ANSWER:")
        print("─"*60)
        print(answer)
        print("─"*60)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global OLLAMA_MODEL   # must be first line in main() before any reference

    parser = argparse.ArgumentParser(
        description="DICOM Neo4j natural-language query pipeline."
    )
    parser.add_argument("-q", "--query",    type=str,
                        help="Single query (non-interactive).")
    parser.add_argument("--uri",      type=str, default=NEO4J_URI)
    parser.add_argument("--user",     type=str, default=NEO4J_USER)
    parser.add_argument("--password", type=str, default=NEO4J_PASSWORD)
    parser.add_argument("--database", type=str, default=NEO4J_DATABASE)
    parser.add_argument("--model",    type=str, default=OLLAMA_MODEL,
                        help="Ollama model name (default: qwen3:35b)")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable Qwen3 extended thinking mode.")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print generated Cypher before answer.")
    parser.add_argument("--examples", action="store_true",
                        help="Print example questions and exit.")
    args = parser.parse_args()

    if args.examples:
        print(EXAMPLE_QUESTIONS)
        return

    # Apply model override after parsing
    OLLAMA_MODEL = args.model

    think   = not args.no_think
    verbose = args.verbose

    # ── Connect Neo4j ─────────────────────────────────────────────────────────
    log.info(f"Connecting to Neo4j at {args.uri} ...")
    try:
        driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
        driver.verify_connectivity()
        # Pre-flight auth check — catches CredentialsExpired before any query
        with driver.session(database="system") as s:
            s.run("SHOW DATABASES").consume()
        log.info("Neo4j connection and auth OK.")
    except Exception as e:
        log.error(f"Neo4j connection failed: {e}")
        sys.exit(1)

    try:
        if args.query:
            # ── Single-shot mode ──────────────────────────────────────────────
            print(f"\nQ: {args.query}\n")
            answer, cypher = query_pipeline(
                args.query, driver, think=think, verbose=verbose
            )
            if verbose and cypher:
                print(f"CYPHER:\n{cypher}\n")
            print("─"*60)
            print(answer)
            print("─"*60)
        else:
            # ── Interactive REPL ──────────────────────────────────────────────
            repl(driver, think=think, verbose=verbose)
    finally:
        driver.close()


if __name__ == "__main__":
    main()