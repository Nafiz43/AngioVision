================================================================================
 AngioVision · metadata_db
 DICOM metadata ingestion + agentic query web app
================================================================================

This directory holds two applications, each refactored from a single legacy
file into a clean, maintainable Python package:

  1. INGESTION  — walk a DICOM tree, parse metadata into SQLite, ingest
                  radiology reports, and embed labeled sequences into ChromaDB.
                    Package:     ingestion/
                    Entry point: run_ingest.py
                    Legacy file: ingest.py        (kept for reference)

  2. QUERY APP  — a Flask web server with an agentic natural-language -> SQL
                  pipeline (smolagents + Ollama) and image RAG (RAD-DINO +
                  ChromaDB), plus a browser UI.
                    Package:     qa_app/
                    Entry point: run_server.py
                    Legacy file: qa_pipe.py       (kept for reference)

NOTE: The original single-file scripts (ingest.py, qa_pipe.py) are intentionally
      preserved. The new packages are drop-in functional replacements; prefer the
      run_ingest.py / run_server.py entry points going forward.

For step-by-step run instructions, see:  how-to-run-query.txt


--------------------------------------------------------------------------------
 DIRECTORY LAYOUT
--------------------------------------------------------------------------------

  metadata_db/
  ├── run_ingest.py            CLI entry point for the ingestion pipeline
  ├── run_server.py            CLI entry point for the query web server
  ├── requirements.txt         All Python dependencies for both apps
  ├── README.txt               This file
  ├── how-to-run-query.txt     How to run the web app (and ingestion)
  │
  ├── ingestion/               INGESTION PACKAGE
  │   ├── __init__.py            Exposes run_ingestion() orchestrator
  │   ├── config.py              Default paths, tuning constants, dep flags
  │   ├── schema.py              SQLite schema, column list, INSERT statements
  │   ├── dicom_parser.py        DICOM parsing, file discovery, frame extraction
  │   ├── store.py               SQLite connect, batch insert, summary report
  │   ├── reports.py             Radiology report CSV ingestion
  │   ├── labels.py              Labeled DSA sequence CSV loading + filtering
  │   ├── embeddings.py          RAD-DINO model load + ChromaDB collection setup
  │   ├── images.py              End-to-end image -> embedding -> ChromaDB
  │   └── pipeline.py            run_ingestion(): orchestrates all stages
  │
  └── qa_app/                  QUERY WEB APP PACKAGE
      ├── __init__.py            Flask app factory: create_app()
      ├── config.py              Default paths and server constants
      ├── state.py               Shared runtime state (AppState singleton)
      ├── deps.py                Centralised optional-dependency detection
      ├── prompts.py             LLM system prompts + DB schema context
      ├── db.py                  SQLite access, cached stats, SQL cleanup
      ├── llm.py                 Ollama chat model management + synthesis
      ├── embeddings.py          RAD-DINO query-time embedding function
      ├── images.py              ChromaDB access, image decode, enrichment, frames
      ├── agent.py               smolagents NL->SQL ToolCallingAgent + SSE events
      ├── routes.py              All HTTP routes (Flask Blueprint)
      ├── templates/
      │   └── index.html         Browser UI (Jinja2 template)
      └── static/
          ├── css/styles.css     Extracted stylesheet
          └── js/app.js          Extracted client-side JavaScript


--------------------------------------------------------------------------------
 WHAT CHANGED IN THE REFACTOR
--------------------------------------------------------------------------------

  - Monolithic files split into single-responsibility modules.
  - Embedded HTML/CSS/JS pulled out of qa_pipe.py into standard Flask
    templates/ and static/ locations (no more 1,200-line string literal).
  - Module-level globals replaced by a single qa_app.state.AppState instance.
  - Flask app created via a create_app() factory + Blueprint (testable, no
    import-time side effects).
  - Optional dependencies centralised in qa_app/deps.py and ingestion/config.py
    so the server still starts when image / agent extras are missing
    (those endpoints return 503 instead of crashing).
  - All public behaviour, CLI flags, API routes, and SSE event names are
    preserved 1:1 with the legacy scripts.


--------------------------------------------------------------------------------
 QUICK START
--------------------------------------------------------------------------------

  pip install -r requirements.txt

  # Build the databases (metadata + reports + image embeddings)
  python3 run_ingest.py

  # Launch the web UI (needs a running Ollama server)
  python3 run_server.py
  # open http://localhost:5050

See how-to-run-query.txt for the full guide, CLI options, and troubleshooting.
