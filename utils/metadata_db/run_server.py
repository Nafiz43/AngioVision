#!/usr/bin/env python3
"""
DICOM Query Web Server  (CLI entry point)

Serves the browser UI and exposes a REST API that bridges:
  - Natural language → SQL, via an AGENTIC pipeline (smolagents ToolCallingAgent)
    that can call the database as many times as it needs — exploring the schema,
    checking real column values, recovering from SQL errors, and refining its
    query — before producing a final answer.
  - SQL → SQLite execution
  - Image RAG (RAD-DINO embeddings): POST /api/image-query accepts a base64 image,
    queries ChromaDB for the top visually similar DICOM sequences, enriches each
    hit with SQLite metadata + report excerpts, and streams a synthesised answer.
  - GET /api/thumbnail and /api/frame render DICOM frames as PNG.

This is the structured replacement for the legacy single-file `qa_pipe.py`; all
logic now lives in the `qa_app/` package.

Usage:
    python3 run_server.py
    python3 run_server.py --db /path/to/dicom_staging.db --port 5050
    python3 run_server.py --model qwen3:14b --no-think
    python3 run_server.py --chromadb /path/to/chromadb
    python3 run_server.py --ollama-host http://localhost:11434 --agent-max-steps 10

Requires: pip install flask flask-cors langchain-ollama 'smolagents[openai]'
          (image RAG also needs: chromadb pillow numpy torch transformers pydicom)
"""

import logging
import argparse
from pathlib import Path

from qa_app import create_app, config, deps, state
from qa_app.llm import set_model

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DICOM Query Web Server (Agentic NL→SQL via smolagents + Image RAG with RAD-DINO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_server.py
  python3 run_server.py --db /data/meta.db --port 5050
  python3 run_server.py --model qwen3:14b --no-think
  python3 run_server.py --chromadb /data/AngioVision/chromadb
  python3 run_server.py --ollama-host http://localhost:11434 --agent-max-steps 12
        """,
    )
    parser.add_argument(
        "--db", type=str, default=str(config.DEFAULT_DB),
        help=f"SQLite database path (default: {config.DEFAULT_DB})",
    )
    parser.add_argument(
        "--chromadb", type=str, default=str(config.DEFAULT_CHROMADB),
        help=f"ChromaDB persistence directory (default: {config.DEFAULT_CHROMADB})",
    )
    parser.add_argument(
        "--port", type=int, default=config.DEFAULT_PORT,
        help=f"Port to serve on (default: {config.DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--model", type=str, default=config.DEFAULT_MODEL,
        help=f"Ollama model (default: {config.DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-think", action="store_true",
        help="Disable Qwen3 thinking mode by default",
    )
    parser.add_argument(
        "--ollama-host", type=str, default=config.DEFAULT_OLLAMA_HOST,
        help=(
            f"Ollama server base URL, used by the smolagents NL→SQL agent "
            f"via its OpenAI-compatible API (default: {config.DEFAULT_OLLAMA_HOST})"
        ),
    )
    parser.add_argument(
        "--agent-max-steps", type=int, default=config.DEFAULT_AGENT_MAX_STEPS,
        help=(
            f"Max sql_query tool calls the NL→SQL agent may make per "
            f"question before it must answer (default: {config.DEFAULT_AGENT_MAX_STEPS})"
        ),
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=config.DEFAULT_MAX_CONCURRENCY,
        help=(
            f"Max heavy jobs (/api/query + /api/image-query combined) that may "
            f"run at once; the rest queue FIFO (default: {config.DEFAULT_MAX_CONCURRENCY})"
        ),
    )
    parser.add_argument(
        "--max-queue", type=int, default=config.DEFAULT_MAX_QUEUE,
        help=(
            f"Max requests allowed to wait in the queue before new ones are "
            f"rejected with a 'busy' message (default: {config.DEFAULT_MAX_QUEUE})"
        ),
    )
    parser.add_argument(
        "--no-clarify", action="store_true",
        help=(
            "Disable the pre-flight clarification gate; answer directly even when "
            "the request is ambiguous (default: clarification enabled)"
        ),
    )
    args = parser.parse_args()

    # ── Configure shared runtime state ───────────────────────────────────────
    state.db_path         = Path(args.db)
    state.chromadb_path   = Path(args.chromadb)
    state.think           = not args.no_think
    state.ollama_host     = args.ollama_host.rstrip("/")
    state.agent_max_steps = args.agent_max_steps
    state.max_concurrency = max(1, args.max_concurrency)
    state.max_queue       = max(0, args.max_queue)
    state.clarify_enabled = not args.no_clarify

    if not state.db_path.exists():
        log.warning(f"DB not found at {state.db_path} — stats will error until DB is reachable")

    if not deps.RADDINO_OK:
        log.warning(
            "RAD-DINO dependencies not installed (torch, transformers). "
            "Install with: pip install torch transformers"
        )

    if not deps.IMAGE_DEPS_OK:
        log.warning(
            "Image RAG dependencies not installed — /api/image-query will return 503. "
            "Install with: pip install chromadb pillow numpy"
        )
    else:
        log.info(f"ChromaDB path  : {state.chromadb_path}")

    if not deps.SMOLAGENTS_OK:
        log.warning(
            "smolagents not installed — /api/query (NL→SQL) will return 503. "
            "Install with: pip install 'smolagents[openai]'"
        )
    else:
        log.info(
            f"Agentic NL→SQL : smolagents ToolCallingAgent via Ollama @ {state.ollama_host} "
            f"(max_steps={state.agent_max_steps})"
        )

    set_model(args.model)

    app = create_app()

    log.info(f"Starting DICOM Query Server on http://{args.host}:{args.port}")
    log.info(f"  Database   : {state.db_path}")
    log.info(f"  Model      : {args.model}")
    log.info(f"  Thinking   : {'ON' if state.think else 'OFF'}")
    log.info(f"  Embeddings : RAD-DINO (if available)")
    log.info(f"  Concurrency: {state.max_concurrency} heavy job(s) at once, queue up to {state.max_queue}")
    log.info(f"  Open       : http://localhost:{args.port}")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
