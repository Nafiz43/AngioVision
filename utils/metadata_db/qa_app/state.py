"""
Shared runtime state for the web server.

Replaces the module-level globals of the original single-file server with a
single `state` instance. Values are set once at startup (by `run_server.py`)
and a few are lazily populated singletons / caches guarded by `state.lock`.
"""

import threading
from pathlib import Path
from typing import Any, Dict, Optional

from . import config


class AppState:
    def __init__(self) -> None:
        # ── Configured at startup ────────────────────────────────────────────
        self.db_path: Path        = config.DEFAULT_DB
        self.chromadb_path: Path  = config.DEFAULT_CHROMADB
        self.think: bool          = True
        self.ollama_host: str     = config.DEFAULT_OLLAMA_HOST
        self.agent_max_steps: int = config.DEFAULT_AGENT_MAX_STEPS
        # Heavy-endpoint concurrency gate (see qa_app/concurrency.py)
        self.max_concurrency: int = config.DEFAULT_MAX_CONCURRENCY
        self.max_queue: int       = config.DEFAULT_MAX_QUEUE
        # Pre-flight clarification gate (see qa_app/clarify.py); per-request
        # flags ("clarify"/"skip_clarify") may still override this default.
        self.clarify_enabled: bool = config.CLARIFY_ENABLED

        # ── Concurrency ──────────────────────────────────────────────────────
        self.lock: threading.Lock = threading.Lock()

        # ── Lazy singletons / caches ─────────────────────────────────────────
        self.ollama: Optional[Any]          = None
        self.db_stats_cache: Optional[dict] = None

        # Per-embedding-model caches (keyed by EMBEDDING_MODELS key). Each model
        # has its own loaded embedding function and its own ChromaDB collection.
        self.embedding_models: Dict[str, Any]   = {}
        self.chroma_collections: Dict[str, Any] = {}


state = AppState()
