"""
AngioVision DICOM Query web application.

Serves the browser UI and a REST API bridging:
  - Natural language → SQL, via an agentic smolagents ToolCallingAgent pipeline
  - SQL → SQLite execution
  - Image RAG: base64 image → RAD-DINO embedding → ChromaDB similarity search
  - DICOM frame thumbnail / full-resolution rendering

Use `create_app()` to build the configured Flask app (see `run_server.py`).
Runtime configuration (db path, ChromaDB path, Ollama host, etc.) is set on the
shared `qa_app.state.state` object before calling `create_app()`.
"""

import sys

try:
    from flask import Flask
    from flask_cors import CORS
except ImportError:
    print("ERROR: pip install flask flask-cors")
    sys.exit(1)

from .state import state

__all__ = ["create_app", "state"]


def create_app() -> "Flask":
    """Build the Flask app: enable CORS and register the API/UI blueprint."""
    app = Flask(__name__)   # templates → qa_app/templates, static → qa_app/static
    CORS(app)

    from .routes import bp
    app.register_blueprint(bp)

    return app
