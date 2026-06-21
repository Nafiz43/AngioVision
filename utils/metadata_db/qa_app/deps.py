"""
Centralised soft (optional) dependency detection.

The server starts even when image / vector-search / agent dependencies are
missing; the corresponding endpoints degrade gracefully (returning 503 or
empty stats). Each flag below reports whether a dependency group is available,
and the module handles are exposed for the rest of the package to use.
"""

import io
import base64

# ── Image / vector search dependencies ──────────────────────────────────────
try:
    import numpy as np
    from PIL import Image as PilImage
    import chromadb as _chromadb_mod
    IMAGE_DEPS_OK = True
except ImportError:
    IMAGE_DEPS_OK = False
    np            = None
    PilImage      = None
    _chromadb_mod = None

# ── RAD-DINO embedding model ─────────────────────────────────────────────────
try:
    import torch
    from transformers import AutoModel, AutoImageProcessor
    RADDINO_OK = True
except ImportError:
    RADDINO_OK         = False
    torch              = None
    AutoModel          = None
    AutoImageProcessor = None

# ── DICOM pixel reading — for frame extraction ───────────────────────────────
try:
    import pydicom as _pydicom_mod
    PYDICOM_OK = True
except ImportError:
    PYDICOM_OK    = False
    _pydicom_mod  = None

# ── Agentic NL→SQL pipeline (smolagents) ─────────────────────────────────────
try:
    from smolagents import ToolCallingAgent, Tool, OpenAIServerModel
    import openai as _openai_mod  # required by OpenAIServerModel under the hood
    SMOLAGENTS_OK   = True
    _smolagents_err = ""
except ImportError as _sma_err:
    SMOLAGENTS_OK     = False
    _smolagents_err   = str(_sma_err)
    ToolCallingAgent  = None
    OpenAIServerModel = None

    class Tool:  # minimal fallback so SQLQueryTool can still be defined/imported
        pass

__all__ = [
    "io", "base64",
    "IMAGE_DEPS_OK", "np", "PilImage", "_chromadb_mod",
    "RADDINO_OK", "torch", "AutoModel", "AutoImageProcessor",
    "PYDICOM_OK", "_pydicom_mod",
    "SMOLAGENTS_OK", "ToolCallingAgent", "Tool", "OpenAIServerModel",
    "_smolagents_err",
]
