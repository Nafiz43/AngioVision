"""Shared fixtures for AngioVision tests."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Add module directories to sys.path so tests can import project scripts
_DIRS_TO_ADD = [
    REPO_ROOT / "fine-tuning",
    REPO_ROOT / "utils",
    REPO_ROOT / "frame-processing",
    REPO_ROOT / "batch-processing",
    REPO_ROOT / "slr",
    REPO_ROOT / "configs",
]

for d in _DIRS_TO_ADD:
    d_str = str(d)
    if d_str not in sys.path:
        sys.path.insert(0, d_str)
