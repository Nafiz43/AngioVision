"""Tests for fine-tuning/calculate_score.py

Imports only the pure utility functions to avoid the module-level
load_validation_csv() call that requires an external settings file.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# We need to patch load_validation_csv at import time because
# calculate_score.py calls it at module level.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _import_calculate_score():
    """Import calculate_score with the settings file lookup stubbed out."""
    repo = Path(__file__).resolve().parent.parent
    ft_dir = str(repo / "fine-tuning")
    if ft_dir not in sys.path:
        sys.path.insert(0, ft_dir)

    # Stub load_validation_csv so it doesn't require the real settings file
    stub_module_code = """
import argparse, sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def load_validation_csv(settings_py):
    return "/tmp/fake_gt.csv"

SETTINGS_FILE = "/tmp/fake_settings.py"
DEFAULT_GT_PATH = "/tmp/fake_gt.csv"
"""

    # Create a minimal mock for the settings loading
    import types

    # We'll import the functions we need by exec'ing the file with mocks
    calc_path = repo / "fine-tuning" / "calculate_score.py"
    source = calc_path.read_text()

    # Replace the problematic module-level call
    source = source.replace(
        'DEFAULT_GT_PATH = load_validation_csv(SETTINGS_FILE)',
        'DEFAULT_GT_PATH = "/tmp/fake_gt.csv"',
    )
    source = source.replace(
        "SETTINGS_FILE = \"/data/Deep_Angiography/AngioVision/configs/settings.py\"",
        "SETTINGS_FILE = \"/tmp/fake_settings.py\"",
    )

    module = types.ModuleType("calculate_score")
    module.__file__ = str(calc_path)
    exec(compile(source, str(calc_path), "exec"), module.__dict__)
    sys.modules["calculate_score"] = module


def _mod():
    return sys.modules["calculate_score"]


# ---------------------------------------------------------------------------
# normalize_str_series
# ---------------------------------------------------------------------------
class TestNormalizeStrSeries:
    def test_strips_and_lowercases(self):
        s = pd.Series(["  Yes ", "NO", " Maybe "])
        result = _mod().normalize_str_series(s)
        assert list(result) == ["yes", "no", "maybe"]

    def test_handles_nan(self):
        s = pd.Series([None, float("nan"), "Hello"])
        result = _mod().normalize_str_series(s)
        assert result.iloc[0] == ""
        assert result.iloc[1] == ""
        assert result.iloc[2] == "hello"

    def test_numeric_values(self):
        s = pd.Series([1, 2, 3])
        result = _mod().normalize_str_series(s)
        assert list(result) == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# keep_yes_no
# ---------------------------------------------------------------------------
class TestKeepYesNo:
    def test_filters_to_yes_no(self):
        df = pd.DataFrame({"Answer": ["yes", "no", "maybe", "YES", "No"]})
        result = _mod().keep_yes_no(df, "Answer", "TEST")
        assert len(result) == 4  # "maybe" is filtered out

    def test_removes_non_yes_no(self):
        df = pd.DataFrame({"Answer": ["yes", "unknown", "no", "idk"]})
        result = _mod().keep_yes_no(df, "Answer", "TEST")
        assert len(result) == 2

    def test_missing_column_returns_empty(self):
        df = pd.DataFrame({"Other": [1, 2, 3]})
        result = _mod().keep_yes_no(df, "Answer", "TEST")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# require_cols
# ---------------------------------------------------------------------------
class TestRequireCols:
    def test_all_present(self):
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        assert _mod().require_cols(df, ["A", "B"], "TAG") is True

    def test_missing_cols(self):
        df = pd.DataFrame({"A": [1]})
        assert _mod().require_cols(df, ["A", "X"], "TAG") is False


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------
class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = pd.Series([1, 0, 1, 0])
        m = _mod().compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0
        assert m["reason"] == "ok"

    def test_all_wrong(self):
        y_true = pd.Series([1, 1, 0, 0])
        y_pred = pd.Series([0, 0, 1, 1])
        m = _mod().compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0
        assert m["tp"] == 0
        assert m["tn"] == 0

    def test_empty_input(self):
        m = _mod().compute_metrics(pd.Series([], dtype=int), pd.Series([], dtype=int))
        assert m["reason"] == "no_rows_for_scoring"

    def test_confusion_matrix_values(self):
        y_true = pd.Series([1, 1, 0, 0, 1])
        y_pred = pd.Series([1, 0, 0, 1, 1])
        m = _mod().compute_metrics(y_true, y_pred)
        assert m["tp"] == 2
        assert m["tn"] == 1
        assert m["fp"] == 1
        assert m["fn"] == 1
        assert m["reason"] == "ok"


# ---------------------------------------------------------------------------
# empty_metrics_dict
# ---------------------------------------------------------------------------
class TestEmptyMetricsDict:
    def test_structure(self):
        m = _mod().empty_metrics_dict("test_reason")
        assert m["tp"] == 0
        assert m["accuracy"] == 0.0
        assert m["reason"] == "test_reason"


# ---------------------------------------------------------------------------
# build_zero_groups
# ---------------------------------------------------------------------------
class TestBuildZeroGroups:
    def test_all_groups_present(self):
        groups = _mod().build_zero_groups("test")
        expected = {"ORIGINAL", "FLIPPED", "ALL_YES", "ALL_NO", "RANDOM"}
        assert set(groups.keys()) == expected
        for v in groups.values():
            assert v["reason"] == "test"


# ---------------------------------------------------------------------------
# safe_write_csv / safe_read_csv (with temp files)
# ---------------------------------------------------------------------------
class TestSafeIO:
    def test_safe_write_and_read(self, tmp_path):
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        out = str(tmp_path / "out.csv")
        assert _mod().safe_write_csv(df, out, "TEST") is True

        loaded = _mod().safe_read_csv(out, "TEST")
        assert loaded is not None
        assert len(loaded) == 2

    def test_safe_read_missing_file(self):
        result = _mod().safe_read_csv("/nonexistent/path.csv", "TEST")
        assert result is None
