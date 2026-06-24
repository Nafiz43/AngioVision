"""Tests for utils/04_generate_consolidated_metadata.py

We import individual helper functions from the module, renamed locally
to avoid naming conflicts.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

# The module uses a hyphenated-like naming via the filename prefix "04_",
# so we load it via importlib or rely on conftest path setup.
repo = Path(__file__).resolve().parent.parent
utils_dir = str(repo / "utils")
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

# Import under a short alias
import importlib

_mod = importlib.import_module("04_generate_consolidated_metadata")

_norm_str = _mod._norm_str
_parse_date_to_int = _mod._parse_date_to_int
_parse_time_to_int = _mod._parse_time_to_int
_guess_kv_columns = _mod._guess_kv_columns
_extract_entries_from_wide = _mod._extract_entries_from_wide
_extract_entries_from_rowwise_kv = _mod._extract_entries_from_rowwise_kv


# ---------------------------------------------------------------------------
# _norm_str
# ---------------------------------------------------------------------------
class TestNormStr:
    def test_strips_whitespace(self):
        assert _norm_str("  hello  ") == "hello"

    def test_none_returns_empty(self):
        assert _norm_str(None) == ""

    def test_nan_returns_empty(self):
        assert _norm_str("nan") == ""
        assert _norm_str("NaN") == ""
        assert _norm_str("NAN") == ""

    def test_none_string_returns_empty(self):
        assert _norm_str("none") == ""
        assert _norm_str("None") == ""

    def test_null_returns_empty(self):
        assert _norm_str("null") == ""

    def test_empty_string(self):
        assert _norm_str("") == ""

    def test_normal_value(self):
        assert _norm_str("1.2.3.456") == "1.2.3.456"

    def test_numeric_input(self):
        assert _norm_str(42) == "42"
        assert _norm_str(3.14) == "3.14"


# ---------------------------------------------------------------------------
# _parse_date_to_int
# ---------------------------------------------------------------------------
class TestParseDateToInt:
    def test_dicom_style_date(self):
        assert _parse_date_to_int("20230415") == 20230415

    def test_iso_date(self):
        assert _parse_date_to_int("2023-04-15") == 20230415

    def test_empty_returns_none(self):
        assert _parse_date_to_int("") is None
        assert _parse_date_to_int(None) is None

    def test_nan_returns_none(self):
        assert _parse_date_to_int("nan") is None

    def test_garbage_returns_none(self):
        assert _parse_date_to_int("notadate") is None

    def test_invalid_month(self):
        # 20231315 has month=13, invalid
        assert _parse_date_to_int("20231315") is None

    def test_boundary_year(self):
        assert _parse_date_to_int("19000101") == 19000101
        assert _parse_date_to_int("21001231") == 21001231


# ---------------------------------------------------------------------------
# _parse_time_to_int
# ---------------------------------------------------------------------------
class TestParseTimeToInt:
    def test_hhmmss(self):
        # 12:30:45 -> 12*3600*1e6 + 30*60*1e6 + 45*1e6 = 45045000000
        result = _parse_time_to_int("123045")
        assert result == 12 * 3600 * 1_000_000 + 30 * 60 * 1_000_000 + 45 * 1_000_000

    def test_hhmm(self):
        result = _parse_time_to_int("1230")
        assert result == 12 * 3600 * 1_000_000 + 30 * 60 * 1_000_000

    def test_hh(self):
        result = _parse_time_to_int("12")
        assert result == 12 * 3600 * 1_000_000

    def test_with_fractional(self):
        result = _parse_time_to_int("123045.500000")
        expected = 12 * 3600 * 1_000_000 + 30 * 60 * 1_000_000 + 45 * 1_000_000 + 500000
        assert result == expected

    def test_colon_format(self):
        result = _parse_time_to_int("12:30:45")
        assert result is not None
        assert result > 0

    def test_empty_returns_none(self):
        assert _parse_time_to_int("") is None
        assert _parse_time_to_int(None) is None

    def test_nan_returns_none(self):
        assert _parse_time_to_int("nan") is None

    def test_invalid_hour(self):
        # Hour 25 is invalid
        assert _parse_time_to_int("250000") is None

    def test_invalid_minute(self):
        assert _parse_time_to_int("126000") is None


# ---------------------------------------------------------------------------
# _guess_kv_columns
# ---------------------------------------------------------------------------
class TestGuessKVColumns:
    def test_information_value(self):
        df = pd.DataFrame({"Information": ["a"], "Value": ["b"]})
        result = _guess_kv_columns(df)
        assert result == ("Information", "Value")

    def test_case_insensitive(self):
        df = pd.DataFrame({"information": ["a"], "value": ["b"]})
        result = _guess_kv_columns(df)
        assert result is not None

    def test_two_column_fallback(self):
        df = pd.DataFrame({"Key": ["a"], "Val": ["b"]})
        result = _guess_kv_columns(df)
        assert result == ("Key", "Val")

    def test_three_columns_no_info_value(self):
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        result = _guess_kv_columns(df)
        assert result is None


# ---------------------------------------------------------------------------
# _extract_entries_from_wide
# ---------------------------------------------------------------------------
class TestExtractEntriesFromWide:
    def test_basic_wide_format(self):
        df = pd.DataFrame({
            "StudyInstanceUID": ["study1"],
            "SOPInstanceUID": ["sop1"],
            "AccessionNumber": ["acc1"],
        })
        entries = _extract_entries_from_wide(df, "test.csv")
        assert len(entries) == 1
        assert entries[0]["SOPInstanceUID"] == "sop1"
        assert entries[0]["StudyInstanceUID"] == "study1"

    def test_no_sop_column(self):
        df = pd.DataFrame({"StudyInstanceUID": ["s1"]})
        entries = _extract_entries_from_wide(df, "test.csv")
        assert entries == []

    def test_empty_sop_filtered(self):
        df = pd.DataFrame({
            "SOPInstanceUID": ["", "sop2"],
            "StudyInstanceUID": ["s1", "s2"],
        })
        entries = _extract_entries_from_wide(df, "test.csv")
        assert len(entries) == 1
        assert entries[0]["SOPInstanceUID"] == "sop2"


# ---------------------------------------------------------------------------
# _extract_entries_from_rowwise_kv
# ---------------------------------------------------------------------------
class TestExtractEntriesFromRowwiseKV:
    def test_basic_kv(self):
        df = pd.DataFrame({
            "Information": ["StudyInstanceUID", "SOPInstanceUID", "AccessionNumber"],
            "Value": ["study1", "sop1", "acc1"],
        })
        entries = _extract_entries_from_rowwise_kv(df, "test.csv")
        assert len(entries) == 1
        assert entries[0]["SOPInstanceUID"] == "sop1"

    def test_missing_sop_returns_empty(self):
        df = pd.DataFrame({
            "Information": ["StudyInstanceUID"],
            "Value": ["study1"],
        })
        entries = _extract_entries_from_rowwise_kv(df, "test.csv")
        assert entries == []
