"""Tests for utils/17_report_augmentor.py"""

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

repo = Path(__file__).resolve().parent.parent
utils_dir = str(repo / "utils")
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

_mod = importlib.import_module("17_report_augmentor")

clean_text = _mod.clean_text
normalize_generated_text = _mod.normalize_generated_text
validate_generated_output = _mod.validate_generated_output
build_prompt = _mod.build_prompt
reorder_columns = _mod.reorder_columns
format_block = _mod.format_block


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------
class TestCleanText:
    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_normalizes_tabs_spaces(self):
        assert clean_text("a\t\t  b") == "a b"

    def test_normalizes_newlines(self):
        result = clean_text("a\r\nb\rc")
        assert "\r" not in result
        assert "a\nb\nc" == result

    def test_collapses_multiple_newlines(self):
        result = clean_text("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_integer_input(self):
        assert clean_text(123) == "123"


# ---------------------------------------------------------------------------
# normalize_generated_text
# ---------------------------------------------------------------------------
class TestNormalizeGeneratedText:
    def test_strips_rephrased_prefix(self):
        result = normalize_generated_text("Rephrased report: Some text here")
        assert result == "Some text here"

    def test_strips_rewritten_prefix(self):
        result = normalize_generated_text("Rewritten Report: Text")
        assert result == "Text"

    def test_strips_surrounding_quotes(self):
        result = normalize_generated_text('"Some quoted text"')
        assert result == "Some quoted text"

    def test_strips_single_quotes(self):
        result = normalize_generated_text("'Some text'")
        assert result == "Some text"

    def test_no_change_for_clean_text(self):
        text = "Normal report text without prefix"
        assert normalize_generated_text(text) == text

    def test_short_text_not_unquoted(self):
        # Single character shouldn't be unquoted
        result = normalize_generated_text("x")
        assert result == "x"


# ---------------------------------------------------------------------------
# validate_generated_output
# ---------------------------------------------------------------------------
class TestValidateGeneratedOutput:
    def test_valid_output(self):
        original = "Patient presented with chest pain and underwent angiography."
        generated = "The patient came in with chest pain and had an angiography procedure."
        valid, reason = validate_generated_output(original, generated)
        assert valid is True
        assert reason == "OK"

    def test_empty_output_invalid(self):
        valid, reason = validate_generated_output("some original", "   ")
        assert valid is False
        assert "empty" in reason.lower()

    def test_bad_prefix_thinking(self):
        valid, reason = validate_generated_output("original", "Thinking about this...")
        assert valid is False
        assert "prefix" in reason.lower()

    def test_bad_prefix_here_is(self):
        valid, reason = validate_generated_output("original", "Here is the rephrased report...")
        assert valid is False

    def test_bad_prefix_certainly(self):
        valid, reason = validate_generated_output("original", "Certainly, here is...")
        assert valid is False

    def test_too_short_output(self):
        original = "This is a moderately long radiology report with many words " * 5
        generated = "Short"
        valid, reason = validate_generated_output(original, generated)
        assert valid is False
        assert "short" in reason.lower()

    def test_borderline_length_passes(self):
        original = "A B C D E F G H I J K L M N O"
        # At least 35% of 15 words = ~5 words, and min 5
        generated = "A B C D E F"
        valid, _ = validate_generated_output(original, generated)
        assert valid is True


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------
class TestBuildPrompt:
    def test_contains_report(self):
        prompt = build_prompt("Some report text", 1)
        assert "Some report text" in prompt

    def test_contains_variation_instruction(self):
        prompt = build_prompt("report", 2)
        assert "reorganize" in prompt.lower() or "structure" in prompt.lower()

    def test_variant_idx_out_of_range_uses_fallback(self):
        prompt = build_prompt("report", 99)
        assert "structural changes" in prompt.lower()

    def test_preserves_rules(self):
        prompt = build_prompt("text", 1)
        assert "clinical meaning" in prompt.lower()


# ---------------------------------------------------------------------------
# reorder_columns
# ---------------------------------------------------------------------------
class TestReorderColumns:
    def test_type_after_report(self):
        df = pd.DataFrame({"Anon Acc #": [1], "radrpt": ["text"], "Type": ["Original"], "Other": [0]})
        result = reorder_columns(df)
        cols = list(result.columns)
        assert cols.index("Type") == cols.index("radrpt") + 1

    def test_no_type_column(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        result = reorder_columns(df)
        assert list(result.columns) == ["A", "B"]

    def test_no_report_column(self):
        df = pd.DataFrame({"Type": ["x"], "A": [1]})
        result = reorder_columns(df)
        assert list(result.columns)[0] == "Type"


# ---------------------------------------------------------------------------
# format_block
# ---------------------------------------------------------------------------
class TestFormatBlock:
    def test_wraps_content(self):
        result = format_block("TITLE", "content here")
        assert "TITLE START" in result
        assert "TITLE END" in result
        assert "content here" in result
