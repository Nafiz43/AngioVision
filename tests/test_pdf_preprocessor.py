"""Tests for slr/00_pdf_preprocessor.py

Only imports the pure text-processing functions; skips PDF I/O.
"""

import importlib
import sys
from pathlib import Path

import pytest

repo = Path(__file__).resolve().parent.parent
slr_dir = str(repo / "slr")
if slr_dir not in sys.path:
    sys.path.insert(0, slr_dir)

# Try importing; skip if pymupdf4llm is not installed
try:
    _mod = importlib.import_module("00_pdf_preprocessor")
except ImportError:
    pytest.skip("pymupdf4llm not installed, skipping pdf preprocessor tests", allow_module_level=True)

normalize_text = _mod.normalize_text
remove_cid_references = _mod.remove_cid_references
remove_header_footer_noise = _mod.remove_header_footer_noise
remove_pipe_flood = _mod.remove_pipe_flood
truncate_at_trailing_sections = _mod.truncate_at_trailing_sections
clean_excessive_whitespace = _mod.clean_excessive_whitespace
_is_valid_heading_format = _mod._is_valid_heading_format
extract_year_from_text = _mod.extract_year_from_text
postprocess = _mod.postprocess
next_number = _mod.next_number


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------
class TestNormalizeText:
    def test_replaces_ligatures(self):
        assert "ffi" in normalize_text("\ufb03cient")
        assert "fi" in normalize_text("\ufb01nding")
        assert "fl" in normalize_text("\ufb02ow")

    def test_strips_null_bytes(self):
        result = normalize_text("hello\x00world")
        assert "\x00" not in result
        assert "helloworld" in result

    def test_unicode_normalization(self):
        # NFKC should normalize compatibility characters
        result = normalize_text("\u2126")  # OHM SIGN -> greek capital omega
        assert result == "\u03A9"


# ---------------------------------------------------------------------------
# remove_cid_references
# ---------------------------------------------------------------------------
class TestRemoveCidReferences:
    def test_removes_cid_tags(self):
        text = "Some (cid:21) text (cid:42) here"
        result = remove_cid_references(text)
        assert "(cid:" not in result
        assert "Some  text  here" in result

    def test_removes_empty_lines_after_cid(self):
        text = "Line1\n(cid:10)(cid:20)\nLine2"
        result = remove_cid_references(text)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# remove_header_footer_noise
# ---------------------------------------------------------------------------
class TestRemoveHeaderFooterNoise:
    def test_removes_page_numbers(self):
        text = "Real content\n123\nMore content"
        result = remove_header_footer_noise(text)
        lines = result.strip().split("\n")
        for line in lines:
            if line.strip():
                assert line.strip() != "123"

    def test_removes_doi(self):
        text = "Content\ndoi: 10.1234/abc\nMore content"
        result = remove_header_footer_noise(text)
        assert "doi:" not in result.lower()

    def test_removes_copyright(self):
        text = "Content\n© 2023 Publisher\nMore content"
        result = remove_header_footer_noise(text)
        assert "©" not in result

    def test_removes_spaced_headers(self):
        text = "Normal text\nC r i t i c a l R e v i e w s\nMore text"
        result = remove_header_footer_noise(text)
        assert "C r i t i c a l" not in result

    def test_preserves_normal_content(self):
        text = "This is normal content\nWith multiple lines\nOf text"
        result = remove_header_footer_noise(text)
        assert "This is normal content" in result


# ---------------------------------------------------------------------------
# remove_pipe_flood
# ---------------------------------------------------------------------------
class TestRemovePipeFlood:
    def test_removes_pipe_heavy_lines(self):
        text = "Normal line\n|||||text|||||more|||||\nAnother line"
        result = remove_pipe_flood(text)
        assert "||||" not in result

    def test_preserves_normal_table(self):
        text = "| Col1 | Col2 | Col3 |\n| --- | --- | --- |\n| val1 | val2 | val3 |"
        result = remove_pipe_flood(text)
        assert "Col1" in result

    def test_converts_pipe_flood_to_prose(self):
        text = "|||word1|||word2|||word3|||"
        result = remove_pipe_flood(text)
        assert "word1" in result
        assert "|||" not in result


# ---------------------------------------------------------------------------
# _is_valid_heading_format
# ---------------------------------------------------------------------------
class TestIsValidHeadingFormat:
    def test_markdown_heading(self):
        assert _is_valid_heading_format("## References") is True

    def test_all_caps(self):
        assert _is_valid_heading_format("REFERENCES") is True

    def test_bare_keyword(self):
        assert _is_valid_heading_format("References") is True
        assert _is_valid_heading_format("bibliography") is True

    def test_mixed_case_prose(self):
        assert _is_valid_heading_format("Data availability statement") is False

    def test_short_string(self):
        assert _is_valid_heading_format("AB") is False


# ---------------------------------------------------------------------------
# truncate_at_trailing_sections
# ---------------------------------------------------------------------------
class TestTruncateAtTrailingSections:
    def test_truncates_at_references(self):
        body = "\n".join([f"Line {i}" for i in range(100)])
        text = body + "\n## References\n[1] Smith et al., 2020"
        result = truncate_at_trailing_sections(text)
        assert "References" not in result
        assert "Smith" not in result

    def test_no_truncation_without_heading(self):
        text = "\n".join([f"Line {i}" for i in range(100)])
        result = truncate_at_trailing_sections(text)
        assert result == text

    def test_does_not_truncate_too_early(self):
        text = "## References\nSome early text"
        result = truncate_at_trailing_sections(text)
        # Should NOT truncate because fewer than _MIN_LINES_BEFORE_TRUNCATION
        assert "References" in result

    def test_truncates_acknowledgments(self):
        body = "\n".join([f"Line {i}" for i in range(100)])
        text = body + "\nACKNOWLEDGMENTS\nThanks to..."
        result = truncate_at_trailing_sections(text)
        assert "ACKNOWLEDGMENTS" not in result


# ---------------------------------------------------------------------------
# clean_excessive_whitespace
# ---------------------------------------------------------------------------
class TestCleanExcessiveWhitespace:
    def test_collapses_triple_newlines(self):
        text = "a\n\n\n\nb"
        result = clean_excessive_whitespace(text)
        assert result == "a\n\nb"

    def test_preserves_double_newlines(self):
        text = "a\n\nb"
        result = clean_excessive_whitespace(text)
        assert result == "a\n\nb"


# ---------------------------------------------------------------------------
# extract_year_from_text
# ---------------------------------------------------------------------------
class TestExtractYearFromText:
    def test_finds_modern_year(self):
        text = "Published in 2023 by the journal of...\nCopyright 2023"
        assert extract_year_from_text(text) == "2023"

    def test_prefers_modern_year(self):
        text = "Cited from 1995, but published 2021, and revised 2021"
        assert extract_year_from_text(text) == "2021"

    def test_no_year_returns_empty(self):
        assert extract_year_from_text("no years here") == ""

    def test_falls_back_to_old_year(self):
        text = "Published in 1998"
        assert extract_year_from_text(text) == "1998"


# ---------------------------------------------------------------------------
# next_number
# ---------------------------------------------------------------------------
class TestNextNumber:
    def test_empty_index(self):
        assert next_number({}) == 1

    def test_existing_entries(self):
        assert next_number({"a.pdf": 3, "b.pdf": 7}) == 8


# ---------------------------------------------------------------------------
# postprocess (integration)
# ---------------------------------------------------------------------------
class TestPostprocess:
    def test_handles_empty_string(self):
        assert postprocess("") == ""

    def test_cleans_cid_and_whitespace(self):
        text = "Some (cid:1) text\n\n\n\nMore text"
        result = postprocess(text)
        assert "(cid:" not in result
        assert "\n\n\n" not in result

    def test_removes_ligatures_and_noise(self):
        text = "E\ufb03cient method\n© 2023 Publisher\nNormal content"
        result = postprocess(text)
        assert "Efficient" in result
        assert "©" not in result
