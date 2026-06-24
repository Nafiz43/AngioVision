"""Tests for batch-processing/merge_csvs.py"""

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

repo = Path(__file__).resolve().parent.parent
bp_dir = str(repo / "batch-processing")
if bp_dir not in sys.path:
    sys.path.insert(0, bp_dir)

_mod = importlib.import_module("merge_csvs")
merge_csvs = _mod.merge_csvs


class TestMergeCsvs:
    def test_merges_two_files(self, tmp_path):
        df1 = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        df2 = pd.DataFrame({"A": [3, 4], "B": ["z", "w"]})
        df1.to_csv(tmp_path / "file1.csv", index=False)
        df2.to_csv(tmp_path / "file2.csv", index=False)

        out = tmp_path / "merged.csv"
        merge_csvs(tmp_path, out)

        result = pd.read_csv(out)
        assert len(result) == 4
        assert list(result.columns) == ["A", "B"]

    def test_raises_on_no_csvs(self, tmp_path):
        out = tmp_path / "merged.csv"
        with pytest.raises(RuntimeError, match="No CSV"):
            merge_csvs(tmp_path, out)

    def test_skips_empty_csvs(self, tmp_path):
        df1 = pd.DataFrame({"A": [1]})
        df1.to_csv(tmp_path / "good.csv", index=False)

        # Create an empty CSV (header only)
        (tmp_path / "empty.csv").write_text("A\n")

        out = tmp_path / "merged.csv"
        merge_csvs(tmp_path, out)
        result = pd.read_csv(out)
        assert len(result) == 1

    def test_handles_subdirectory_csvs(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        df1 = pd.DataFrame({"Col": [10, 20]})
        df1.to_csv(sub / "data.csv", index=False)

        out = tmp_path / "merged.csv"
        merge_csvs(tmp_path, out)
        result = pd.read_csv(out)
        assert len(result) == 2

    def test_all_empty_raises(self, tmp_path):
        (tmp_path / "empty.csv").write_text("A\n")
        out = tmp_path / "merged.csv"
        with pytest.raises(RuntimeError, match="empty or unreadable"):
            merge_csvs(tmp_path, out)
