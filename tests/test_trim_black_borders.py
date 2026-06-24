"""Tests for frame-processing/00_trim_black_borders_from_frames.py"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

repo = Path(__file__).resolve().parent.parent
fp_dir = str(repo / "frame-processing")
if fp_dir not in sys.path:
    sys.path.insert(0, fp_dir)

_mod = importlib.import_module("00_trim_black_borders_from_frames")
compute_bbox_nonblack = _mod.compute_bbox_nonblack


class TestComputeBboxNonblack:
    def test_full_white_image(self):
        gray = np.full((100, 100), 255, dtype=np.uint8)
        bbox = compute_bbox_nonblack(gray, thresh=5)
        assert bbox == (0, 0, 100, 100)

    def test_all_black_returns_none(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        bbox = compute_bbox_nonblack(gray, thresh=5)
        assert bbox is None

    def test_content_in_center(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[20:80, 30:70] = 200
        bbox = compute_bbox_nonblack(gray, thresh=5)
        assert bbox == (30, 20, 70, 80)

    def test_single_bright_pixel(self):
        gray = np.zeros((50, 50), dtype=np.uint8)
        gray[25, 25] = 100
        bbox = compute_bbox_nonblack(gray, thresh=5)
        assert bbox == (25, 25, 26, 26)

    def test_content_at_edges(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        gray[0, 0] = 50
        gray[9, 9] = 50
        bbox = compute_bbox_nonblack(gray, thresh=5)
        assert bbox == (0, 0, 10, 10)

    def test_threshold_filters_low_values(self):
        gray = np.full((10, 10), 3, dtype=np.uint8)  # all below thresh=5
        bbox = compute_bbox_nonblack(gray, thresh=5)
        assert bbox is None

    def test_threshold_exact_boundary(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        gray[5, 5] = 5  # exactly at threshold, not above
        bbox = compute_bbox_nonblack(gray, thresh=5)
        assert bbox is None

        gray[5, 5] = 6  # above threshold
        bbox = compute_bbox_nonblack(gray, thresh=5)
        assert bbox == (5, 5, 6, 6)

    def test_rectangular_image(self):
        gray = np.zeros((50, 200), dtype=np.uint8)
        gray[10:40, 50:150] = 128
        bbox = compute_bbox_nonblack(gray, thresh=5)
        assert bbox == (50, 10, 150, 40)
