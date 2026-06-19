"""Tests for frame-processing/00_mosaic_creation.py"""

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

repo = Path(__file__).resolve().parent.parent
fp_dir = str(repo / "frame-processing")
if fp_dir not in sys.path:
    sys.path.insert(0, fp_dir)

_mod = importlib.import_module("00_mosaic_creation")

pick_frames = _mod.pick_frames
_fit_to_box = _mod._fit_to_box
create_mosaic_image = _mod.create_mosaic_image
list_frame_files = _mod.list_frame_files


# ---------------------------------------------------------------------------
# pick_frames
# ---------------------------------------------------------------------------
class TestPickFrames:
    def test_stride_one_no_limit(self):
        frames = list(range(10))
        result = pick_frames(frames, max_frames=0, stride=1)
        assert result == frames

    def test_stride_two(self):
        frames = list(range(10))
        result = pick_frames(frames, max_frames=0, stride=2)
        assert result == [0, 2, 4, 6, 8]

    def test_max_frames_limits_output(self):
        frames = list(range(20))
        result = pick_frames(frames, max_frames=5, stride=1)
        assert len(result) == 5

    def test_max_frames_one_picks_middle(self):
        frames = list(range(10))
        result = pick_frames(frames, max_frames=1, stride=1)
        assert len(result) == 1
        assert result[0] == frames[5]

    def test_stride_zero_defaults_to_one(self):
        frames = list(range(5))
        result = pick_frames(frames, max_frames=0, stride=0)
        assert result == frames

    def test_negative_stride(self):
        frames = list(range(5))
        result = pick_frames(frames, max_frames=0, stride=-1)
        assert result == frames

    def test_empty_input(self):
        assert pick_frames([], max_frames=5, stride=1) == []


# ---------------------------------------------------------------------------
# _fit_to_box
# ---------------------------------------------------------------------------
class TestFitToBox:
    def test_preserves_target_size(self):
        img = Image.new("RGB", (100, 200))
        result = _fit_to_box(img, (50, 50))
        assert result.size == (50, 50)

    def test_square_image_in_square_box(self):
        img = Image.new("RGB", (100, 100))
        result = _fit_to_box(img, (50, 50))
        assert result.size == (50, 50)

    def test_landscape_in_portrait_box(self):
        img = Image.new("RGB", (200, 100))
        result = _fit_to_box(img, (50, 100))
        assert result.size == (50, 100)

    def test_zero_size_image(self):
        img = Image.new("RGB", (0, 0))
        result = _fit_to_box(img, (50, 50))
        assert result.size == (50, 50)


# ---------------------------------------------------------------------------
# create_mosaic_image
# ---------------------------------------------------------------------------
class TestCreateMosaicImage:
    def test_creates_mosaic(self, tmp_path):
        # Create small test images
        frame_paths = []
        for i in range(4):
            img = Image.fromarray(
                np.full((32, 32, 3), i * 50, dtype=np.uint8)
            )
            p = tmp_path / f"frame_{i}.png"
            img.save(p)
            frame_paths.append(p)

        out_path = tmp_path / "mosaic.png"
        result = create_mosaic_image(
            frame_paths, out_path,
            tile_size=(32, 32), max_cols=2, threads=1,
        )
        assert result is not None
        assert out_path.exists()

        mosaic = Image.open(out_path)
        assert mosaic.size == (64, 64)  # 2 cols x 2 rows

    def test_empty_input_returns_none(self, tmp_path):
        out_path = tmp_path / "mosaic.png"
        result = create_mosaic_image([], out_path)
        assert result is None

    def test_single_frame(self, tmp_path):
        img = Image.fromarray(np.full((32, 32, 3), 100, dtype=np.uint8))
        p = tmp_path / "frame.png"
        img.save(p)

        out_path = tmp_path / "mosaic.png"
        result = create_mosaic_image(
            [p], out_path,
            tile_size=(64, 64), threads=1,
        )
        assert result is not None
        assert out_path.exists()


# ---------------------------------------------------------------------------
# list_frame_files
# ---------------------------------------------------------------------------
class TestListFrameFiles:
    def test_finds_frames_in_subdir(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for ext in [".png", ".jpg", ".bmp"]:
            (frames_dir / f"img{ext}").write_bytes(b"fake")

        result = list_frame_files(tmp_path, frames_subdir="frames")
        assert len(result) == 3

    def test_empty_frames_dir(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        result = list_frame_files(tmp_path, frames_subdir="frames")
        # Falls back to rglob, finds nothing
        assert len(result) == 0

    def test_sorted_by_name(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for name in ["c.png", "a.png", "b.png"]:
            (frames_dir / name).write_bytes(b"fake")

        result = list_frame_files(tmp_path, frames_subdir="frames")
        names = [p.name for p in result]
        assert names == sorted(names)
