"""Tests for utils/09_data_augmentor_20percent.py"""

import importlib
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

repo = Path(__file__).resolve().parent.parent
utils_dir = str(repo / "utils")
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

_mod = importlib.import_module("09_data_augmentor_20percent")

select_frame_subset = _mod.select_frame_subset
make_policy = _mod.make_policy
rotate_image = _mod.rotate_image
zoom_in_image = _mod.zoom_in_image
adjust_brightness = _mod.adjust_brightness
adjust_contrast = _mod.adjust_contrast
adjust_gamma = _mod.adjust_gamma
add_gaussian_noise = _mod.add_gaussian_noise
apply_policy = _mod.apply_policy


# ---------------------------------------------------------------------------
# select_frame_subset
# ---------------------------------------------------------------------------
class TestSelectFrameSubset:
    def test_empty_input(self):
        assert select_frame_subset([], keep_fraction=0.2) == []

    def test_all_frames_when_fraction_is_one(self):
        frames = list(range(10))
        result = select_frame_subset(frames, keep_fraction=1.0)
        assert result == frames

    def test_respects_min_frames(self):
        frames = list(range(5))
        result = select_frame_subset(frames, keep_fraction=0.01, min_frames=3)
        assert len(result) >= 3

    def test_twenty_percent_of_100(self):
        frames = list(range(100))
        result = select_frame_subset(frames, keep_fraction=0.20, min_frames=1)
        assert len(result) == 20

    def test_small_list_keeps_at_least_one(self):
        frames = list(range(3))
        result = select_frame_subset(frames, keep_fraction=0.01, min_frames=1)
        assert len(result) >= 1

    def test_result_is_sorted_subset(self):
        frames = list(range(50))
        result = select_frame_subset(frames, keep_fraction=0.2, min_frames=1)
        # Check all elements are from original
        for item in result:
            assert item in frames
        # Check sorted (indices should be increasing for range)
        assert result == sorted(result)

    def test_never_exceeds_input_length(self):
        frames = list(range(5))
        result = select_frame_subset(frames, keep_fraction=2.0, min_frames=10)
        assert len(result) <= len(frames)


# ---------------------------------------------------------------------------
# make_policy
# ---------------------------------------------------------------------------
class TestMakePolicy:
    def test_returns_dict(self):
        rng = random.Random(42)
        policy = make_policy(rng)
        assert isinstance(policy, dict)
        assert "type" in policy

    def test_at_least_one_augmentation_enabled(self):
        rng = random.Random(42)
        for _ in range(20):
            policy = make_policy(rng)
            aug_flags = [
                policy.get("rotate"),
                policy.get("zoom"),
                policy.get("brightness"),
                policy.get("contrast"),
                policy.get("gamma"),
                policy.get("gaussian_noise"),
            ]
            assert any(aug_flags), f"Policy {policy['type']} has no augmentation enabled"

    def test_deterministic_with_same_seed(self):
        p1 = make_policy(random.Random(99))
        p2 = make_policy(random.Random(99))
        assert p1 == p2


# ---------------------------------------------------------------------------
# Image augmentation helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_image():
    return Image.fromarray(np.full((64, 64), 128, dtype=np.uint8))


@pytest.fixture
def sample_rgb_image():
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr)


class TestRotateImage:
    def test_preserves_size(self, sample_image):
        rotated = rotate_image(sample_image, 5.0)
        assert rotated.size == sample_image.size

    def test_negative_rotation(self, sample_image):
        rotated = rotate_image(sample_image, -5.0)
        assert rotated.size == sample_image.size


class TestZoomInImage:
    def test_preserves_size(self, sample_image):
        zoomed = zoom_in_image(sample_image, 1.2)
        assert zoomed.size == sample_image.size

    def test_no_zoom_returns_copy(self, sample_image):
        result = zoom_in_image(sample_image, 1.0)
        assert result.size == sample_image.size

    def test_zoom_less_than_one(self, sample_image):
        result = zoom_in_image(sample_image, 0.8)
        assert result.size == sample_image.size


class TestAdjustBrightness:
    def test_no_change_at_factor_one(self, sample_image):
        result = adjust_brightness(sample_image, 1.0)
        assert result.size == sample_image.size

    def test_darkens_below_one(self, sample_image):
        result = adjust_brightness(sample_image, 0.5)
        arr = np.array(result)
        assert arr.mean() < 128


class TestAdjustContrast:
    def test_no_change_at_factor_one(self, sample_image):
        result = adjust_contrast(sample_image, 1.0)
        assert result.size == sample_image.size


class TestAdjustGamma:
    def test_gamma_one_no_change(self, sample_image):
        result = adjust_gamma(sample_image, 1.0)
        arr_in = np.array(sample_image)
        arr_out = np.array(result)
        np.testing.assert_array_almost_equal(arr_in, arr_out, decimal=0)

    def test_gamma_less_than_one_brightens(self, sample_image):
        result = adjust_gamma(sample_image, 0.5)
        assert np.array(result).mean() > np.array(sample_image).mean()

    def test_gamma_zero_raises(self, sample_image):
        with pytest.raises(ValueError):
            adjust_gamma(sample_image, 0)

    def test_negative_gamma_raises(self, sample_image):
        with pytest.raises(ValueError):
            adjust_gamma(sample_image, -1.0)


class TestAddGaussianNoise:
    def test_output_same_size(self, sample_image):
        rng = np.random.default_rng(42)
        result = add_gaussian_noise(sample_image, sigma=5.0, rng=rng)
        assert result.size == sample_image.size

    def test_noise_changes_pixels(self, sample_image):
        rng = np.random.default_rng(42)
        result = add_gaussian_noise(sample_image, sigma=20.0, rng=rng)
        arr_in = np.array(sample_image)
        arr_out = np.array(result)
        assert not np.array_equal(arr_in, arr_out)


class TestApplyPolicy:
    def test_rotate_policy(self, sample_image):
        policy = {
            "type": "rotate", "rotate": True, "degrees": 5,
            "zoom": False, "brightness": False, "contrast": False,
            "gamma": False, "gaussian_noise": False,
        }
        rng = np.random.default_rng(0)
        result = apply_policy(sample_image, policy, rng)
        assert result.size == sample_image.size

    def test_combined_policy(self, sample_rgb_image):
        policy = {
            "type": "rotate_zoom_brightness_contrast",
            "rotate": True, "degrees": 5,
            "zoom": True, "zoom_factor": 1.1,
            "brightness": True, "brightness_factor": 0.95,
            "contrast": True, "contrast_factor": 1.05,
            "gamma": False, "gaussian_noise": False,
        }
        rng = np.random.default_rng(0)
        result = apply_policy(sample_rgb_image, policy, rng)
        assert result.size == sample_rgb_image.size
