"""Tests for fine-tuning/pipeline.py"""

import math

import pytest

from pipeline import (
    ContrastiveTrainer,
    FrozenTextEncoder,
    MergeLayer,
    StudySample,
    TrainableVisionEncoder,
    build_toy_dataset,
    dot,
    l2_normalize,
    parse_args,
)


# ---------------------------------------------------------------------------
# l2_normalize
# ---------------------------------------------------------------------------
class TestL2Normalize:
    def test_unit_length(self):
        vec = [3.0, 4.0]
        normed = l2_normalize(vec)
        length = math.sqrt(sum(v * v for v in normed))
        assert abs(length - 1.0) < 1e-6

    def test_zero_vector(self):
        normed = l2_normalize([0.0, 0.0, 0.0])
        assert all(abs(v) < 1e-6 for v in normed)

    def test_single_element(self):
        normed = l2_normalize([5.0])
        assert abs(normed[0] - 1.0) < 1e-6

    def test_preserves_direction(self):
        vec = [1.0, 2.0, 3.0]
        normed = l2_normalize(vec)
        ratios_orig = [vec[1] / vec[0], vec[2] / vec[0]]
        ratios_normed = [normed[1] / normed[0], normed[2] / normed[0]]
        for ro, rn in zip(ratios_orig, ratios_normed):
            assert abs(ro - rn) < 1e-6

    def test_negative_values(self):
        vec = [-3.0, 4.0]
        normed = l2_normalize(vec)
        length = math.sqrt(sum(v * v for v in normed))
        assert abs(length - 1.0) < 1e-6
        assert normed[0] < 0


# ---------------------------------------------------------------------------
# dot
# ---------------------------------------------------------------------------
class TestDot:
    def test_orthogonal(self):
        assert dot([1, 0], [0, 1]) == 0.0

    def test_parallel(self):
        assert abs(dot([2, 3], [2, 3]) - 13.0) < 1e-9

    def test_antiparallel(self):
        assert abs(dot([1, 0], [-1, 0]) - (-1.0)) < 1e-9

    def test_empty_vectors(self):
        assert dot([], []) == 0.0


# ---------------------------------------------------------------------------
# TrainableVisionEncoder
# ---------------------------------------------------------------------------
class TestTrainableVisionEncoder:
    def test_encode_returns_correct_dim(self):
        enc = TrainableVisionEncoder(embedding_dim=8, seed=0)
        result = enc.encode([1.0, 2.0, 3.0])
        assert len(result) == 8

    def test_encode_empty_sequence(self):
        enc = TrainableVisionEncoder(embedding_dim=4, seed=0)
        result = enc.encode([])
        assert result == [0.0] * 4

    def test_deterministic_with_same_seed(self):
        enc1 = TrainableVisionEncoder(embedding_dim=4, seed=42)
        enc2 = TrainableVisionEncoder(embedding_dim=4, seed=42)
        r1 = enc1.encode([1.0, 2.0])
        r2 = enc2.encode([1.0, 2.0])
        assert r1 == r2

    def test_update_changes_weights(self):
        enc = TrainableVisionEncoder(embedding_dim=4, seed=0)
        old_weights = list(enc.weights)
        enc.update([0.1, 0.2, 0.3, 0.4], avg_input=1.0, lr=0.01)
        assert enc.weights != old_weights


# ---------------------------------------------------------------------------
# FrozenTextEncoder
# ---------------------------------------------------------------------------
class TestFrozenTextEncoder:
    def test_encode_returns_correct_dim(self):
        enc = FrozenTextEncoder(embedding_dim=16)
        result = enc.encode("hello world test")
        assert len(result) == 16

    def test_encode_is_unit_length(self):
        enc = FrozenTextEncoder(embedding_dim=16)
        result = enc.encode("some text here")
        length = math.sqrt(sum(v * v for v in result))
        assert abs(length - 1.0) < 1e-6

    def test_deterministic(self):
        enc = FrozenTextEncoder(embedding_dim=16)
        r1 = enc.encode("coronary artery stenosis")
        r2 = enc.encode("coronary artery stenosis")
        assert r1 == r2

    def test_different_texts_different_embeddings(self):
        enc = FrozenTextEncoder(embedding_dim=32)
        r1 = enc.encode("left coronary artery")
        r2 = enc.encode("right hepatic vein")
        assert r1 != r2

    def test_empty_string(self):
        enc = FrozenTextEncoder(embedding_dim=8)
        result = enc.encode("")
        assert len(result) == 8


# ---------------------------------------------------------------------------
# MergeLayer
# ---------------------------------------------------------------------------
class TestMergeLayer:
    def test_single_vector(self):
        ml = MergeLayer()
        result = ml.fuse([[1.0, 2.0, 3.0]])
        assert result == [1.0, 2.0, 3.0]

    def test_average_of_two(self):
        ml = MergeLayer()
        result = ml.fuse([[2.0, 4.0], [4.0, 6.0]])
        assert abs(result[0] - 3.0) < 1e-9
        assert abs(result[1] - 5.0) < 1e-9

    def test_empty_raises(self):
        ml = MergeLayer()
        with pytest.raises(ValueError, match="at least one"):
            ml.fuse([])


# ---------------------------------------------------------------------------
# ContrastiveTrainer
# ---------------------------------------------------------------------------
class TestContrastiveTrainer:
    def test_init(self):
        trainer = ContrastiveTrainer(embedding_dim=8, num_sequences=2)
        assert len(trainer.vision_encoders) == 2

    def test_clip_loss_nonnegative(self):
        trainer = ContrastiveTrainer(embedding_dim=8, num_sequences=2)
        v = [l2_normalize([1.0] * 8), l2_normalize([0.5] * 8)]
        t = [l2_normalize([0.8] * 8), l2_normalize([0.3] * 8)]
        loss = trainer.clip_loss(v, t)
        assert loss >= 0.0

    def test_train_reduces_loss(self):
        dataset = build_toy_dataset(num_sequences=2, sequence_length=4)
        trainer = ContrastiveTrainer(embedding_dim=8, num_sequences=2, lr=0.05)

        # Compute initial similarities
        initial_sims = [trainer.infer_similarity(s) for s in dataset]

        trainer.train(dataset, epochs=30)

        final_sims = [trainer.infer_similarity(s) for s in dataset]
        # After training, paired similarities should generally increase
        assert sum(final_sims) > sum(initial_sims)

    def test_infer_similarity_returns_float(self):
        trainer = ContrastiveTrainer(embedding_dim=8, num_sequences=2)
        sample = StudySample(
            sequence_values=[[0.5, 0.6], [0.3, 0.4]],
            report_text="test report",
        )
        sim = trainer.infer_similarity(sample)
        assert isinstance(sim, float)


# ---------------------------------------------------------------------------
# build_toy_dataset
# ---------------------------------------------------------------------------
class TestBuildToyDataset:
    def test_returns_three_samples(self):
        ds = build_toy_dataset(num_sequences=3, sequence_length=4)
        assert len(ds) == 3

    def test_sequence_count(self):
        ds = build_toy_dataset(num_sequences=2, sequence_length=4)
        for sample in ds:
            assert len(sample.sequence_values) == 2

    def test_sequence_length(self):
        ds = build_toy_dataset(num_sequences=3, sequence_length=3)
        for sample in ds:
            for seq in sample.sequence_values:
                assert len(seq) == 3

    def test_has_report_text(self):
        ds = build_toy_dataset(num_sequences=1, sequence_length=1)
        for sample in ds:
            assert len(sample.report_text) > 0


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------
class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.epochs == 20
        assert args.embedding_dim == 16
        assert args.num_sequences == 3

    def test_custom_values(self):
        args = parse_args(["--epochs", "50", "--lr", "0.1"])
        assert args.epochs == 50
        assert abs(args.lr - 0.1) < 1e-9
