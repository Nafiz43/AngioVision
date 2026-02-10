"""Minimal multimodal fine-tuning prototype.

This script implements a lightweight version of the pipeline shown in the
fine-tuning diagram:

study input -> frame sequences + text report ->
trainable vision encoders -> per-sequence features -> fusion ->
frozen text encoder -> contrastive training objective.

It intentionally avoids heavy model dependencies to keep the example simple.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence


def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec)) + 1e-12
    return [v / norm for v in vec]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class StudySample:
    """A single study containing multiple visual sequences and one report."""

    sequence_values: List[List[float]]
    report_text: str


class TrainableVisionEncoder:
    """Tiny trainable encoder: weighted average over sequence values."""

    def __init__(self, embedding_dim: int, seed: int | None = None) -> None:
        rng = random.Random(seed)
        self.weights = [rng.uniform(-0.3, 0.3) for _ in range(embedding_dim)]
        self.bias = [0.0 for _ in range(embedding_dim)]

    def encode(self, sequence: Sequence[float]) -> List[float]:
        if not sequence:
            return [0.0] * len(self.weights)
        avg = sum(sequence) / len(sequence)
        return [avg * w + b for w, b in zip(self.weights, self.bias)]

    def update(self, grad: Sequence[float], avg_input: float, lr: float) -> None:
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grad[i] * avg_input
            self.bias[i] -= lr * grad[i]


class FrozenTextEncoder:
    """Frozen bag-of-words style encoder (deterministic hash mapping)."""

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.embedding_dim
        words = [w.strip(".,!?;:()[]{}\"'").lower() for w in text.split()]
        words = [w for w in words if w]
        for word in words:
            idx = hash(word) % self.embedding_dim
            vec[idx] += 1.0
        return l2_normalize(vec)


class MergeLayer:
    """Simple fusion layer that averages per-sequence features."""

    def fuse(self, feature_list: Sequence[Sequence[float]]) -> List[float]:
        if not feature_list:
            raise ValueError("Expected at least one feature vector to fuse.")
        dim = len(feature_list[0])
        merged = [0.0] * dim
        for feat in feature_list:
            for i, value in enumerate(feat):
                merged[i] += value
        count = float(len(feature_list))
        return [value / count for value in merged]


class ContrastiveTrainer:
    """CLIP-like contrastive optimization for visual encoders only."""

    def __init__(self, embedding_dim: int, num_sequences: int, lr: float = 1e-2) -> None:
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.vision_encoders = [
            TrainableVisionEncoder(embedding_dim=embedding_dim, seed=10 + i)
            for i in range(num_sequences)
        ]
        self.merge = MergeLayer()
        self.text_encoder = FrozenTextEncoder(embedding_dim=embedding_dim)

    def encode_visual(self, sample: StudySample) -> tuple[List[float], List[float]]:
        features = [
            encoder.encode(sequence)
            for encoder, sequence in zip(self.vision_encoders, sample.sequence_values)
        ]
        merged = self.merge.fuse(features)
        return l2_normalize(merged), [sum(seq) / max(len(seq), 1) for seq in sample.sequence_values]

    def clip_loss(self, visual_batch: Sequence[Sequence[float]], text_batch: Sequence[Sequence[float]]) -> float:
        total = 0.0
        for i in range(len(visual_batch)):
            sims = [dot(visual_batch[i], t) for t in text_batch]
            max_sim = max(sims)
            exp_sims = [math.exp(s - max_sim) for s in sims]
            denom = sum(exp_sims)
            p_pos = exp_sims[i] / denom
            total += -math.log(max(p_pos, 1e-12))
        return total / len(visual_batch)

    def train(self, dataset: Sequence[StudySample], epochs: int = 30) -> None:
        for epoch in range(1, epochs + 1):
            visual_embeddings = []
            avg_inputs_per_sample = []
            text_embeddings = [self.text_encoder.encode(sample.report_text) for sample in dataset]

            for sample in dataset:
                visual, avg_inputs = self.encode_visual(sample)
                visual_embeddings.append(visual)
                avg_inputs_per_sample.append(avg_inputs)

            loss = self.clip_loss(visual_embeddings, text_embeddings)

            # Approximate gradient: align each sample's visual embedding to its paired text embedding.
            for sample_idx, sample in enumerate(dataset):
                v = visual_embeddings[sample_idx]
                t = text_embeddings[sample_idx]
                grad_vec = [v_i - t_i for v_i, t_i in zip(v, t)]

                for enc_idx, encoder in enumerate(self.vision_encoders):
                    scale = 1.0 / len(self.vision_encoders)
                    scaled_grad = [g * scale for g in grad_vec]
                    avg_in = avg_inputs_per_sample[sample_idx][enc_idx]
                    encoder.update(scaled_grad, avg_in, self.lr)

            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(f"epoch={epoch:03d} loss={loss:.4f}")

    def infer_similarity(self, sample: StudySample) -> float:
        visual, _ = self.encode_visual(sample)
        text = self.text_encoder.encode(sample.report_text)
        return dot(visual, text)


def build_toy_dataset(num_sequences: int, sequence_length: int) -> List[StudySample]:
    """Create a very small synthetic dataset for a runnable demo."""

    return [
        StudySample(
            sequence_values=[
                [0.85, 0.75, 0.95, 0.8][:sequence_length],
                [0.65, 0.6, 0.7, 0.62][:sequence_length],
                [0.92, 0.89, 0.94, 0.87][:sequence_length],
            ][:num_sequences],
            report_text="Left coronary injection with moderate proximal LAD stenosis.",
        ),
        StudySample(
            sequence_values=[
                [0.12, 0.2, 0.1, 0.18][:sequence_length],
                [0.33, 0.28, 0.31, 0.35][:sequence_length],
                [0.22, 0.19, 0.25, 0.21][:sequence_length],
            ][:num_sequences],
            report_text="Right coronary angiogram with mild diffuse irregularity.",
        ),
        StudySample(
            sequence_values=[
                [0.72, 0.78, 0.74, 0.7][:sequence_length],
                [0.4, 0.45, 0.43, 0.41][:sequence_length],
                [0.83, 0.8, 0.85, 0.79][:sequence_length],
            ][:num_sequences],
            report_text="Balanced circulation pattern with no critical obstruction.",
        ),
    ]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal multimodal fine-tuning demo.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of toy training epochs.")
    parser.add_argument("--embedding-dim", type=int, default=16, help="Shared embedding dimension.")
    parser.add_argument("--num-sequences", type=int, default=3, help="Number of visual sequences per study.")
    parser.add_argument("--sequence-length", type=int, default=4, help="Length of each sequence.")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate for vision encoders.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    dataset = build_toy_dataset(args.num_sequences, args.sequence_length)

    trainer = ContrastiveTrainer(
        embedding_dim=args.embedding_dim,
        num_sequences=args.num_sequences,
        lr=args.lr,
    )

    print("Initial paired similarity:")
    for idx, sample in enumerate(dataset):
        print(f"  study_{idx}: {trainer.infer_similarity(sample):.4f}")

    trainer.train(dataset=dataset, epochs=args.epochs)

    print("\nFinal paired similarity:")
    for idx, sample in enumerate(dataset):
        print(f"  study_{idx}: {trainer.infer_similarity(sample):.4f}")


if __name__ == "__main__":
    main()
