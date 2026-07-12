"""
angio_ft.data
──────────────
Study-level dataset for contrastive training.

A "study" (keyed by an accession number) owns several frame *sequences*
(one per SOPInstanceUID) plus one or more report texts. The dataset:

  • builds a report map (accession -> list of report variants) via groupby,
  • pre-scans the filesystem once and caches frame paths per sequence,
  • optionally uniformly subsamples frames per sequence,
  • samples one report variant per __getitem__ (reproducible via worker seeding).

Lifted behaviour-for-behaviour from ``custom_framework_train_temporal.py``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import find_frame_files_for_sop, parse_sop_instance_uids


def random_keep_fraction(items: List[Any], fraction: float, rng: random.Random) -> List[Any]:
    """Randomly keep ~`fraction` of items (at least 1), preserving order."""
    if fraction >= 1.0 or len(items) <= 1:
        return items
    k = max(1, int(round(len(items) * fraction)))
    keep = sorted(rng.sample(range(len(items)), k))
    return [items[i] for i in keep]


class StudyDataset(Dataset):
    def __init__(
        self,
        meta_csv: Path,
        reports_csv: Path,
        base_frames_dir: Path,
        report_text_col: str = "radrpt",
        anon_col: str = "Anon Acc #",
        sop_col: str = "SOPInstanceUIDs",
        report_type_col: str = "Type",
        min_frames_per_sequence: int = 1,
        max_sequences_per_study: Optional[int] = None,
        max_frames_per_sequence: Optional[int] = None,
        drop_missing_reports: bool = True,
        report_sampling: str = "uniform",
        report_sampling_seed: int = 42,
        frame_keep_fraction: float = 1.0,
        frame_keep_seed: int = 42,
    ):
        self.base_frames_dir = Path(base_frames_dir)
        self.report_text_col = report_text_col
        self.anon_col = anon_col
        self.sop_col = sop_col
        self.report_type_col = report_type_col
        self.min_frames_per_sequence = min_frames_per_sequence
        self.max_sequences_per_study = max_sequences_per_study
        self.max_frames_per_sequence = max_frames_per_sequence
        self.drop_missing_reports = drop_missing_reports
        self.report_sampling = report_sampling
        self.report_sampling_seed = int(report_sampling_seed)
        self.frame_keep_fraction = float(frame_keep_fraction)
        self._frame_keep_rng = random.Random(int(frame_keep_seed))

        if self.report_sampling != "uniform":
            raise ValueError(
                f"Unsupported report_sampling={self.report_sampling!r}. Only 'uniform' is supported."
            )

        meta = pd.read_csv(meta_csv)
        reports = pd.read_csv(reports_csv)

        # ── build report_map with groupby instead of row loop ──
        self.report_map: Dict[str, List[Dict[str, str]]] = {}
        reports = reports.copy()
        reports[self.anon_col] = reports[self.anon_col].astype(str).str.strip()
        reports[self.report_text_col] = (
            reports[self.report_text_col].fillna("").astype(str).str.strip()
        )
        reports[self.report_type_col] = (
            reports.get(self.report_type_col, pd.Series("Unknown", index=reports.index))
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )
        # Filter to non-empty texts before grouping
        valid_reports = reports[reports[self.report_text_col] != ""]
        for acc, grp in valid_reports.groupby(self.anon_col, sort=False):
            self.report_map[str(acc)] = [
                {"text": row[self.report_text_col], "type": row[self.report_type_col]}
                for _, row in grp.iterrows()
            ]

        if self.drop_missing_reports:
            mask = meta[self.anon_col].astype(str).str.strip().isin(self.report_map)
            meta = meta[mask].reset_index(drop=True)

        self.meta = meta
        self.report_count_map: Dict[str, int] = {
            acc: len(rpts) for acc, rpts in self.report_map.items()
        }
        self._log_report_summary()

        # ── pre-build per-accession frame-path cache ──────────
        print("[INFO] Building frame-path cache (one-time filesystem scan)...")
        self._frame_cache: Dict[Tuple[str, str], List[Path]] = {}
        self._sequences_cache: Dict[str, List[List[Path]]] = {}
        self._build_frame_cache()
        print(f"[INFO] Frame-path cache built: {len(self._frame_cache)} SOP entries cached.")

    # ── frame cache helpers ────────────────────────────────────────────────

    def _build_frame_cache(self) -> None:
        """Pre-populate frame paths for every (acc, sop_uid) in meta."""
        for _, row in self.meta.iterrows():
            acc = str(row.get(self.anon_col, "")).strip()
            sop_uids = parse_sop_instance_uids(row.get(self.sop_col, ""))
            if self.max_sequences_per_study is not None:
                sop_uids = sop_uids[: self.max_sequences_per_study]

            sequences: List[List[Path]] = []
            for sop in sop_uids:
                key = (acc, sop)
                if key not in self._frame_cache:
                    self._frame_cache[key] = find_frame_files_for_sop(
                        self.base_frames_dir, acc, sop
                    )
                frame_files = list(self._frame_cache[key])  # shallow copy for safety

                # Random frame subsample (e.g. --20% keeps a random fifth)
                frame_files = random_keep_fraction(
                    frame_files, self.frame_keep_fraction, self._frame_keep_rng
                )

                # Uniform temporal subsample preserving order
                if (
                    self.max_frames_per_sequence is not None
                    and len(frame_files) > self.max_frames_per_sequence
                ):
                    idxs = (
                        torch.linspace(0, len(frame_files) - 1, steps=self.max_frames_per_sequence)
                        .long()
                        .tolist()
                    )
                    frame_files = [frame_files[i] for i in idxs]

                if len(frame_files) >= self.min_frames_per_sequence:
                    sequences.append(frame_files)

            self._sequences_cache[acc] = sequences

    # ── report summary ────────────────────────────────────────────────────

    def _log_report_summary(self) -> None:
        n = len(self.report_count_map)
        if n == 0:
            print("[WARN] No usable reports found in reports_csv.")
            return
        counts = list(self.report_count_map.values())
        print("[INFO] Report variant summary by accession:")
        print(f"       Accessions with >=1 report : {n}")
        print(f"       Min reports per accession  : {min(counts)}")
        print(f"       Max reports per accession  : {max(counts)}")
        print(f"       Mean reports per accession : {sum(counts)/len(counts):.2f}")
        for acc, cnt in sorted(self.report_count_map.items())[:10]:
            print(f"       {acc} -> {cnt}")

    # ── sampling ──────────────────────────────────────────────────────────

    def _sample_report_uniformly(self, reports_for_acc: List[Dict[str, str]]) -> Dict[str, str]:
        """Uniform sample; uses Python's random module which is worker-safe."""
        n = len(reports_for_acc)
        if n == 0:
            return {"text": "", "type": "Missing"}
        if n == 1:
            return reports_for_acc[0]
        # random.randrange is seeded deterministically in worker_init_fn
        return reports_for_acc[random.randrange(n)]

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.meta.iloc[idx]
        acc = str(row.get(self.anon_col, "")).strip()

        # No filesystem access - use pre-built cache
        sequences = self._sequences_cache.get(acc, [])

        reports_for_acc = self.report_map.get(acc, [])
        chosen = self._sample_report_uniformly(reports_for_acc)

        return {
            "acc": acc,
            "sequences": sequences,
            "text": chosen.get("text", ""),
            "report_type": chosen.get("type", "Unknown"),
            "num_reports_for_acc": self.report_count_map.get(acc, 0),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    kept = [
        b for b in batch
        if b["text"] and isinstance(b["sequences"], list) and len(b["sequences"]) > 0
    ]
    return {
        "acc":                [b["acc"]                for b in kept],
        "sequences":          [b["sequences"]          for b in kept],
        "text":               [b["text"]               for b in kept],
        "report_type":        [b["report_type"]        for b in kept],
        "num_reports_for_acc": [b["num_reports_for_acc"] for b in kept],
    }


def worker_init_fn(worker_id: int) -> None:
    """Seed every worker reproducibly so report sampling is deterministic."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
