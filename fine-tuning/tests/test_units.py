"""
CPU-only unit tests for the pure / numerics-critical pieces of angio_ft.

Goal: lock in the "numerics identical to the legacy scripts" guarantee and the
reproducibility machinery BEFORE spending GPU-days on experiments. Everything
here runs on CPU in <1s and needs no model downloads.

Run:  python3 -m pytest tests/test_units.py -q
"""

from __future__ import annotations

import ast
import hashlib
import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Make the package importable regardless of CWD.
_PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT))

from angio_ft.common import build_sinusoidal_position_encoding, pool_stack  # noqa: E402
from angio_ft.losses import clip_loss_chunked, siglip_loss  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# pool_stack
# ─────────────────────────────────────────────────────────────────────────────

def test_pool_stack_modes_match_reference():
    torch.manual_seed(0)
    x = torch.randn(7, 5)
    assert torch.allclose(pool_stack(x, "max"), x.max(dim=0).values)
    assert torch.allclose(pool_stack(x, "mean"), x.mean(dim=0))
    assert torch.allclose(pool_stack(x, "logsumexp"), torch.logsumexp(x, dim=0))


def test_pool_stack_shape_and_guards():
    assert pool_stack(torch.randn(3, 8), "mean").shape == (8,)
    with pytest.raises(ValueError):
        pool_stack(torch.randn(3, 4, 2), "max")   # not 2D
    with pytest.raises(ValueError):
        pool_stack(torch.empty(0, 4), "max")       # empty N
    with pytest.raises(ValueError):
        pool_stack(torch.randn(3, 4), "median")    # unknown mode


def test_streaming_pool_matches_pool_stack():
    """The trainable-ViT path pools in chunks via _init/_update/_finalize_running;
    it must equal the whole-tensor pool_stack for every mode."""
    from angio_ft.models import PooledCLIP  # imported lazily (pulls transformers)

    torch.manual_seed(1)
    x = torch.randn(13, 6)
    for mode in ("max", "mean", "logsumexp"):
        running, aux = PooledCLIP._init_running(None, x.device, x.size(1), mode)
        for i in range(0, x.size(0), 4):
            running, aux = PooledCLIP._update_running(None, running, aux, x[i:i + 4], mode)
        out = PooledCLIP._finalize_running(None, running, aux, mode)
        assert torch.allclose(out, pool_stack(x, mode), atol=1e-5), mode


# ─────────────────────────────────────────────────────────────────────────────
# sinusoidal temporal encoding
# ─────────────────────────────────────────────────────────────────────────────

def test_sinusoidal_encoding_deterministic_and_shaped():
    pos = torch.arange(10)
    a = build_sinusoidal_position_encoding(pos, 16, torch.device("cpu"), torch.float32)
    b = build_sinusoidal_position_encoding(pos, 16, torch.device("cpu"), torch.float32)
    assert a.shape == (10, 16)
    assert torch.equal(a, b)                       # deterministic, no params
    # position 0 -> sin(0)=0 on even dims, cos(0)=1 on odd dims
    assert torch.allclose(a[0, 0::2], torch.zeros_like(a[0, 0::2]))
    assert torch.allclose(a[0, 1::2], torch.ones_like(a[0, 1::2]))


def test_sinusoidal_encoding_odd_dim_and_1d():
    # dim=1 special-case returns the raw position.
    pos = torch.tensor([0, 3, 7])
    pe1 = build_sinusoidal_position_encoding(pos, 1, torch.device("cpu"), torch.float32)
    assert torch.allclose(pe1[:, 0], pos.float())
    # odd dim must still fill without shape errors.
    pe = build_sinusoidal_position_encoding(pos, 7, torch.device("cpu"), torch.float32)
    assert pe.shape == (3, 7) and torch.isfinite(pe).all()
    with pytest.raises(ValueError):
        build_sinusoidal_position_encoding(pos.unsqueeze(0), 8, torch.device("cpu"), torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# contrastive losses
# ─────────────────────────────────────────────────────────────────────────────

def _normed(n, d, seed):
    torch.manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d), dim=-1)


def test_clip_loss_chunk_invariant():
    img, txt = _normed(8, 16, 2), _normed(8, 16, 3)
    scale = torch.tensor(10.0)
    ref = clip_loss_chunked(img, txt, scale, chunk=8)
    for c in (1, 2, 3, 5):
        assert torch.allclose(clip_loss_chunked(img, txt, scale, chunk=c), ref, atol=1e-5)


def test_clip_loss_perfect_alignment_is_low():
    emb = _normed(6, 16, 4)
    aligned = clip_loss_chunked(emb, emb, torch.tensor(100.0))
    misaligned = clip_loss_chunked(emb, emb.flip(0), torch.tensor(100.0))
    assert aligned.item() < 1e-2
    assert misaligned.item() > aligned.item()


def test_clip_loss_known_value():
    # Orthogonal-ish 2x2: identical embeds, scale 0 -> uniform logits ->
    # cross-entropy = ln(B) exactly.
    emb = _normed(4, 8, 5)
    loss = clip_loss_chunked(emb, emb, torch.tensor(0.0))
    assert math.isclose(loss.item(), math.log(4), rel_tol=1e-5)


def test_siglip_loss_chunk_invariant_and_bias_sign():
    img, txt = _normed(8, 16, 6), _normed(8, 16, 7)
    scale, bias = torch.tensor(10.0), torch.tensor(-10.0)
    ref = siglip_loss(img, txt, scale, bias, chunk=8)
    for c in (1, 2, 3, 5):
        assert torch.allclose(siglip_loss(img, txt, scale, bias, chunk=c), ref, atol=1e-5)
    # Aligned pairs should score better (lower loss) than reversed.
    aligned = siglip_loss(img, img, torch.tensor(50.0), bias)
    reversed_ = siglip_loss(img, img.flip(0), torch.tensor(50.0), bias)
    assert aligned.item() < reversed_.item()


# ─────────────────────────────────────────────────────────────────────────────
# hypothesis sets (prompt ensembling)
# ─────────────────────────────────────────────────────────────────────────────

def test_hypothesis_sets_default_is_single_pair():
    from angio_ft.qa_eval import make_yes_no_hypotheses, make_yes_no_hypothesis_sets

    q = "Is there a stent?"
    yes_h, no_h = make_yes_no_hypotheses(q)
    yes_list, no_list = make_yes_no_hypothesis_sets(q, ensemble=False)
    assert yes_list == [yes_h] and no_list == [no_h]     # legacy path untouched


def test_hypothesis_sets_ensemble_expands_and_keeps_legacy_first():
    from angio_ft.qa_eval import make_yes_no_hypotheses, make_yes_no_hypothesis_sets

    for q in ("Is there a stent?", "Any dissection?", "Totally novel question?"):
        yes_h, no_h = make_yes_no_hypotheses(q)
        yes_list, no_list = make_yes_no_hypothesis_sets(q, ensemble=True)
        assert len(yes_list) == len(no_list) >= 2
        assert yes_list[0] == yes_h and no_list[0] == no_h   # legacy sentence first
        assert len(set(yes_list)) == len(yes_list)           # no duplicate prompts


# ─────────────────────────────────────────────────────────────────────────────
# config_hash / run_dir_name (extracted from engine.py without importing torch-heavy deps)
# ─────────────────────────────────────────────────────────────────────────────

def _load_engine_pure_symbols():
    src = (_PKG_ROOT / "angio_ft" / "engine.py").read_text()
    tree = ast.parse(src)
    ns = {"hashlib": hashlib, "Optional": None}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("config_hash", "run_dir_name"):
            exec(compile(ast.Module([node], []), "engine.py", "exec"), ns)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_HASH_EXCLUDE":
            exec(compile(ast.Module([node], []), "engine.py", "exec"), ns)
    return ns


def _base_args():
    return types.SimpleNamespace(
        arch="clip", head_lr=1e-4, vision_backbone_lr=1e-5, text_lr=5e-5,
        pooling="max", epochs=5, batch_size=2, seed=42,
        temporal_mode="sinusoidal", max_sequences_per_study=4, max_frames_per_sequence=16,
        # perf-only / excluded:
        num_workers=4, io_threads=4, out_dir="x", run_name=None, resume=False, force=False,
    )


def test_config_hash_sensitivity():
    ns = _load_engine_pure_symbols()
    ch = ns["config_hash"]
    a = _base_args()
    base = ch(a)
    # result-affecting flags change the hash
    a.head_lr = 3e-4;  assert ch(a) != base
    a = _base_args(); a.seed = 7;  assert ch(a) != base
    a = _base_args(); a.epochs = 3;  assert ch(a) != base
    # perf-only flags do NOT
    a = _base_args(); a.num_workers = 32;  assert ch(a) == base
    a = _base_args(); a.io_threads = 16;  assert ch(a) == base


def test_run_dir_name_includes_hash():
    ns = _load_engine_pure_symbols()
    name = ns["run_dir_name"]("clip", "tempON", 5, 16, 4, 16, cfg_hash="deadbeef")
    assert name == "clip_tempON_5_16_4_16_hdeadbeef"
    # None limits render as the string "None"
    name2 = ns["run_dir_name"]("siglip", "tempOFF", 1, 2, None, None, cfg_hash="cafe")
    assert name2 == "siglip_tempOFF_1_2_None_None_hcafe"


# ─────────────────────────────────────────────────────────────────────────────
# GradCache contrastive accumulation == naive full-batch gradient
# ─────────────────────────────────────────────────────────────────────────────

def test_gradcache_matches_full_batch_gradient():
    import contextlib

    from angio_ft.contrastive_accum import gradcache_contrastive_step

    torch.manual_seed(0)
    D_in, D = 12, 8
    lin_i = torch.nn.Linear(D_in, D)
    lin_t = torch.nn.Linear(D_in, D)
    logit_scale = torch.nn.Parameter(torch.tensor(2.5))
    params = list(lin_i.parameters()) + list(lin_t.parameters()) + [logit_scale]

    # 3 micro-batches of 4 examples each -> effective batch of 12.
    micro = [
        {"img": torch.randn(4, D_in), "txt": torch.randn(4, D_in)}
        for _ in range(3)
    ]

    def forward_fn(mb):
        img = torch.nn.functional.normalize(lin_i(mb["img"]), dim=-1)
        txt = torch.nn.functional.normalize(lin_t(mb["txt"]), dim=-1)
        return img, txt, logit_scale

    def loss_fn(img, txt, scale):
        return clip_loss_chunked(img, txt, scale, chunk=4)

    # ---- Naive full-batch reference ----
    for p in params:
        p.grad = None
    big = {
        "img": torch.cat([m["img"] for m in micro], 0),
        "txt": torch.cat([m["txt"] for m in micro], 0),
    }
    img_all, txt_all, scale = forward_fn(big)
    loss_ref = loss_fn(img_all, txt_all, scale)
    loss_ref.backward()
    ref_grads = [p.grad.clone() for p in params]

    # ---- GradCache ----
    for p in params:
        p.grad = None
    loss_gc = gradcache_contrastive_step(
        micro_batches=micro,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        amp_ctx_factory=contextlib.nullcontext,
        scaler=None,
    )

    assert math.isclose(loss_gc.item(), loss_ref.item(), rel_tol=1e-5)
    for p, g_ref in zip(params, ref_grads):
        assert torch.allclose(p.grad, g_ref, atol=1e-5), \
            f"grad mismatch (max diff {(p.grad - g_ref).abs().max().item():.2e})"


# ─────────────────────────────────────────────────────────────────────────────
# checkpoint payload round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_payload_roundtrip(tmp_path):
    payload = {
        "step": 123, "epoch": 4,
        "model_state": {"w": torch.randn(3, 3)},
        "opt_state": {"lr": 1e-4},
        "config": {"arch": "clip", "embed_dim": 256},
    }
    p = tmp_path / "last.pt"
    torch.save(payload, p)
    loaded = torch.load(p, map_location="cpu")
    assert loaded["step"] == 123 and loaded["epoch"] == 4
    assert loaded["config"]["arch"] == "clip"
    assert torch.equal(loaded["model_state"]["w"], payload["model_state"]["w"])


# ─────────────────────────────────────────────────────────────────────────────
# siglip2 architecture plumbing
# ─────────────────────────────────────────────────────────────────────────────

def test_arch_choices_and_sigmoid_dispatch():
    from angio_ft.constants import ARCH_CHOICES, ARCH_DEFAULT_VIT, arch_uses_sigmoid

    assert ARCH_CHOICES == ("clip", "siglip", "siglip2", "xclip")
    assert not arch_uses_sigmoid("clip")
    assert arch_uses_sigmoid("siglip")
    assert arch_uses_sigmoid("siglip2")
    assert not arch_uses_sigmoid("xclip")  # X-CLIP trains with softmax loss
    # every arch has a default vision tower
    for arch in ARCH_CHOICES:
        assert ARCH_DEFAULT_VIT[arch]


def test_train_parser_accepts_siglip2_and_epoch_flags():
    from angio_ft.cli import build_train_argparser, build_validate_argparser

    ap = build_train_argparser()
    args = ap.parse_args([
        "--arch", "siglip2",
        "--meta_csv", "m.csv", "--reports_csv", "r.csv", "--base_frames_dir", "f",
        "--val_fraction", "0.2", "--epoch_qa_eval", "--validation_csv", "gt.csv",
    ])
    assert args.arch == "siglip2"
    assert args.vit_name is None          # resolved per-arch inside engine
    assert args.val_fraction == 0.2
    assert args.epoch_qa_eval is True
    assert args.validation_csv == "gt.csv"

    vp = build_validate_argparser()
    vargs = vp.parse_args(["--checkpoint", "x", "--out_csv", "o.csv", "--arch", "siglip2"])
    assert vargs.arch == "siglip2"


def test_config_hash_epoch_eval_flags_are_excluded_but_val_fraction_counts():
    ns = _load_engine_pure_symbols()
    ch = ns["config_hash"]

    def _args(**over):
        a = _base_args()
        a.val_fraction = 0.0
        a.epoch_qa_eval = False
        a.validation_csv = ""
        for k, v in over.items():
            setattr(a, k, v)
        return a

    base = ch(_args())
    # eval-only flags do NOT change the hash
    assert ch(_args(epoch_qa_eval=True)) == base
    assert ch(_args(validation_csv="other.csv")) == base
    # val_fraction changes the training data -> must change the hash
    assert ch(_args(val_fraction=0.1)) != base


# ─────────────────────────────────────────────────────────────────────────────
# epoch metrics CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def test_epoch_metrics_writer_incremental_and_resume_safe(tmp_path):
    import csv as _csv

    from angio_ft.engine import _METRIC_FIELDS, _METRIC_GROUPS, _EpochMetricsWriter

    path = tmp_path / "epoch_metrics.csv"
    w = _EpochMetricsWriter(path)

    # header covers losses + every baseline metric
    assert w.HEADER[:3] == ["epoch", "train_loss", "val_loss"]
    for g in ("ORIGINAL", "FLIPPED", "ALL_YES", "ALL_NO", "RANDOM"):
        for f in ("ACCURACY", "PRECISION", "RECALL", "F1", "TP", "TN", "FP", "FN"):
            assert f"{g}_{f}" in w.HEADER

    w.add(1, 2.5, 3.5, {"ORIGINAL_ACCURACY": 0.75, "ALL_YES_TP": 12})
    # incremental: row is on disk immediately, without any explicit flush/close
    with open(path) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert float(rows[0]["train_loss"]) == 2.5
    assert float(rows[0]["ORIGINAL_ACCURACY"]) == 0.75
    assert rows[0]["ALL_YES_TP"] == "12"
    assert rows[0]["RANDOM_F1"] == ""            # absent metrics stay blank

    # "resume": a new writer on the same file must append, not rewrite
    w2 = _EpochMetricsWriter(path)
    w2.add(2, 2.0, None, None)
    with open(path) as f:
        rows = list(_csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert rows[1]["val_loss"] == ""


def test_epoch_metrics_header_matches_parse_score_output_keys():
    """Every QA column the writer emits must be a key parse_score_output produces."""
    from angio_ft.engine import _EpochMetricsWriter
    from angio_ft.qa_eval import parse_score_output

    parsed = parse_score_output("")   # all keys present, values None
    for col in _EpochMetricsWriter.HEADER[3:]:
        assert col in parsed, col


# ─────────────────────────────────────────────────────────────────────────────
# study-level train/val split determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_val_split_is_seed_deterministic_and_disjoint():
    n, frac, seed = 23, 0.25, 42
    def split(seed):
        n_val = max(1, min(int(round(n * frac)), n - 1))
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=gen).tolist()
        return perm[:n_val], perm[n_val:]

    v1, t1 = split(seed)
    v2, t2 = split(seed)
    assert (v1, t1) == (v2, t2)                    # same seed -> same split
    assert not set(v1) & set(t1)                   # disjoint
    assert sorted(v1 + t1) == list(range(n))       # exhaustive
    v3, _ = split(seed + 1)
    assert v3 != v1                                # different seed -> different split


def test_video_frame_sampler_pads_and_subsamples():
    from angio_ft.models import sample_video_frame_indices

    # exact fit: identity
    assert sample_video_frame_indices(4, 4) == [0, 1, 2, 3]
    # shorter than clip: pad by repetition, order preserved, endpoints kept
    idx = sample_video_frame_indices(3, 8)
    assert len(idx) == 8 and idx[0] == 0 and idx[-1] == 2
    assert idx == sorted(idx)
    assert set(idx) == {0, 1, 2}
    # longer than clip: uniform subsample, endpoints kept
    idx = sample_video_frame_indices(100, 8)
    assert len(idx) == 8 and idx[0] == 0 and idx[-1] == 99
    assert idx == sorted(set(idx))
    # degenerate cases
    assert sample_video_frame_indices(1, 4) == [0, 0, 0, 0]
    assert sample_video_frame_indices(5, 1) == [0]
    assert sample_video_frame_indices(0, 4) == []


def test_xclip_end_to_end_encode(tmp_path):
    """X-CLIP video tower: joint clip forward, pad path, both encode APIs."""
    import torch
    from PIL import Image
    import numpy as np
    from transformers import XCLIPVisionConfig, XCLIPVisionModel, CLIPImageProcessor
    import angio_ft.models as M

    cfg = XCLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, image_size=32, patch_size=8, num_frames=4,
        mit_hidden_size=32, mit_intermediate_size=64,
        mit_num_hidden_layers=1, mit_num_attention_heads=2)
    tower = XCLIPVisionModel(cfg)
    proc = CLIPImageProcessor(size={"height": 32, "width": 32},
                              crop_size={"height": 32, "width": 32})

    # build a PooledCLIP without hitting from_pretrained: patch the loader
    orig = M.load_vision_tower
    M.load_vision_tower = lambda name: (tower, "video")
    try:
        import transformers
        bert_cfg = transformers.BertConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=64,
            max_position_embeddings=64)
        orig_auto = transformers.AutoModel.from_pretrained
        transformers.AutoModel.from_pretrained = staticmethod(
            lambda name: transformers.BertModel(bert_cfg))
        try:
            model = M.PooledCLIP(vit_name="tiny-xclip", text_model_name="tiny-bert",
                                 embed_dim=16, arch="xclip", temporal_mode="none")
        finally:
            transformers.AutoModel.from_pretrained = orig_auto
    finally:
        M.load_vision_tower = orig

    assert model.vision_feature_mode == "video"
    assert model.video_num_frames == 4

    # 3 frames on disk (< num_frames=4) -> pad-by-repetition path
    frames = []
    for k in range(3):
        p = tmp_path / f"f{k}.png"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(p)
        frames.append(p)

    device = torch.device("cpu")
    feat = model.encode_framepaths_pooled(frames, proc, device, pooling="max")
    assert feat.shape == (32,)

    emb, status = model.encode_sequence_from_frames(
        proc, frames, device, frame_chunk_size=16, max_frames=None)
    assert status == "ok" and emb.shape == (1, 16)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(1), atol=1e-5)


def _write_metadata_csv(d, rows):
    d.mkdir(parents=True, exist_ok=True)
    lines = ["Information,Value"] + [f"{k},{v}" for k, v in rows]
    (d / "metadata.csv").write_text("\n".join(lines) + "\n")


def test_validation_sequence_discovery_is_recursive(tmp_path):
    """Both the legacy flat layout and s01's nested <acc>/<sop>/ layout must
    be discovered; dirs with only one signal are returned for error logging."""
    from angio_ft.qa_eval import find_validation_sequence_dirs

    # legacy flat: sequence dir directly under data_dir
    flat = tmp_path / "seq_00"
    _write_metadata_csv(flat, [("AccessionNumber", "A1"), ("SOPInstanceUID", "1.1")])
    (flat / "frames").mkdir()

    # s01 nested: data_dir/<acc>/<sop>/{metadata.csv,frames/}
    nested = tmp_path / "ACC002" / "1.2.3.4"
    _write_metadata_csv(nested, [("AccessionNumber", "ACC002"), ("SOPInstanceUID", "1.2.3.4")])
    (nested / "frames").mkdir()

    # deeply buried (10 levels)
    deep = tmp_path.joinpath(*[f"lvl{i}" for i in range(10)]) / "seqX"
    _write_metadata_csv(deep, [("AccessionNumber", "A3"), ("SOPInstanceUID", "9.9")])
    (deep / "Frames").mkdir()  # capitalised variant

    # partial dirs: still discovered so errors are reported, not silent
    only_meta = tmp_path / "only_meta"
    _write_metadata_csv(only_meta, [("AccessionNumber", "A4"), ("SOPInstanceUID", "4.4")])
    only_frames = tmp_path / "only_frames"
    (only_frames / "frames").mkdir(parents=True)

    # pure wrapper dir (no signals) must NOT appear
    (tmp_path / "ACC002" / "empty_wrapper").mkdir()

    found = find_validation_sequence_dirs(tmp_path)
    assert flat in found and nested in found and deep in found
    assert only_meta in found and only_frames in found
    assert (tmp_path / "ACC002") not in found
    assert (tmp_path / "ACC002" / "empty_wrapper") not in found


def test_metadata_reader_accepts_s01_lowercase_keys(tmp_path):
    from angio_ft.qa_eval import read_metadata_key_value_csv

    # s01 fallback rows only (DICOM keyword rows dropped as nullish)
    d1 = tmp_path / "s1"
    _write_metadata_csv(d1, [("accession_number", "ACC009"), ("sop_instance_uid", "1.2.9")])
    acc, sop, status = read_metadata_key_value_csv(d1)
    assert (acc, sop, status) == ("ACC009", "1.2.9", "ok")

    # canonical keys still take precedence over fallbacks
    d2 = tmp_path / "s2"
    _write_metadata_csv(d2, [("AccessionNumber", "CANON"), ("accession_number", "FALLBACK"),
                             ("SOPInstanceUID", "2.2"), ("sop_instance_uid", "9.9")])
    acc, sop, status = read_metadata_key_value_csv(d2)
    assert (acc, sop, status) == ("CANON", "2.2", "ok")

    # still errors cleanly when identity is truly absent
    d3 = tmp_path / "s3"
    _write_metadata_csv(d3, [("PatientSex", "F")])
    acc, sop, status = read_metadata_key_value_csv(d3)
    assert acc is None and status.startswith("metadata_missing_accession")


# ─────────────────────────────────────────────────────────────────────────────
# grad-context behaviour: eval builds no graph, training numerics untouched
# ─────────────────────────────────────────────────────────────────────────────

def _build_tiny_pooled_clip():
    """Tiny random ViT ('cls' mode, the rad-dino path) + tiny BERT, one
    trainable ViT block — no downloads, mirrors the real default setup."""
    import transformers
    import angio_ft.models as M

    torch.manual_seed(42)
    vit_cfg = transformers.ViTConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=2, image_size=32, patch_size=8,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    tower = transformers.ViTModel(vit_cfg)
    bert_cfg = transformers.BertConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)

    orig_loader = M.load_vision_tower
    orig_auto = transformers.AutoModel.from_pretrained
    M.load_vision_tower = lambda name: (tower, "cls")
    transformers.AutoModel.from_pretrained = staticmethod(
        lambda name: transformers.BertModel(bert_cfg))
    try:
        return M.PooledCLIP(
            vit_name="tiny-vit", text_model_name="tiny-bert", embed_dim=16,
            arch="clip", temporal_mode="sinusoidal", vit_trainable_blocks=1)
    finally:
        M.load_vision_tower = orig_loader
        transformers.AutoModel.from_pretrained = orig_auto


class _IdentityProcessor:
    def __call__(self, images, return_tensors="pt", **kw):
        import numpy as np
        arrs = [np.asarray(im.resize((32, 32)), dtype=np.float32) / 255.0
                for im in images]
        px = torch.tensor(np.stack(arrs)).permute(0, 3, 1, 2)
        return {"pixel_values": (px - 0.5) / 0.5}


def test_eval_no_grad_builds_no_graph_but_training_grads_flow(tmp_path):
    """Regression for the enable_grad -> nullcontext fix: with a TRAINABLE ViT,
    an outer no_grad() must fully suppress graph construction (validation-loss
    passes previously recorded a transient graph), while normal training-mode
    forwards must still produce gradients for the trainable block."""
    import numpy as np
    from PIL import Image

    model = _build_tiny_pooled_clip()
    proc = _IdentityProcessor()
    rng = np.random.default_rng(0)
    frames = []
    for k in range(9):
        p = tmp_path / f"f{k}.png"
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(p)
        frames.append(p)

    device = torch.device("cpu")

    # eval under no_grad: pooled sequence feature must NOT carry a graph
    model.eval()
    with torch.no_grad():
        feat = model.encode_framepaths_pooled(
            frames, proc, device, chunk_size=4, pooling="max", io_threads=2)
    assert not feat.requires_grad and feat.grad_fn is None

    # training mode: grads must flow into the trainable ViT block
    model.train()
    feat = model.encode_framepaths_pooled(
        frames, proc, device, chunk_size=4, pooling="max", io_threads=2)
    assert feat.requires_grad
    feat.sum().backward()
    trainable_grads = [p.grad for p in model.vit.parameters()
                       if p.requires_grad and p.grad is not None]
    assert trainable_grads and any(g.abs().sum() > 0 for g in trainable_grads)
