#!/usr/bin/env python3
"""Study<->report retrieval eval on the seeded val split of a training run.

Diagnoses contrastive training independently of the QA readout: good
recall@K here + chance QA means the QA prompt/readout is the problem,
not the training.

    python3 eval_retrieval.py --run_dir checkpoints/<run> [--checkpoint epoch_3.pt]
"""
import argparse, shlex
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from angio_ft.cli import build_train_argparser
from angio_ft.data import StudyDataset, collate_fn
from angio_ft.models import PooledCLIP
from angio_ft.common import get_vit_processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default="last.pt", help="file name inside run_dir, or absolute path")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    # Replay the exact training invocation so dataset/split/model match.
    argv = shlex.split(run_dir.joinpath("train_cmd.sh").read_text().splitlines()[1])
    targs = build_train_argparser().parse_args(argv[2:])  # strip python + train.py

    ckpt_path = Path(a.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / a.checkpoint
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    print(f"[INFO] checkpoint: {ckpt_path} (epoch {payload.get('epoch')}) arch={cfg['arch']}")

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = PooledCLIP(
        vit_name=cfg["vit_name"], text_model_name=cfg["bert_name"], embed_dim=cfg["embed_dim"],
        arch=cfg["arch"], frame_pooling=cfg["frame_pooling"], sequence_pooling=cfg["sequence_pooling"],
        temporal_mode=cfg["temporal_mode"], temporal_on_frames=cfg["temporal_on_frames"],
        temporal_on_sequences=cfg["temporal_on_sequences"],
        frame_temporal_scale=cfg["frame_temporal_scale"],
        sequence_temporal_scale=cfg["sequence_temporal_scale"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    dataset = StudyDataset(
        meta_csv=Path(targs.meta_csv), reports_csv=Path(targs.reports_csv),
        base_frames_dir=Path(targs.base_frames_dir), report_text_col=targs.report_text_col,
        anon_col=targs.anon_col, sop_col=targs.sop_col, report_type_col=targs.report_type_col,
        min_frames_per_sequence=targs.min_frames_per_sequence,
        max_sequences_per_study=targs.max_sequences_per_study,
        max_frames_per_sequence=targs.max_frames_per_sequence,
        drop_missing_reports=not targs.keep_missing_reports,
        report_sampling=targs.report_sampling, report_sampling_seed=targs.report_sampling_seed,
        frame_keep_fraction=0.2 if getattr(targs, "frames_20pct", False) else 1.0,
        frame_keep_seed=targs.seed,
    )
    # Identical split logic to engine.train()
    n = len(dataset)
    n_val = max(1, min(int(round(n * float(targs.val_fraction))), n - 1))
    gen = torch.Generator().manual_seed(int(targs.seed))
    perm = torch.randperm(n, generator=gen).tolist()
    val_loader = DataLoader(Subset(dataset, perm[:n_val]), batch_size=a.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"[INFO] val studies: {n_val} / {n}")

    processor = get_vit_processor(cfg["vit_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["bert_name"])

    img_all, txt_all = [], []
    with torch.no_grad():
        for b in val_loader:
            if not b["text"]:
                continue
            img, txt, _ = model(
                batch_sequences=b["sequences"], texts=b["text"], processor=processor,
                tokenizer=tokenizer, device=device,
                frame_chunk_size=targs.frame_chunk_size,
                vit_image_size=cfg.get("vit_image_size"), io_threads=targs.io_threads,
            )
            img_all.append(img.cpu()); txt_all.append(txt.cpu())
    img_all = torch.cat(img_all); txt_all = torch.cat(txt_all)
    N = img_all.size(0)
    sim = img_all @ txt_all.T  # (N studies, N reports), already L2-normalized

    def recalls(s):
        order = s.argsort(dim=1, descending=True)
        ranks = (order == torch.arange(s.size(0))[:, None]).float().argmax(dim=1)
        out = {f"R@{k}": (ranks < k).float().mean().item() for k in (1, 5, 10)}
        out["median_rank"] = ranks.median().item() + 1
        return out

    i2t, t2i = recalls(sim), recalls(sim.T)
    print(f"[RESULT] N={N}  chance R@1={1/N:.3f} R@5={5/N:.3f} R@10={10/N:.3f}")
    print("[RESULT] study->report: " + "  ".join(f"{k}={v:.3f}" for k, v in i2t.items()))
    print("[RESULT] report->study: " + "  ".join(f"{k}={v:.3f}" for k, v in t2i.items()))


if __name__ == "__main__":
    main()
