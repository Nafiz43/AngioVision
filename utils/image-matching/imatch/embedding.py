"""Embedding-model loading and temporal sequence pooling.

torch / transformers / open_clip are imported lazily inside
``load_embedding_model`` so the rest of the pipeline (CSV loading, splits,
aggregation) stays importable on machines without them.
"""

import logging
import sys
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

from .config import HF_ALIASES, OPENCLIP_ALIASES


def load_embedding_model(model_name: str, device: Optional[str] = None) -> tuple[Callable, str, int]:
    """
    Load by alias or HF model ID.
    Returns (embed_fn, model_id, emb_dim).
    embed_fn: (frames: list[np.ndarray], batch_size: int) -> list[list[float]]
    All embeddings are L2-normalised.
    """
    try:
        import torch
        from PIL import Image as PILImage
    except ImportError:
        print("ERROR: pip install torch pillow"); sys.exit(1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name in OPENCLIP_ALIASES:
        try:
            import open_clip
        except ImportError:
            print("ERROR: pip install open-clip-torch"); sys.exit(1)
        arch, pretrained = OPENCLIP_ALIASES[model_name]
        log.info(f"Loading OpenCLIP {arch}/{pretrained} on {device} …")
        oc_model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        oc_model = oc_model.eval().to(device)
        model_id = f"openclip/{arch}/{pretrained}"

        def embed_fn(frames, batch_size=16):
            all_embs = []
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                tensors = torch.stack([preprocess(PILImage.fromarray(f)) for f in batch]).to(device)
                with torch.no_grad():
                    feats = oc_model.encode_image(tensors)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                all_embs.extend(feats.cpu().float().numpy().tolist())
            return all_embs
    else:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            print("ERROR: pip install transformers torch pillow"); sys.exit(1)
        hf_id = HF_ALIASES.get(model_name, model_name); model_id = hf_id
        log.info(f"Loading HuggingFace '{hf_id}' on {device} …")
        processor = AutoImageProcessor.from_pretrained(hf_id)
        hf_model  = AutoModel.from_pretrained(hf_id).eval().to(device)

        def embed_fn(frames, batch_size=16):
            all_embs = []
            for i in range(0, len(frames), batch_size):
                batch  = frames[i:i + batch_size]
                pil    = [PILImage.fromarray(f) for f in batch]
                inputs = processor(images=pil, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = hf_model(**inputs)
                    cls = out.last_hidden_state[:, 0, :]
                    cls = cls / cls.norm(dim=-1, keepdim=True)
                all_embs.extend(cls.cpu().float().numpy().tolist())
            return all_embs

    test_emb = embed_fn([np.zeros((224, 224, 3), dtype=np.uint8)], batch_size=1)
    emb_dim  = len(test_emb[0])
    log.info(f"Model ready — id={model_id}  dim={emb_dim}")
    return embed_fn, model_id, emb_dim


def compute_temporal_embedding(frame_embeddings: list[list[float]]) -> list[float]:
    """
    Aggregate N per-frame embeddings into ONE temporal sequence descriptor.

    mean pool → average spatial appearance over the run  (anatomy)
    std  pool → frame-to-frame variation                 (contrast dynamics)

    Concatenated and L2-normalised → 2×D vector.
    N=1: std=0, result = [frame_emb, 0…0] normalised (spatial only).
    """
    embs      = np.array(frame_embeddings, dtype=np.float32)
    mean_pool = embs.mean(axis=0)
    std_pool  = embs.std(axis=0)
    descriptor = np.concatenate([mean_pool, std_pool])
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor /= norm
    return descriptor.tolist()
