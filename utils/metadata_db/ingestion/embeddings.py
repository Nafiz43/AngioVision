"""RAD-DINO embedding model loading and ChromaDB collection setup."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .config import (
    CHROMA_AVAILABLE,
    RAD_DINO_AVAILABLE,
    RAD_DINO_MODEL_ID,
    CHROMA_COLLECTION,
    EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    resolve_embedding_model,
)

if CHROMA_AVAILABLE:
    import chromadb

if RAD_DINO_AVAILABLE:
    import torch
    from PIL import Image as PILImage
    from transformers import AutoImageProcessor, AutoModel

log = logging.getLogger(__name__)


def load_embedding_model(model_key: str = DEFAULT_EMBEDDING_MODEL, device: Optional[str] = None):
    """
    Load the embedding model named by an EMBEDDING_MODELS key exactly once and
    return (model, processor, device).

    Both microsoft/rad-dino and google/vit-base-patch16-224 are ViTs that expose a
    CLS token at index 0 of last_hidden_state, which embed_frames() extracts and
    L2-normalises. RAD-DINO is pre-trained on radiology images; ViT-Base is the
    generic ImageNet model.
    """
    if not RAD_DINO_AVAILABLE:
        raise ImportError(
            "transformers or torch not installed. "
            "Run: pip install transformers torch pillow"
        )

    key   = resolve_embedding_model(model_key)
    hf_id = EMBEDDING_MODELS[key]["hf_id"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Loading embedding model '{key}' ({hf_id}) on {device} (once) …")
    processor = AutoImageProcessor.from_pretrained(hf_id)
    model     = AutoModel.from_pretrained(hf_id)
    model.eval()
    model = model.to(device)
    log.info(f"Embedding model '{key}' ready.")
    return model, processor, device


def load_rad_dino_model(device: Optional[str] = None):
    """Back-compat wrapper: load the default RAD-DINO model."""
    return load_embedding_model("rad-dino", device=device)


def embed_frames(
    frames: list[np.ndarray],
    model,
    processor,
    device: str,
) -> list[list[float]]:
    """
    Embed a list of uint8 RGB numpy frames with the given model.

    The CLS token (index 0 of last_hidden_state) is extracted and L2-normalised
    so ChromaDB cosine similarity == dot product. Returns one float vector per
    frame (768-dim for both RAD-DINO and ViT-Base).
    """
    pil_images = [PILImage.fromarray(f) for f in frames]

    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)

    return cls_emb.cpu().float().numpy().tolist()


# Back-compat alias (RAD-DINO-specific name retained for older imports).
embed_frames_rad_dino = embed_frames


def setup_chromadb(chroma_path: Path, collection_name: str = CHROMA_COLLECTION):
    """
    Initialise a persistent ChromaDB client and return (client, collection).

    No embedding function is attached to the collection — embeddings are
    pre-computed by embed_frames() and passed via the embeddings= kwarg in
    collection.add(). Each embedding model uses its own collection name.
    """
    if not CHROMA_AVAILABLE:
        raise ImportError("chromadb not installed. Run: pip install chromadb")
    chroma_path.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(
        f"ChromaDB '{collection_name}' at {chroma_path} — "
        f"existing items: {collection.count():,}"
    )
    return client, collection
