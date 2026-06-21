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
)

if CHROMA_AVAILABLE:
    import chromadb

if RAD_DINO_AVAILABLE:
    import torch
    from PIL import Image as PILImage
    from transformers import AutoImageProcessor, AutoModel

log = logging.getLogger(__name__)


def load_rad_dino_model(device: Optional[str] = None):
    """
    Load microsoft/rad-dino exactly once and return (model, processor, device).

    RAD-DINO is a ViT pre-trained on a large collection of radiology images
    (chest X-ray, CT, MRI, fluoroscopy/DSA). It produces 768-dim CLS-token
    embeddings that are substantially more discriminative for angiographic
    sequences than generic CLIP features.
    """
    if not RAD_DINO_AVAILABLE:
        raise ImportError(
            "transformers or torch not installed. "
            "Run: pip install transformers torch pillow"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Loading {RAD_DINO_MODEL_ID} on {device} (once) …")
    processor = AutoImageProcessor.from_pretrained(RAD_DINO_MODEL_ID)
    model     = AutoModel.from_pretrained(RAD_DINO_MODEL_ID)
    model.eval()
    model = model.to(device)
    log.info("RAD-DINO model ready  (embedding dim: 768).")
    return model, processor, device


def embed_frames_rad_dino(
    frames: list[np.ndarray],
    model,
    processor,
    device: str,
) -> list[list[float]]:
    """
    Embed a list of uint8 RGB numpy frames with RAD-DINO.

    The CLS token (index 0 of last_hidden_state) is extracted and L2-normalised
    so ChromaDB cosine similarity == dot product. Returns one 768-dim float
    vector per frame.
    """
    pil_images = [PILImage.fromarray(f) for f in frames]

    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)

    return cls_emb.cpu().float().numpy().tolist()


def setup_chromadb(chroma_path: Path):
    """
    Initialise a persistent ChromaDB client and return (client, collection).

    No embedding function is attached to the collection — embeddings are
    pre-computed by embed_frames_rad_dino() and passed via the embeddings=
    kwarg in collection.add().
    """
    if not CHROMA_AVAILABLE:
        raise ImportError("chromadb not installed. Run: pip install chromadb")
    chroma_path.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(
        f"ChromaDB '{CHROMA_COLLECTION}' at {chroma_path} — "
        f"existing items: {collection.count():,}"
    )
    return client, collection
