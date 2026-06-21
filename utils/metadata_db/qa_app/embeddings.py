"""Query-time image-embedding functions (must match the model used at ingestion)."""

from __future__ import annotations

import logging
from typing import List, Optional

from . import config, deps
from .deps import RADDINO_OK, torch, AutoModel, AutoImageProcessor, np, PilImage
from .state import state

log = logging.getLogger(__name__)


class HFImageEmbeddingFunction:
    """
    Generic HuggingFace image-embedding function for DICOM frames.

    Loads the model named by an EMBEDDING_MODELS registry key (e.g. "rad-dino"
    → microsoft/rad-dino, "vit-base" → google/vit-base-patch16-224) via the
    transformers AutoImageProcessor + AutoModel API. The CLS token (index 0 of
    last_hidden_state) is extracted and L2-normalised so ChromaDB cosine
    similarity behaves like a dot product.

    The model used here at query time MUST match the model used to ingest the
    collection being searched — query vectors from a different model live in a
    different vector space and produce meaningless similarity results.
    """

    def __init__(self, model_key: str = config.DEFAULT_EMBEDDING_MODEL) -> None:
        if not RADDINO_OK:
            raise ImportError(
                "torch and transformers required: "
                "pip install torch transformers"
            )

        self.model_key = config.resolve_embedding_model(model_key)
        spec           = config.EMBEDDING_MODELS[self.model_key]
        self.model_id  = spec["hf_id"]
        self.label     = spec["label"]
        self.collection_name = spec["collection"]

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model     = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        log.info(f"Embedding model '{self.model_key}' ({self.model_id}) loaded on {self.device}")

    def name(self) -> str:
        """Return the registry key of this embedding function."""
        return self.model_key

    def __call__(self, input_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Embed a batch of images with the loaded model.

        Args:
            input_list: List of uint8 RGB numpy arrays (H×W×3)

        Returns:
            List of L2-normalised CLS-token embedding vectors (float32)
        """
        embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for img_array in input_list:
                # Ensure RGB uint8
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                img = PilImage.fromarray(img_array.astype("uint8"), "RGB")

                # AutoImageProcessor handles resize + normalisation
                inputs = self.processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                # CLS token from last hidden state → 768-dim
                cls_emb = outputs.last_hidden_state[:, 0, :]   # (1, 768)

                # L2 normalise
                cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=-1)
                embeddings.append(cls_emb.cpu().numpy().flatten())

        return embeddings


def get_embedding_model(model_key: Optional[str] = None) -> Optional["HFImageEmbeddingFunction"]:
    """
    Get or lazy-initialise the embedding model for the given registry key.

    Models are cached per key in state.embedding_models so each is loaded at most
    once. Returns None if dependencies are missing or the model fails to load.
    """
    key = config.resolve_embedding_model(model_key)
    cached = state.embedding_models.get(key)
    if cached is not None:
        return cached

    if not RADDINO_OK:
        log.warning("Embedding model dependencies not available (torch/transformers)")
        return None
    try:
        ef = HFImageEmbeddingFunction(key)
        state.embedding_models[key] = ef
        return ef
    except Exception as exc:
        log.error(f"Failed to load embedding model '{key}': {exc}")
        return None


def get_raddino_model() -> Optional["HFImageEmbeddingFunction"]:
    """Back-compat shim: return the default RAD-DINO embedding model."""
    return get_embedding_model("rad-dino")
