"""RAD-DINO query-time embedding function (must match the ingestion model)."""

from __future__ import annotations

import logging
from typing import List, Optional

from . import deps
from .deps import RADDINO_OK, torch, AutoModel, AutoImageProcessor, np, PilImage
from .state import state

log = logging.getLogger(__name__)


class RADDINOEmbeddingFunction:
    """
    RAD-DINO embedding function for medical imaging DICOM frames.

    Uses microsoft/rad-dino (ViT trained on ~1M radiology images) via HuggingFace
    transformers. MUST match the embedding model used during ChromaDB ingestion so
    that query embeddings live in the same vector space as indexed embeddings.

    The CLS token (index 0 of last_hidden_state) is extracted and L2-normalised as
    the 768-dimensional embedding vector.
    """

    # HuggingFace model identifier — MUST match ingestion pipeline
    MODEL_ID = "microsoft/rad-dino"

    @classmethod
    def name(cls) -> str:
        """Return the name of this embedding function."""
        return "raddino"

    def __init__(self, model_id: str = MODEL_ID) -> None:
        if not RADDINO_OK:
            raise ImportError(
                "torch and transformers required: "
                "pip install torch transformers"
            )

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model     = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        log.info(f"RAD-DINO ({model_id}) loaded on {self.device}")

    def __call__(self, input_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Embed a batch of images using RAD-DINO.

        Args:
            input_list: List of uint8 RGB numpy arrays (H×W×3)

        Returns:
            List of L2-normalised 768-dimensional embedding vectors (float32)
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


def get_raddino_model() -> Optional["RADDINOEmbeddingFunction"]:
    """Get or lazy-initialize the RAD-DINO embedding model (singleton pattern)."""
    if state.raddino_model is None:
        if not RADDINO_OK:
            log.warning("RAD-DINO dependencies not available")
            return None
        try:
            state.raddino_model = RADDINOEmbeddingFunction()
        except Exception as exc:
            log.error(f"Failed to load RAD-DINO model: {exc}")
            return None
    return state.raddino_model
