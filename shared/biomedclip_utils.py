"""BioMedCLIP model loading, compatibility patching, and zero-shot classification."""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import open_clip
    import open_clip.factory as ocf
except ImportError:
    open_clip = None  # type: ignore[assignment]
    ocf = None  # type: ignore[assignment]


# ---------- Checkpoint compatibility patch ----------

def patch_openclip_for_position_ids() -> None:
    """Patch ``open_clip.factory.load_checkpoint`` to handle missing
    ``position_ids`` buffers in BioMedCLIP checkpoints.
    """
    if ocf is None:
        return
    if getattr(ocf, "_position_ids_patch_applied", False):
        return

    def _load_checkpoint_compat(model, checkpoint_path, strict=True, **kwargs):
        try:
            state_dict = ocf.load_state_dict(checkpoint_path, **kwargs)
        except TypeError:
            state_dict = ocf.load_state_dict(checkpoint_path)

        key = "text.transformer.embeddings.position_ids"
        model_sd = model.state_dict()

        if key not in state_dict and key in model_sd:
            state_dict[key] = model_sd[key]
        if key in state_dict and key not in model_sd:
            del state_dict[key]

        ocf.resize_pos_embed(state_dict, model)
        ocf.resize_text_pos_embed(state_dict, model)

        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        return incompatible_keys

    ocf.load_checkpoint = _load_checkpoint_compat
    ocf._position_ids_patch_applied = True


# ---------- Model loading ----------

def to_hf_hub_id(model_arg: str) -> str:
    """Ensure *model_arg* has the ``hf-hub:`` prefix required by open_clip."""
    if model_arg.startswith("hf-hub:"):
        return model_arg
    return f"hf-hub:{model_arg}"


def load_biomedclip_model(
    model_name: str,
    device: str = None,
) -> Tuple[Any, Any, Any, torch.device]:
    """Load a BioMedCLIP model via ``open_clip``.

    Returns ``(model, preprocess_val, tokenizer, device)``.
    """
    if open_clip is None:
        raise ImportError("open_clip_torch is required: pip install open_clip_torch transformers")

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    hf_id = to_hf_hub_id(model_name)

    print(f"Loading BioMedCLIP model: {model_name} on {device_t}")
    print(f"Using open_clip id: {hf_id}")

    patch_openclip_for_position_ids()

    model, _, preprocess_val = open_clip.create_model_and_transforms(hf_id)
    tokenizer = open_clip.get_tokenizer(hf_id)

    model = model.to(device_t)
    model.eval()

    print("BioMedCLIP model loaded successfully")
    return model, preprocess_val, tokenizer, device_t


# ---------- Zero-shot classification ----------

def biomedclip_classify(
    image_path,
    question: str,
    options: List[str],
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """Use BioMedCLIP to pick the best *option* for a given *question* and image."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        prompts = [f"{question} {opt}" for opt in options]
        text_tokens = tokenizer(prompts).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = 100.0 * image_features @ text_features.T
            probs = torch.softmax(similarity, dim=-1).detach().cpu().numpy()[0]

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx] * 100.0)
        answer = options[best_idx]

        all_scores = {opt: float(p * 100.0) for opt, p in zip(options, probs)}
        top_indices = np.argsort(probs)[-3:][::-1]
        evidence = [f"{options[i]}: {probs[i]*100.0:.1f}%" for i in top_indices]

        return {
            "answer": answer,
            "confidence": round(confidence, 2),
            "evidence": evidence,
            "all_scores": all_scores,
            "notes": "BioMedCLIP zero-shot classification",
        }
    except Exception as e:
        return {
            "answer": "Error",
            "confidence": 0,
            "evidence": [],
            "all_scores": {},
            "notes": f"BioMedCLIP error: {str(e)[:200]}",
        }
