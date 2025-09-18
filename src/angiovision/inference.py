"""Core inference utilities for running Qwen2.5-VL on angiography images."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .qwen_vl_utils import process_vision_info

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


ImageInput = Union[str, Image.Image]


@dataclass
class GenerationConfig:
    """Configuration parameters for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None


class QwenVisualLanguageClient:
    """Convenience wrapper around the Qwen2.5-VL model for question answering."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device_map: Optional[Union[str, Iterable[int]]] = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype or self._infer_default_dtype()
        self.generation_config = generation_config or GenerationConfig()

        self.model: Optional[Qwen2_5_VLForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None

    @staticmethod
    def _infer_default_dtype() -> torch.dtype:
        if torch.cuda.is_available():
            # Qwen models are optimised for bfloat16 on modern GPUs.
            return torch.bfloat16
        if torch.backends.mps.is_available():
            return torch.float16
        return torch.float32

    def load(self) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
        if self.model is None or self.processor is None:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        return self.model, self.processor

    def unload(self) -> None:
        """Free the loaded model to release GPU memory."""
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()

    @staticmethod
    def _ensure_image(image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def _build_messages(
        self,
        question: str,
        image_inputs: Sequence[ImageInput],
        system_prompt: str,
        context: Optional[Sequence[str]] = None,
    ) -> List[dict]:
        images = [self._ensure_image(item) for item in image_inputs]

        user_content: List[dict] = [
            {"type": "image", "image": img} for img in images
        ]
        if context:
            for text in context:
                if text.strip():
                    user_content.append({"type": "text", "text": text})
        user_content.append({"type": "text", "text": question})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        return messages

    def ask(
        self,
        question: str,
        image_inputs: Sequence[ImageInput],
        system_prompt: str = "You are a concise interventional radiology assistant.",
        context: Optional[Sequence[str]] = None,
    ) -> str:
        model, processor = self.load()
        messages = self._build_messages(question, image_inputs, system_prompt, context)

        prompt = processor.apply_chat_template(
            conversations=messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        vision_inputs = process_vision_info(messages)

        inputs = processor(
            text=[prompt],
            images=vision_inputs["images"],
            videos=vision_inputs["videos"],
            return_tensors="pt",
        )

        inputs = {
            key: value.to(model.device, dtype=self._maybe_cast_dtype(value))
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.generation_config.max_new_tokens,
                do_sample=self.generation_config.temperature > 0,
                temperature=self.generation_config.temperature if self.generation_config.temperature > 0 else None,
                top_p=self.generation_config.top_p,
                top_k=self.generation_config.top_k,
                repetition_penalty=self.generation_config.repetition_penalty,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        generated_tokens = output_ids[:, inputs["input_ids"].shape[-1] :]
        text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return text.strip()

    def _maybe_cast_dtype(self, tensor: torch.Tensor) -> Optional[torch.dtype]:
        if tensor.dtype in (torch.float32, torch.float64):
            return self.torch_dtype
        return None


def ask_qwen_about_images(
    question: str,
    image_inputs: Sequence[ImageInput],
    system_prompt: str = "You are a concise interventional radiology assistant.",
    context: Optional[Sequence[str]] = None,
    generation_config: Optional[GenerationConfig] = None,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    client = QwenVisualLanguageClient(
        model_id=model_id,
        generation_config=generation_config,
    )
    return client.ask(
        question=question,
        image_inputs=image_inputs,
        system_prompt=system_prompt,
        context=context,
    )
