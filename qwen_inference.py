# qwen25_vl_infer.py
from typing import List, Union
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info  # helper for vision inputs

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

def load_model(model_id: str = MODEL_ID):
    device_map = "auto"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def _as_image(obj: Union[str, Image.Image]) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    return Image.open(obj).convert("RGB")

def ask_qwen_about_images(
    question: str,
    image_inputs: List[Union[str, Image.Image]],
    system: str = "You are a concise interventional radiology assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.2,
):
    """
    image_inputs: list of file paths or PIL Images
    """
    model, processor = load_model()

    images = [_as_image(x) for x in image_inputs]

    # Build chat-style messages (supports multiple images interleaved with text)
    # You can add more text turns if needed.
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": question}]
            ),
        },
    ]

    # Convert to a single template string for generation
    prompt = processor.apply_chat_template(
        conversations=messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Extract pixel values from the messages (helper handles resizing/packing)
    image_inputs_processed = process_vision_info(messages)

    inputs = processor(
        text=[prompt],
        images=image_inputs_processed["images"],
        videos=image_inputs_processed["videos"],
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device, dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else None)
              for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    text = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
    return text.strip()

if __name__ == "__main__":
    # Example usage (replace with your angio frames):
    images = [
        "sample_angio_frame_1.png",
        # "sample_angio_frame_2.png",  # you can pass multiple
    ]
    question = (
        "What arteries are being catheterized in this image? "
        "Answer with specific vessel names (e.g., RCA, LAD, LCx, femoral, radial) "
        "and a short rationale. If uncertain, state uncertainty."
    )
    print(ask_qwen_about_images(question, images))
