import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import csv
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# ==========  CONFIGURATION ==========
# Path to base processor model (cloned from Hugging Face)
base_processor_path = "/models/videollama3-base"

# Path to your fine-tuned model checkpoint
finetuned_model_dir = "/VideoLLaMA3/work_dirs_seq/videollama3_qwen2.5_2b/stage1_a100/checkpoint-1823"

# Folder containing DSA video files (change to your actual path)
video_folder = "/VideoLLaMA3/test_seq/videos"

# Output CSV file
output_csv = "dsa_video_responses.csv"

# Question for the model
question = "What artery is being catheterized in this DSA sequence?"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========  LOAD MODEL & PROCESSOR ==========
print("🔄 Loading processor from base model...")
processor = AutoProcessor.from_pretrained(base_processor_path, trust_remote_code=True)

print(" Loading fine-tuned model from checkpoint...")
model = AutoModelForCausalLM.from_pretrained(
    finetuned_model_dir,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16 
)
model.eval()

# ==========  INFERENCE FUNCTION ==========
def analyze_video(video_path):
    conversation = [
        {"role": "system", "content": "You are a medical imaging assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": video_path,
                        "fps": 0.2,
                        "max_frames": 4
                    }
                },
                {"type": "text", "text": question}
            ],
        },
    ]

    try:
        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16 if device == "cuda" else torch.float32)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64)
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Clear memory after processing
        del inputs, generated_ids
        torch.cuda.empty_cache()

        return response

    except Exception as e:
        return f" Error: {str(e)}"

# ==========  PROCESS VIDEO FOLDER ==========
results = []

print(f" Looking for videos in: {video_folder}")
for filename in sorted(os.listdir(video_folder)):
    if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        video_path = os.path.join(video_folder, filename)
        print(f" Processing: {filename}")
        result = analyze_video(video_path)
        print(f" Response: {result}\n")
        results.append({"video": filename, "response": result})

# ========== SAVE TO CSV ==========
print(f" Saving results to: {output_csv}")
with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["video", "response"])
    writer.writeheader()
    writer.writerows(results)

print("\n All done. Inference complete!")

# ========== CLEANUP ==========
del model, processor
torch.cuda.empty_cache()

