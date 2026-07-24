"""
inference_demo.py

Minimal single-image inference example against a fine-tuned checkpoint.
Useful for a README GIF/screenshot or a live demo in an interview.

    python src/inference_demo.py --checkpoint checkpoints/forgery-lora/epoch2 \
        --image data/synthetic/images/sample_00013.png
"""
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

DEFAULT_INSTRUCTION = (
    "Examine this identity document image. Is it authentic or has it been "
    "tampered with? If tampered, describe what was altered and where."
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = AutoModelForVision2Seq.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": args.instruction}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    print(f"Image: {args.image}")
    print(f"Model output:\n{text}")

if __name__ == "__main__":
    main()
