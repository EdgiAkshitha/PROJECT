"""
dataset.py

Wraps the synthetic manifest (see data_synthesis.py) into a PyTorch Dataset
that yields VLM chat-format examples:

    {"image": PIL.Image, "messages": [
        {"role": "user", "content": <instruction>},
        {"role": "assistant", "content": <target_text>}
    ]}

Designed to plug into a Qwen2-VL / LLaVA-style processor's chat template.
Kept deliberately framework-light (no HF imports here) so it's easy to
adapt to whichever VLM checkpoint you pick.
"""

import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ForgeryVLMDataset(Dataset):
    def __init__(self, manifest_path: str, images_dir: str, split="train", val_frac=0.15, seed=42):
        self.images_dir = Path(images_dir)
        records = [json.loads(l) for l in open(manifest_path)]

        import random
        rng = random.Random(seed)
        rng.shuffle(records)
        n_val = int(len(records) * val_frac)
        self.records = records[n_val:] if split == "train" else records[:n_val]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image = Image.open(self.images_dir / r["id"]).convert("RGB")
        messages = [
            {"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["target_text"]},
        ]
        return {
            "image": image,
            "messages": messages,
            "label": r["label"],
            "forgery_type": r["forgery_type"],
            "bbox": r["bbox"],
            "id": r["id"],
        }


def collate_for_processor(batch, processor, max_length=512):
    """
    Generic collate_fn: applies the VLM processor's chat template to build
    input_ids/pixel_values/labels. Works for most HF VLM processors that
    expose apply_chat_template + a callable(text=..., images=...) interface
    (Qwen2-VL, LLaVA-NeXT, Idefics2, etc. all follow this pattern).
    """
    texts, images = [], []
    for ex in batch:
        prompt = processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        texts.append(prompt)
        images.append(ex["image"])

    enc = processor(text=texts, images=images, padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt")
    # Standard causal-LM SFT: labels = input_ids, with padding masked to -100
    labels = enc["input_ids"].clone()
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
    enc["labels"] = labels
    return enc
