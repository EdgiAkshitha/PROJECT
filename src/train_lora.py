"""
train_lora.py

LoRA / PEFT supervised fine-tuning of an open-weight Vision-Language Model
on the synthetic document-forgery dataset.

Default base model: Qwen/Qwen2-VL-2B-Instruct
  - small enough to fine-tune on a single free-tier Colab T4/L4 with LoRA
  - strong OCR + document understanding out of the box, so fine-tuning is
    teaching a *new task* (forgery reasoning) rather than teaching it to
    read, which is a more realistic/achievable research goal for an
    internship-scale project.

Swap --base_model for e.g. "llava-hf/llava-v1.6-mistral-7b-hf" if you have
more VRAM, or a smaller "Qwen/Qwen2-VL-2B-Instruct" for CPU/low-VRAM.

Run (on a machine/Colab with a GPU and internet access to huggingface.co):

    pip install -r requirements.txt
    python src/data_synthesis.py --n_authentic 500 --n_forged 500
    python src/train_lora.py --epochs 3 --batch_size 2 --lr 2e-4

This script is intentionally NOT executed inside the sandbox that produced
this project (no GPU / no huggingface.co egress there) -- run it in Colab
or on your own machine. It is written to run as-is once dependencies and a
GPU are available.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from dataset import ForgeryVLMDataset, collate_for_processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--manifest", default="data/synthetic/manifest.jsonl")
    ap.add_argument("--images_dir", default="data/synthetic/images")
    ap.add_argument("--output_dir", default="checkpoints/forgery-lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_lora] device={device}, base_model={args.base_model}")

    processor = AutoProcessor.from_pretrained(args.base_model)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Target the LLM attention/MLP projections (standard for Qwen2-VL/LLaVA-style
    # decoders). Vision tower is left frozen -- we're teaching new *reasoning*
    # over features the pretrained vision encoder already extracts well,
    # which is both cheaper and less prone to catastrophic forgetting of
    # low-level OCR/vision skills than fine-tuning the vision tower too.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = ForgeryVLMDataset(args.manifest, args.images_dir, split="train")
    val_ds = ForgeryVLMDataset(args.manifest, args.images_dir, split="val")
    print(f"[train_lora] train={len(train_ds)} val={len(val_ds)}")

    collate_fn = lambda b: collate_for_processor(b, processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running_loss += loss.item()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(f"[epoch {epoch}] step {step}/{total_steps} loss={running_loss:.4f}")
                running_loss = 0.0

        # cheap end-of-epoch val loss
        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_loss += model(**batch).loss.item()
                n += 1
        print(f"[epoch {epoch}] val_loss={val_loss / max(n,1):.4f}")
        model.train()

        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(os.path.join(args.output_dir, f"epoch{epoch}"))
        processor.save_pretrained(os.path.join(args.output_dir, f"epoch{epoch}"))

    print(f"[train_lora] Done. LoRA adapters saved under {args.output_dir}/")


if __name__ == "__main__":
    main()
