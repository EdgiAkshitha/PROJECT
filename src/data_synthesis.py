"""
data_synthesis.py

Generates a labeled synthetic dataset of "tampered" vs "authentic" ID-style
documents from a folder of clean source document/ID images.

Why synthetic data at all (interview talking point):
  Real labeled forged-ID datasets are scarce, sensitive, and mostly private
  (that's precisely why HyperVerge needs research here). The standard
  research pattern -- and what the JD calls "synthetic data generation and
  data augmentation strategies" -- is to programmatically manufacture
  realistic tampering on clean documents so we have ground truth for both
  the tampered region (a mask) and a natural-language description of the
  tampering, which becomes the VLM's training target.

Forgery types implemented (each is a well-known real-world tampering mode
in identity-fraud literature):
  1. copy_move   - a region of the image (e.g. a digit) is copied and pasted
                    elsewhere in the same image, at a slightly different
                    position (classic ID-number alteration).
  2. splicing    - a region from a *different* source image is pasted in
                    (e.g. swapping a photo/signature from another document).
  3. text_edit   - a text region is blanked out and re-rendered with
                    different content, simulating DOB / name / ID-number
                    editing.
  4. inpaint_removal - a region (e.g. a stamp/watermark) is removed via
                    OpenCV inpainting, simulating stamp/hologram removal.

Each sample is written as:
  data/synthetic/images/<id>.png
  data/synthetic/labels/<id>.json   { label, forgery_type, bbox, instruction, target_text }

The labels/*.json "instruction" + "target_text" fields are exactly what
src/dataset.py turns into VLM chat-style training examples.
"""

import os
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def load_or_make_source_images(raw_dir: Path, n: int, size=(640, 400)):
    """
    If the user has dropped real (non-sensitive, license-cleared) template
    document images into data/raw/, use those. Otherwise, fall back to
    procedurally-generated mock "ID card" images so the pipeline is runnable
    end-to-end with zero external data (useful for a first smoke test / demo).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpg"))
    if len(existing) >= n:
        return existing[:n]

    print(f"[data_synthesis] Found {len(existing)} real images in {raw_dir}, "
          f"generating {n - len(existing)} mock ID templates to fill the gap. "
          f"Replace these with real (license-cleared) samples for a real run, "
          f"e.g. MIDV-500/MIDV-2020 (https://ftimage.ru/en/midv) or your own "
          f"scanned test documents.")

    made = []
    for i in range(len(existing), n):
        img = Image.new("RGB", size, color=(235, 235, 228))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, size[0] - 10, size[1] - 10], outline=(30, 30, 30), width=3)
        fields = [
            f"NAME: {random.choice(['A SHARMA', 'R KUMAR', 'S PATEL', 'M IYER'])}",
            f"DOB: {random.randint(1,28):02d}/{random.randint(1,12):02d}/{random.randint(1970,2005)}",
            f"ID NO: {random.randint(100000000, 999999999)}",
            f"ISSUED: {random.randint(2015,2024)}",
        ]
        y = 60
        for f in fields:
            draw.text((40, y), f, fill=(20, 20, 20))
            y += 50
        # fake photo box
        draw.rectangle([size[0]-180, 40, size[0]-40, 220], outline=(0,0,0), width=2)
        draw.ellipse([size[0]-150, 70, size[0]-70, 150], fill=(180,150,130))
        # fake signature
        draw.line([(40, size[1]-60), (90, size[1]-90), (140, size[1]-50), (190, size[1]-80)],
                   fill=(10,10,120), width=3)
        path = raw_dir / f"mock_id_{i:04d}.png"
        img.save(path)
        made.append(path)

    return existing + made


def random_bbox(w, h, min_frac=0.10, max_frac=0.30):
    bw = int(w * random.uniform(min_frac, max_frac))
    bh = int(h * random.uniform(min_frac, max_frac))
    x = random.randint(0, max(1, w - bw))
    y = random.randint(0, max(1, h - bh))
    return x, y, bw, bh


# --------------------------------------------------------------------------
# Forgery operators
# --------------------------------------------------------------------------

def forge_copy_move(img: np.ndarray):
    h, w = img.shape[:2]
    x, y, bw, bh = random_bbox(w, h)
    patch = img[y:y+bh, x:x+bw].copy()
    # paste at a shifted location
    nx = min(w - bw, max(0, x + random.choice([-1, 1]) * random.randint(bw, bw*2)))
    ny = min(h - bh, max(0, y + random.randint(-bh, bh)))
    out = img.copy()
    out[ny:ny+bh, nx:nx+bw] = patch
    bbox = [nx, ny, nx+bw, ny+bh]
    desc = ("A region of the image was duplicated from elsewhere in the same "
            "document and pasted over the target area (copy-move forgery), "
            "likely altering a numeric or text field.")
    return out, bbox, desc


def forge_splicing(img: np.ndarray, donor: np.ndarray):
    h, w = img.shape[:2]
    dh, dw = donor.shape[:2]
    bw, bh = int(w * random.uniform(0.15, 0.28)), int(h * random.uniform(0.15, 0.28))
    bw, bh = min(bw, dw - 1), min(bh, dh - 1)
    dx, dy = random.randint(0, dw - bw - 1), random.randint(0, dh - bh - 1)
    patch = cv2.resize(donor[dy:dy+bh, dx:dx+bw], (bw, bh))
    x, y = random.randint(0, w - bw), random.randint(0, h - bh)
    out = img.copy()
    # light alpha blend at edges to look "almost clean" -> harder negative
    out[y:y+bh, x:x+bw] = cv2.addWeighted(out[y:y+bh, x:x+bw], 0.08, patch, 0.92, 0)
    bbox = [x, y, x+bw, y+bh]
    desc = ("Content from a different source document was spliced into this "
            "region, most consistent with a photo or signature swap.")
    return out, bbox, desc


def forge_text_edit(img: np.ndarray):
    h, w = img.shape[:2]
    x, y, bw, bh = random_bbox(w, h, 0.08, 0.18)
    out = img.copy()
    cv2.rectangle(out, (x, y), (x+bw, y+bh), (235, 235, 228), -1)  # blank it
    new_text = str(random.randint(10000000, 99999999))
    cv2.putText(out, new_text, (x+2, y+bh-4), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (15, 15, 15), 1, cv2.LINE_AA)
    bbox = [x, y, x+bw, y+bh]
    desc = ("A text field was erased and re-rendered with different content, "
            "consistent with digit/date/name substitution.")
    return out, bbox, desc


def forge_inpaint_removal(img: np.ndarray):
    h, w = img.shape[:2]
    x, y, bw, bh = random_bbox(w, h, 0.08, 0.20)
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (x, y), (x+bw, y+bh), 255, -1)
    out = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    bbox = [x, y, x+bw, y+bh]
    desc = ("A stamp, watermark, or security mark appears to have been "
            "digitally removed and the background reconstructed.")
    return out, bbox, desc


FORGERY_OPS = {
    "copy_move": forge_copy_move,
    "text_edit": forge_text_edit,
    "inpaint_removal": forge_inpaint_removal,
    # splicing handled separately since it needs a donor image
}


# --------------------------------------------------------------------------
# Main dataset builder
# --------------------------------------------------------------------------

def build_dataset(raw_dir: str, out_dir: str, n_authentic: int, n_forged: int, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total_sources_needed = max(n_authentic, n_forged, 8)
    sources = load_or_make_source_images(raw_dir, total_sources_needed)
    imgs = [cv2.imread(str(p)) for p in sources]

    records = []
    sid = 0

    # authentic samples (with light natural augmentation, NOT forgery)
    for i in range(n_authentic):
        base = imgs[i % len(imgs)].copy()
        # benign augmentation only: jpeg-like noise, slight rotation, brightness
        if random.random() < 0.5:
            base = cv2.convertScaleAbs(base, alpha=random.uniform(0.9, 1.1), beta=random.randint(-8, 8))
        fname = f"sample_{sid:05d}.png"
        cv2.imwrite(str(img_out / fname), base)
        records.append({
            "id": fname,
            "label": "authentic",
            "forgery_type": None,
            "bbox": None,
            "instruction": "Examine this identity document image. Is it authentic or has it been tampered with? If tampered, describe what was altered and where.",
            "target_text": "AUTHENTIC. No signs of digital tampering, copy-move duplication, splicing, or text substitution were detected in this document.",
        })
        sid += 1

    # forged samples
    op_names = list(FORGERY_OPS.keys()) + ["splicing"]
    for i in range(n_forged):
        base = imgs[i % len(imgs)].copy()
        op = random.choice(op_names)
        if op == "splicing":
            donor = imgs[(i + 1) % len(imgs)]
            out_img, bbox, desc = forge_splicing(base, donor)
        else:
            out_img, bbox, desc = FORGERY_OPS[op](base)

        fname = f"sample_{sid:05d}.png"
        cv2.imwrite(str(img_out / fname), out_img)
        h, w = out_img.shape[:2]
        bx = [round(bbox[0]/w, 4), round(bbox[1]/h, 4), round(bbox[2]/w, 4), round(bbox[3]/h, 4)]
        records.append({
            "id": fname,
            "label": "tampered",
            "forgery_type": op,
            "bbox": bx,  # normalized [x1,y1,x2,y2]
            "instruction": "Examine this identity document image. Is it authentic or has it been tampered with? If tampered, describe what was altered and where.",
            "target_text": f"TAMPERED. {desc} Approximate location (normalized bbox): {bx}.",
        })
        sid += 1

    random.shuffle(records)
    for r in records:
        with open(lbl_out / (r["id"].replace(".png", ".json")), "w") as f:
            json.dump(r, f, indent=2)

    manifest = out_dir / "manifest.jsonl"
    with open(manifest, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[data_synthesis] Wrote {len(records)} samples "
          f"({n_authentic} authentic / {n_forged} tampered) to {out_dir}")
    print(f"[data_synthesis] Manifest: {manifest}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/synthetic")
    ap.add_argument("--n_authentic", type=int, default=200)
    ap.add_argument("--n_forged", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_dataset(args.raw_dir, args.out_dir, args.n_authentic, args.n_forged, args.seed)
