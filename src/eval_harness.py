"""
eval_harness.py

"Design evaluation benchmarks beyond clean validation datasets. Build
evaluation harnesses against noisy, adversarial, and real-world inputs."
(direct quote from the JD -- this file is the answer to that bullet.)

A model that scores 95% on a clean validation split is not interesting for
a production fraud system: real uploads arrive JPEG-recompressed, at an
angle, poorly lit, or slightly blurred from a phone camera -- and a
motivated fraudster will deliberately degrade the image to hide tampering
evidence. This harness measures accuracy DEGRADATION under each corruption,
not just clean accuracy, which is the metric that actually matters for
identity verification in production.

Corruption families implemented:
  - JPEG recompression (quality 30 / 15)   -> real-world upload artifacts
  - Gaussian blur                          -> phone camera / low-res scan
  - Gaussian noise                         -> low-light sensor noise
  - Rotation (+/- 5 deg, +/- 15 deg)        -> off-axis phone capture
  - Brightness/contrast shift              -> poor lighting
  - Adversarial: light Gaussian blur specifically over the tampered bbox
    ("smoothing the seam") -> a plausible fraudster countermeasure

Usage:
    python src/eval_harness.py --checkpoint checkpoints/forgery-lora/epoch2 \
        --manifest data/synthetic/manifest.jsonl --images_dir data/synthetic/images

Outputs a markdown + JSON report to eval_reports/.
"""

import argparse
import json
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# Corruption functions: each takes (np.ndarray BGR image, record) -> np.ndarray
# --------------------------------------------------------------------------

def corrupt_jpeg(img, record, quality=30):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def corrupt_blur(img, record, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def corrupt_noise(img, record, sigma=15):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def corrupt_rotate(img, record, angle=10):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(235, 235, 228))


def corrupt_brightness(img, record, delta=-40):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=delta)


def corrupt_seam_smooth(img, record, ksize=9):
    """Adversarial: blur *only* the tampered region to hide the forgery seam."""
    out = img.copy()
    if record.get("bbox") is None:
        return corrupt_blur(img, record, ksize=3)  # authentic sample: mild global blur as control
    h, w = img.shape[:2]
    x1, y1, x2, y2 = record["bbox"]
    x1, x2 = int(x1 * w), int(x2 * w)
    y1, y2 = int(y1 * h), int(y2 * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        region = out[y1:y2, x1:x2]
        out[y1:y2, x1:x2] = cv2.GaussianBlur(region, (ksize, ksize), 0)
    return out


CORRUPTIONS = {
    "clean":               lambda img, r: img,
    "jpeg_q30":            lambda img, r: corrupt_jpeg(img, r, 30),
    "jpeg_q15":            lambda img, r: corrupt_jpeg(img, r, 15),
    "blur_k5":             lambda img, r: corrupt_blur(img, r, 5),
    "blur_k9":             lambda img, r: corrupt_blur(img, r, 9),
    "gaussian_noise_s15":  lambda img, r: corrupt_noise(img, r, 15),
    "gaussian_noise_s30":  lambda img, r: corrupt_noise(img, r, 30),
    "rotate_5deg":         lambda img, r: corrupt_rotate(img, r, 5),
    "rotate_15deg":        lambda img, r: corrupt_rotate(img, r, 15),
    "low_brightness":      lambda img, r: corrupt_brightness(img, r, -40),
    "adversarial_seam_smooth": lambda img, r: corrupt_seam_smooth(img, r, 9),
}


# --------------------------------------------------------------------------
# Model interface (swap this stub for your real fine-tuned VLM inference)
# --------------------------------------------------------------------------

class ForgeryVLMPredictor:
    """
    Thin wrapper so eval_harness.py can be used two ways:
      1. Real run: pass --checkpoint, this loads your LoRA-tuned VLM and
         calls processor/model.generate() for each image.
      2. Dry run / CI smoke test (what runs inside this sandbox): falls back
         to a trivial heuristic baseline so the *harness plumbing* itself
         (corruptions -> metrics -> report) can be validated without a GPU.
         Swap USE_REAL_MODEL logic below once you have a trained checkpoint.
    """
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path
        self.real_model_loaded = False
        if checkpoint_path:
            try:
                import torch
                from transformers import AutoProcessor, AutoModelForVision2Seq
                self.processor = AutoProcessor.from_pretrained(checkpoint_path)
                self.model = AutoModelForVision2Seq.from_pretrained(checkpoint_path)
                self.model.eval()
                self.real_model_loaded = True
            except Exception as e:
                print(f"[eval_harness] Could not load real model ({e}); "
                      f"falling back to heuristic baseline for harness validation.")

    def predict(self, pil_image, instruction):
        if self.real_model_loaded:
            import torch
            messages = [{"role": "user", "content": instruction}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt], images=[pil_image], return_tensors="pt")
            with torch.no_grad():
                out_ids = self.model.generate(**inputs, max_new_tokens=128)
            text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            return text

        # ---- Heuristic fallback baseline (edge-density variance) ----
        # NOT the real model. Only exists so the harness code path is fully
        # exercised and testable without GPU access. Replace by always
        # passing --checkpoint once you have a trained adapter.
        arr = np.array(pil_image.convert("L"))
        edges = cv2.Canny(arr, 80, 160)
        score = edges.mean()
        label = "TAMPERED" if score > 8.5 else "AUTHENTIC"
        return f"{label}. (heuristic baseline prediction, edge_density={score:.2f})"


def parse_prediction(text):
    return "tampered" if text.strip().upper().startswith("TAMPERED") else "authentic"


# --------------------------------------------------------------------------
# Main eval loop
# --------------------------------------------------------------------------

def run_eval(manifest_path, images_dir, checkpoint, out_dir, max_samples=None):
    images_dir = Path(images_dir)
    records = [json.loads(l) for l in open(manifest_path)]
    if max_samples:
        records = records[:max_samples]

    predictor = ForgeryVLMPredictor(checkpoint)
    results = {name: {"correct": 0, "total": 0} for name in CORRUPTIONS}
    per_type_results = {}  # forgery_type -> {corruption -> correct/total}

    for r in records:
        img_path = images_dir / r["id"]
        img_bgr = cv2.imread(str(img_path))
        gt = r["label"]
        ftype = r["forgery_type"] or "authentic"
        per_type_results.setdefault(ftype, {name: {"correct": 0, "total": 0} for name in CORRUPTIONS})

        for cname, cfn in CORRUPTIONS.items():
            corrupted = cfn(img_bgr, r)
            pil_img = Image.fromarray(cv2.cvtColor(corrupted, cv2.COLOR_BGR2RGB))
            pred_text = predictor.predict(pil_img, r["instruction"])
            pred_label = parse_prediction(pred_text)

            correct = int(pred_label == gt)
            results[cname]["correct"] += correct
            results[cname]["total"] += 1
            per_type_results[ftype][cname]["correct"] += correct
            per_type_results[ftype][cname]["total"] += 1

    # ---- build report ----
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {name: round(100 * v["correct"] / max(v["total"], 1), 1) for name, v in results.items()}
    clean_acc = summary.get("clean", 0)

    md = ["# Forgery-Detection Robustness Report\n"]
    md.append(f"Model source: `{checkpoint or 'heuristic-baseline (dry run)'}`\n")
    md.append(f"Samples evaluated: {len(records)}\n")
    md.append("\n## Accuracy by corruption (vs. clean baseline)\n")
    md.append("| Corruption | Accuracy | Δ vs clean |")
    md.append("|---|---|---|")
    for name, acc in summary.items():
        delta = round(acc - clean_acc, 1)
        flag = " ⚠️" if delta <= -10 and name != "clean" else ""
        md.append(f"| {name} | {acc}% | {delta:+.1f}{flag} |")

    md.append("\n## Accuracy by forgery type (clean vs. hardest corruption)\n")
    md.append("| Forgery type | Clean acc | Worst corruption | Worst acc |")
    md.append("|---|---|---|---|")
    for ftype, corr_res in per_type_results.items():
        clean = 100 * corr_res["clean"]["correct"] / max(corr_res["clean"]["total"], 1)
        worst_name, worst_acc = None, 101
        for cname, v in corr_res.items():
            if cname == "clean":
                continue
            acc = 100 * v["correct"] / max(v["total"], 1)
            if acc < worst_acc:
                worst_acc, worst_name = acc, cname
        md.append(f"| {ftype} | {clean:.1f}% | {worst_name} | {worst_acc:.1f}% |")

    report_md = "\n".join(md)
    (out_dir / "robustness_report.md").write_text(report_md)
    with open(out_dir / "robustness_report.json", "w") as f:
        json.dump({"summary": summary, "per_type": per_type_results}, f, indent=2)

    print(report_md)
    print(f"\n[eval_harness] Report written to {out_dir}/robustness_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="Path to fine-tuned model/processor dir. Omit for heuristic dry-run.")
    ap.add_argument("--manifest", default="data/synthetic/manifest.jsonl")
    ap.add_argument("--images_dir", default="data/synthetic/images")
    ap.add_argument("--out_dir", default="eval_reports")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()
    run_eval(args.manifest, args.images_dir, args.checkpoint, args.out_dir, args.max_samples)
