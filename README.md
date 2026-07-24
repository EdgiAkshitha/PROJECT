# Document Forgery Detection with a Fine-Tuned Vision-Language Model

A research-style project applying LoRA fine-tuning to a small open-weight VLM
(Qwen2-VL-2B-Instruct) to detect **and explain** tampering in identity
documents, with an evaluation harness that measures robustness under
real-world image degradation and simple adversarial countermeasures.

Built to demonstrate the exact skill set in HyperVerge's DL/ML Research
Intern (LLMs & VLMs) role: multimodal fine-tuning (SFT/LoRA), document &
identity-verification-adjacent modeling, synthetic data generation, and
evaluation harnesses against noisy/adversarial inputs — rather than a
generic tutorial-style classifier or an API-wrapper project.

## Why this project (not just another classifier)

Most "forgery detection" portfolio projects are a binary CNN classifier
trained on a clean Kaggle dataset. Two things make this different, and both
map directly to the JD:

1. **VLM reasoning, not just classification.** The model is trained to
   *explain* what was altered and roughly where (e.g. "text field re-rendered,
   consistent with DOB substitution"), which is closer to the "financial
   risk reasoning" and "document intelligence" language in the JD than a
   plain authentic/fake logit.
2. **Robustness is treated as the actual metric.** Clean-validation accuracy
   is close to meaningless for a production fraud system — real uploads are
   JPEG-compressed, rotated, poorly lit, and a motivated fraudster will
   specifically try to hide the tampering seam. `eval_harness.py` measures
   accuracy *degradation* under each of these, plus one adversarial case
   (locally blurring just the tampered region).

## Project structure

```
src/
  data_synthesis.py   # generates labeled synthetic tampered/authentic docs
  dataset.py           # PyTorch Dataset -> VLM chat-format training examples
  train_lora.py        # LoRA/PEFT SFT fine-tuning of Qwen2-VL-2B-Instruct
  eval_harness.py       # robustness benchmark: clean vs noisy/adversarial
  inference_demo.py    # single-image inference for demos
notebooks/
  colab_quickstart.ipynb  # run the whole pipeline on a free Colab GPU
data/
  raw/          # drop real (license-cleared) document images here
  synthetic/    # generated images + JSON labels + manifest.jsonl
eval_reports/   # robustness_report.md / .json output
```

## How it works

**1. Synthetic data generation** (`src/data_synthesis.py`)
Real labeled forged-ID datasets are scarce and sensitive — which is exactly
why this is a genuine research problem, not a solved one. The generator
programmatically manufactures four realistic tampering modes on clean
document images, each a known real-world fraud pattern:

| Forgery type | What it simulates |
|---|---|
| `copy_move` | A field (e.g. a digit) duplicated elsewhere in the same doc |
| `splicing` | Content pasted in from a *different* source document (photo/signature swap) |
| `text_edit` | A field blanked and re-rendered with different text (DOB/name/ID edits) |
| `inpaint_removal` | A stamp/watermark digitally removed via inpainting |

Each sample gets a ground-truth label, forgery type, normalized bounding box,
and a natural-language target description — this becomes the VLM's SFT
target text.

Ships with a zero-dependency mock-ID generator so the pipeline runs
end-to-end with no external data (`data/raw/` is auto-populated if empty).
**For a real submission, swap in real document images** — e.g.
[MIDV-500/MIDV-2020](https://ftimage.ru/en/midv) (public ID-document dataset
built for exactly this kind of research) or your own scanned test documents.

**2. LoRA fine-tuning** (`src/train_lora.py`)
Standard PEFT/LoRA SFT on Qwen2-VL-2B-Instruct, targeting the LLM decoder's
attention/MLP projections while keeping the vision tower frozen — cheaper,
and avoids degrading the pretrained model's existing OCR/vision skills while
teaching it the new forgery-reasoning task. Runs on a single free-tier Colab
T4/L4 GPU.

**3. Robustness evaluation harness** (`src/eval_harness.py`) — the core
deliverable. Reports accuracy per corruption type, degradation vs. clean
baseline, and a breakdown by forgery type so you can see, e.g., "splicing
detection collapses under JPEG compression but copy-move detection doesn't."
Corruptions: JPEG recompression, Gaussian blur/noise, rotation, low
brightness, and an adversarial seam-smoothing attack.

## Results template (fill in after your own training run)

| Corruption | Accuracy | Δ vs clean |
|---|---|---|
| clean | — | — |
| jpeg_q30 | — | — |
| blur_k9 | — | — |
| rotate_15deg | — | — |
| adversarial_seam_smooth | — | — |

*(The harness auto-generates this table as `eval_reports/robustness_report.md`
— replace this template with your real numbers once trained.)*

## Running it

```bash
pip install -r requirements.txt

# 1. Generate synthetic dataset (1000 samples, ~1 min, no GPU needed)
python src/data_synthesis.py --n_authentic 500 --n_forged 500

# 2. LoRA fine-tune (needs a GPU — Colab T4 is enough; see notebooks/colab_quickstart.ipynb)
python src/train_lora.py --epochs 3 --batch_size 2 --lr 2e-4

# 3. Robustness evaluation
python src/eval_harness.py --checkpoint checkpoints/forgery-lora/epoch2

# 4. Single-image demo
python src/inference_demo.py --checkpoint checkpoints/forgery-lora/epoch2 \
    --image data/synthetic/images/sample_00013.png
```

`eval_harness.py` also runs with `--checkpoint` omitted, using a heuristic
edge-density baseline — useful to sanity-check the corruption/reporting
pipeline itself before you have a trained model (this is how the pipeline
was validated while building this repo, without GPU access).

## What I'd extend this into with more time

- Swap synthetic forgeries for a real dataset (MIDV-500/2020) once fine-tuning
  is validated, and add a held-out "hard negative" set of authentic documents
  with heavy natural wear (creases, glare) to reduce false positives.
- Add localization metrics (IoU between predicted and ground-truth bbox, not
  just text-level tampered/authentic accuracy).
- Preference-alignment pass (DPO) using pairs of good vs. vague/incorrect
  explanations, to sharpen the *reasoning* quality, not just the label.
- Quantify inference latency/cost at different LoRA ranks and quantization
  levels — accuracy/latency/cost tradeoffs the JD explicitly calls out.

## Notes on running this yourself

This repo was built and its data/eval pipeline smoke-tested in a sandboxed
environment without GPU or Hugging Face Hub access — `data_synthesis.py` and
the corruption/report logic in `eval_harness.py` were run end-to-end and
verified. `train_lora.py` and the real-model path in `eval_harness.py`
require a GPU and internet access to download the base model; run those via
`notebooks/colab_quickstart.ipynb` on a free Colab GPU.
