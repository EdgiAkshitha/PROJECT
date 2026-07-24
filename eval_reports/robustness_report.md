# Forgery-Detection Robustness Report

Model source: `heuristic-baseline (dry run)`

Samples evaluated: 40


## Accuracy by corruption (vs. clean baseline)

| Corruption | Accuracy | Δ vs clean |
|---|---|---|
| clean | 45.0% | +0.0 |
| jpeg_q30 | 45.0% | +0.0 |
| jpeg_q15 | 42.5% | -2.5 |
| blur_k5 | 45.0% | +0.0 |
| blur_k9 | 45.0% | +0.0 |
| gaussian_noise_s15 | 50.0% | +5.0 |
| gaussian_noise_s30 | 55.0% | +10.0 |
| rotate_5deg | 45.0% | +0.0 |
| rotate_15deg | 45.0% | +0.0 |
| low_brightness | 45.0% | +0.0 |
| adversarial_seam_smooth | 45.0% | +0.0 |

## Accuracy by forgery type (clean vs. hardest corruption)

| Forgery type | Clean acc | Worst corruption | Worst acc |
|---|---|---|---|
| text_edit | 0.0% | jpeg_q30 | 0.0% |
| copy_move | 0.0% | jpeg_q30 | 0.0% |
| splicing | 0.0% | jpeg_q30 | 0.0% |
| authentic | 100.0% | gaussian_noise_s30 | 0.0% |
| inpaint_removal | 0.0% | jpeg_q30 | 0.0% |