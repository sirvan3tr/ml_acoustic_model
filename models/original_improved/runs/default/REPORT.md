# Experiment Report: `original_improved` (`default`)

## Run Command

```bash
uv run models/original_improved/train.py --device cuda:1
```

## Duration

- Epochs: `30`

## Model Summary (Presentation-Ready)

### Problem framing

Multitask learning with mixed geometry:

- Distance (`m`) regression
- Height (`m`) regression
- Azimuth circular regression
- Side 4-class prediction represented as circular `sin/cos`

### Architecture

- Input feature stack: `16` channels
- Feature composition:
  - `8` log-mel channels
  - `8` IPD channels (mel-phase relative to array-mean phase)
- Backbone: 8-stage ResNet-style CNN (`32 -> 64 -> 128 -> 192 -> 256 -> 512 -> 512 -> 512`)
- Early downsampling delay: `conv2_x` keeps stride `1` to preserve spatial/phase detail
- Head: `Linear(512->256) + Dropout(0.5) + Linear(256->6)`
- Output vector: `[dist_norm, height_norm, sin(az), cos(az), sin(side), cos(side)]`

### Loss design

- Base per-target objective: independent MSE across 6 outputs
- Task balancing: Kendall homoscedastic weighting (`exp(-s_i) * L_i + s_i`)
- Target normalization:
  - distance and height are scaled by `/20` before loss
  - improves scale parity against angular `sin/cos` targets
- Optimizer/schedule:
  - `AdamW` (`lr=3e-4`, `weight_decay=1e-4`)
  - `CosineAnnealingLR`

## Key Results

- Best validation total loss: `-4.2418` at epoch `30`
- Best distance MAE: `0.713 m` at epoch `28`
- Best height MAE: `0.770 m` at epoch `30`
- Best azimuth circular MAE: `2.489 deg` at epoch `28`
- Best side accuracy: `0.8184` at epoch `27`
- Best side MAE estimate: `18.11 deg` at epoch `27`

## End-of-Run Snapshot (Epoch 30)

- `val_total`: `-4.2418`
- `val_dist_mae`: `0.764 m`
- `val_height_mae`: `0.770 m`
- `val_az_mae`: `2.603 deg`
- `val_side_acc`: `0.8105`
- `s_dist`: `-0.802`
- `s_height`: `-0.802`
- `s_sin_az`: `-0.839`
- `val_side_mae_est`: `18.63 deg`

## Late-Stage Stability (Mean of Epochs 26-30)

- `val_total`: `-4.1755`
- `val_dist_mae`: `0.7478 m`
- `val_height_mae`: `0.8712 m`
- `val_az_mae`: `2.6530 deg`
- `val_side_acc`: `0.8126`
- `s_dist`: `-0.791`
- `s_height`: `-0.791`
- `s_sin_az`: `-0.828`
- `val_side_mae_est`: `18.45 deg`

## Training Dynamics

- Fast early convergence:
  - epoch 1: distance/height MAE near `5 m`, azimuth MAE `45.1 deg`
  - epoch 9: distance MAE `1.703 m`, azimuth MAE `3.818 deg`, side acc `0.773`
- Strong final-phase refinement:
  - distance MAE improved to `0.713 m` (epoch 28)
  - azimuth MAE improved to `2.489 deg` (epoch 28)
  - side accuracy stabilized around `~0.81`

## Side Performance Detail

- Epoch 27 side confusion (`Front/Right/Back/Left`) shows strong diagonal concentration:
  - Front: `317/384`
  - Right: `307/384`
  - Back: `328/384`
  - Left: `305/384`
- Side remains stable through final epochs (epoch 30 side accuracy `0.8105`).

## Comparison vs Paper Benchmark

Paper primary benchmark (44.1 kHz, FFT=1024, hop=256, Mel=256):

- Distance MAE: `0.36 m`
- Height MAE: `0.30 m`
- Azimuth MAE: `0.94 deg`
- Side MAE: `9.50 deg`

This run (`original_improved`) best observed values:

- Distance MAE: `0.713 m` (worse)
- Height MAE: `0.770 m` (worse)
- Azimuth MAE: `2.489 deg` (worse)
- Side accuracy: `0.818` (metric differs from paper side MAE)
- Side MAE estimate: `18.11 deg` (worse than paper side MAE)

## Comparison vs `original` Baseline (`models/original/runs/default`)

- Side accuracy improved from `0.266` best (`original`) to `0.818` best (`original_improved`).
- Side MAE estimate improved from `~88.30 deg` best (`original`) to `~18.11 deg` best (`original_improved`).
- This is a substantial reduction in side-collapse relative to the original paper-style baseline implementation.

## Saved Artifacts

- `best.pt` (epoch `30`)
- `config.json`
- `side_confusion_epoch_001.csv` ... `side_confusion_epoch_030.csv`

## Reproducibility Note

- This report uses persisted artifacts plus captured console metrics from the training run.
- For future runs, adding `metrics.csv` and `last.pt` in training output would simplify report generation and cross-run comparisons.
