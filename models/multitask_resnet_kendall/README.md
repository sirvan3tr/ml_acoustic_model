# Multitask ResNet + Kendall Weighting

First baseline model for this repo.

## Objective

Predict four targets jointly from 8-channel mel spectrogram inputs:

- Distance (`m`) regression
- Height (`m`) regression
- Azimuth (`deg`) regression with circular-aware loss
- Side (`Front/Right/Back/Left`) 4-class classification

## Architecture

- Input: `8 x 256 x 256` log-mel tensor
- Backbone: small ResNet with 4 residual stages (`32 -> 64 -> 128 -> 128`)
- Heads:
  - `distance`: scalar
  - `height`: scalar
  - `azimuth_deg`: scalar (degrees)
  - `side_logits`: 4 logits

Implementation: `models/multitask_resnet_kendall/model.py`

## Losses

Per-task losses:

- Distance: `L1` (MAE)
- Height: `L1` (MAE)
- Azimuth: `1 - cos(pred_rad - target_rad)` (circular)
- Side: `CrossEntropy`

Task balancing:

- Learned Kendall uncertainty weighting with trainable `s_i = log(sigma_i^2)`
- Total loss:
  - `sum(exp(-s_i) * L_i + s_i)`

Implementation: `models/multitask_resnet_kendall/losses.py`

## Features

Waveform `[C, T]` is converted to normalized log-mel:

- `n_fft=1024`
- `hop_length=512`
- `n_mels=256`
- resized to `256x256`

Implementation: `models/multitask_resnet_kendall/features.py`

## Data

Uses shared dataset:

- `acoustic_ml.data.AcousticUAVDataset`
- Source filter: `Drone`
- Per-file temporal split: `75% train / 25% test`

## Train

Default run:

```bash
uv run python models/multitask_resnet_kendall/train.py \
  --data-root mic_array_data \
  --output-dir models/multitask_resnet_kendall/runs/default
```

Quick smoke run:

```bash
uv run python models/multitask_resnet_kendall/train.py \
  --epochs 1 \
  --max-train-steps 5 \
  --max-val-steps 2 \
  --num-workers 0
```

## Artifacts

Training writes:

- `config.json`
- `metrics.csv` (epoch-level train/val metrics + learned `s_i`)
- `best.pt`
- `last.pt`

to the selected run directory.

