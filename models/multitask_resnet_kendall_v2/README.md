# Multitask ResNet + Kendall Weighting (v2)

Second iteration of the first model, focused on fixing azimuth/side collapse observed in v1.

## What Changed vs v1

- Azimuth head now predicts `sin/cos` (`2` outputs) instead of raw degrees.
- Azimuth loss uses cosine similarity on unit vectors.
- Validation converts predicted `sin/cos` to degrees with `atan2` for circular MAE reporting.
- Mel normalization is now global per sample (across all channels), not per-channel, preserving directional level cues.
- Side confusion matrix is written every epoch.

## Model

- Input: `8 x 256 x 256` log-mel.
- Backbone: small ResNet stages `32 -> 64 -> 128 -> 128`.
- Heads:
  - `distance`: scalar regression
  - `height`: scalar regression
  - `azimuth_xy`: 2D vector `[sin, cos]`
  - `side_logits`: 4-class logits

## Losses

- Distance: `L1`
- Height: `L1`
- Azimuth: `1 - cos_sim(normalize(pred_xy), normalize(target_xy))`
- Side: `CrossEntropy`
- Total: Kendall weighting `sum(exp(-s_i) * L_i + s_i)`

## Run

```bash
uv run python models/multitask_resnet_kendall_v2/train.py \
  --data-root mic_array_data \
  --output-dir models/multitask_resnet_kendall_v2/runs/exp01
```

Quick smoke run:

```bash
uv run python models/multitask_resnet_kendall_v2/train.py \
  --epochs 1 \
  --max-train-steps 5 \
  --max-val-steps 2 \
  --num-workers 0
```

## Outputs

- `best.pt`, `last.pt`
- `metrics.csv`
- `side_confusion_epoch_XXX.csv`
- `config.json`

