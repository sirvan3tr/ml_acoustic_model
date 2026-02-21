# Multitask ResNet + Kendall Weighting (v3)

This version targets the persistent side-class collapse seen in `v2`.

## Changes vs v2

- Input features now include both:
  - base log-mel channels (`8`)
  - relative-channel log-mel (`8`) where each channel is centered by channel mean
- Model input channels are therefore `16` by default.
- Kendall side task uses a **weight floor** so side cannot be down-weighted below a minimum.

## Why

- Side/orientation depends on cross-channel directional cues.
- Relative-channel features make those cues explicit.
- Side-weight floor prevents uncertainty weighting from suppressing side learning in early epochs.

## Train

```bash
uv run python models/multitask_resnet_kendall_v3/train.py \
  --data-root mic_array_data \
  --output-dir models/multitask_resnet_kendall_v3/runs/exp01
```

Optional knobs:

- `--side-weight-floor 1.0` (default)
- `--window-seconds 3.0 --hop-seconds 1.5` for longer context
- `--no-relative-channels --model-in-channels 8` to disable the v3 feature extension

## Outputs

- `best.pt`, `last.pt`
- `metrics.csv`
- `side_confusion_epoch_XXX.csv`
- `config.json`
