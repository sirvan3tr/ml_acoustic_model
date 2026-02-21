# Multitask ResNet + KL Side Balance

Alternative to Kendall weighting.

## Core idea

- Use fixed task weights instead of learned uncertainty weights.
- Keep azimuth as `sin/cos` vector loss.
- Add KL-divergence regularizer that pushes batch-mean side probabilities toward uniform to reduce class-collapse.

## Loss

- `L_distance = L1`
- `L_height = L1`
- `L_azimuth = 1 - cos_sim(pred_xy, target_xy)`
- `L_side = CE(label_smoothing) + w_side_kl * KL(U || mean_batch_probs)`
- `L_total = w_distance * L_distance + w_height * L_height + w_azimuth * L_azimuth + w_side * L_side`

## Features

- Global-normalized log-mel.
- Relative-channel mel features appended by default (`8 -> 16` channels).

## Train

```bash
uv run python models/multitask_resnet_klbalance/train.py \
  --data-root mic_array_data \
  --output-dir models/multitask_resnet_klbalance/runs/exp01 \
  --window-seconds 3.0 \
  --hop-seconds 1.5 \
  --learning-rate 1e-4 \
  --w-side 2.0 \
  --w-side-kl 0.5
```

Optional knobs:

- `--w-side 3.0` (stronger side emphasis)
- `--w-side-kl 0.8` (stronger anti-collapse regularization)
- `--side-label-smoothing 0.05`
- `--no-relative-channels --model-in-channels 8`

## Outputs

- `best.pt`, `last.pt`
- `metrics.csv`
- `side_confusion_epoch_XXX.csv`
- `config.json`
