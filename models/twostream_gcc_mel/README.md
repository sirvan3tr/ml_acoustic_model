# Two-Stream GCC-PHAT + Mel Model

Physics-informed multitask model with separate feature streams:

- **Mel stream** for spectral/range cues (distance, height)
- **GCC-PHAT stream** for inter-microphone delay cues (azimuth, side)

## Targets

- `distance` (m): regression (L1)
- `height` (m): regression (L1)
- `azimuth` (deg): circular regression via predicted `[sin, cos]`
- `azimuth class` (8 bins): auxiliary classification head for angle stabilization
- `side` (`Front/Right/Back/Left`): 4-class cross-entropy

## Architecture

```text
Waveform [B,8,T]
  -> Feature builder
     - mel: [B,8,256,256]
     - gcc-phat mean/std: [B,56,2*max_lag+1] (default)
     - channel energy: [B,8]
  -> Mel 2D CNN stream
  -> GCC 1D CNN stream
  -> Energy MLP
  -> Shared fusion -> range tower + direction tower
  -> 5 task heads (distance/height from range, azimuth_xy + azimuth_class + side from direction)
```

## Loss Weighting

Uses fixed robust weighting:

- `L_distance`: L1
- `L_height`: L1
- `L_azimuth`: vector cosine loss on `[sin, cos]`
- `L_azimuth_cls`: CE on 8 azimuth bins
- `L_side`: focal CE + KL(U || mean batch probs)
- `L_total = w_d * L_d + w_h * L_h + w_a * L_a + w_ac * L_ac + w_s * L_s`

## Augmentations

Training-time waveform augmentations:

- microphone dropout
- ambient noise mixing from dataset ambient recordings
- gain jitter (global + light per-channel)
- optional circular channel-ring shifts (`circular_shift_prob`, off by default)

## Train

```bash
uv run python models/twostream_gcc_mel/train.py \
  --data-root mic_array_data \
  --output-dir models/twostream_gcc_mel/runs/exp01 \
  --window-seconds 3.0 \
  --hop-seconds 1.5 \
  --learning-rate 1e-4
```

Quick smoke run:

```bash
uv run python models/twostream_gcc_mel/train.py \
  --epochs 1 \
  --max-train-steps 2 \
  --max-val-steps 1 \
  --num-workers 0 \
  --disable-augment
```

Useful knobs:

- `--w-side 4.0 --w-side-kl 0.5 --side-focal-gamma 1.5`
- `--w-azimuth 2.0 --w-azimuth-cls 1.0`
- `--no-gcc-std` to disable GCC temporal std channels
- `--global-gain-db-max` and `--per-channel-gain-db-max` to tune gain jitter

## Artifacts

- `config.json`
- `metrics.csv`
- `side_confusion_epoch_XXX.csv`
- `best.pt`
- `last.pt`
