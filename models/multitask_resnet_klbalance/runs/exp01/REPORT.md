# Experiment Report: `multitask_resnet_klbalance` (`exp01`)

## Run Command

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

## Duration

- Epochs: `30`

## Model Summary (Presentation-Ready)

### Problem framing

This is a **multitask supervised learning** problem with mixed geometry:

- Distance (`m`): regression
- Height (`m`): regression
- Azimuth (`deg`): circular regression
- Side (`Front/Right/Back/Left`): 4-class classification

The design goal was to avoid the known failure mode where one task dominates optimization and side/azimuth collapse.

### Architecture

- Backbone: small ResNet-style CNN on spectrogram input.
- Input channels: `16` (`8` log-mel + `8` relative-channel log-mel).
- Heads:
  - distance scalar
  - height scalar
  - azimuth vector (`sin(theta), cos(theta)`)
  - side logits (4 classes)

### Why these design decisions

- **Relative-channel features (`8 -> 16`)**:
  - Direction is encoded by channel-to-channel differences; per-channel features alone were too weak for side.
- **Azimuth as `sin/cos`**:
  - Removes wrap-around discontinuity (e.g., `359 deg` vs `1 deg`).
  - Gives smooth gradients for circular targets.
- **Fixed task weights instead of Kendall**:
  - Earlier Kendall variants tended to down-weight hard tasks too early.
  - Fixed weighting gave more predictable optimization.
- **Side KL regularizer**:
  - Added `KL(U || mean_batch_probs)` to discourage collapse into one side class.
  - Combined with side CE as:
    - `L_side = CE + w_side_kl * KL`

### Training objective

Per-task losses:

- `L_distance = L1`
- `L_height = L1`
- `L_azimuth = 1 - cos_sim(pred_xy, target_xy)`
- `L_side = CE(label_smoothing) + w_side_kl * KL(U || mean_batch_probs)`

Total:

- `L_total = w_distance * L_distance + w_height * L_height + w_azimuth * L_azimuth + w_side * L_side`

For this run:

- `w_distance=1.0`
- `w_height=1.0`
- `w_azimuth=1.0`
- `w_side=2.0`
- `w_side_kl=0.5`

## Key Results

- Best validation total loss: `2.2001` at epoch `27`
- Best side accuracy: `0.6484` at epoch `26`
- Best azimuth circular MAE: `5.5462 deg` at epoch `26`
- Best distance MAE: `0.1122 m` at epoch `27`
- Best height MAE: `0.0585 m` at epoch `29`

## End-of-Run Snapshot (Epoch 30)

- `val_total`: `2.2087`
- `val_dist_mae`: `0.1203 m`
- `val_height_mae`: `0.1205 m`
- `val_az_mae`: `7.0144 deg`
- `val_side_acc`: `0.6109`
- `val_side_ce`: `0.9560`
- `val_side_kl`: `0.0450`

## Late-Stage Stability (Mean of Epochs 26-30)

- `val_total`: `2.2065`
- `val_dist_mae`: `0.1202 m`
- `val_height_mae`: `0.0947 m`
- `val_az_mae`: `6.2799 deg`
- `val_side_acc`: `0.6178`
- `val_side_ce`: `0.9696`
- `val_side_kl`: `0.0429`

## Interpretation

- Distance and height converged to low error and remained stable late in training.
- Azimuth improved from early instability to strong angular performance (~`5.5-7.0 deg`).
- Side classification moved from near-random (`~0.25`) to materially useful (`~0.61-0.65`), with a clear jump after epoch ~18.
- `best.pt` is the strongest overall checkpoint (`epoch 27`) by `val_total`.

## Comparison vs Paper Benchmark

Paper primary benchmark (44.1 kHz, FFT=1024, hop=256, Mel=256):

- Distance MAE: `0.36 m`
- Height MAE: `0.30 m`
- Azimuth MAE: `0.94 deg`
- Side MAE: `9.50 deg`

This run (`exp01`) best observed values:

- Distance MAE: `0.112 m` (better than paper benchmark)
- Height MAE: `0.059 m` (better than paper benchmark)
- Azimuth MAE: `5.546 deg` (worse than paper benchmark)
- Side accuracy: `0.648` (metric differs from paper side MAE)
- Side MAE estimate from confusion matrix (best-side epoch 26): `~40.08 deg` (worse than paper side MAE)

Note: side MAE estimate assumes class angle mapping Front/Right/Back/Left = `0/90/180/270 deg`.

## Saved Artifacts

- `best.pt` (epoch `27`)
- `last.pt` (epoch `30`)
- `metrics.csv`
- `side_confusion_epoch_001.csv` ... `side_confusion_epoch_030.csv`
- `config.json`
