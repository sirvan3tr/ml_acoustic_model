# Experiment Report: `twostream_gcc_mel` (`exp01`)

## Run Command

```bash
uv run python models/twostream_gcc_mel/train.py \
  --data-root mic_array_data \
  --output-dir models/twostream_gcc_mel/runs/exp01 \
  --window-seconds 3.0 \
  --hop-seconds 1.5 \
  --learning-rate 1e-4
```

## Duration

- Epochs: `30`

## Model Summary (Presentation-Ready)

### Problem framing

Multitask learning with mixed output geometry:

- Distance (`m`) regression
- Height (`m`) regression
- Azimuth circular regression
- Side 4-class classification

### Architecture

- Two-stream input representation:
  - Mel spectrogram stream (spectral/range cues)
  - GCC-PHAT stream (inter-mic delay cues for direction)
- Extra channel-energy stream (per-channel RMS summary)
- Shared fusion with task heads:
  - Distance head
  - Height head
  - Azimuth vector head (`sin(theta), cos(theta)`)
  - Auxiliary azimuth classification head (8 bins)
  - Side classification head (+ energy-prior blend)

### Loss design

- `L_distance = L1`
- `L_height = L1`
- `L_azimuth = 1 - cos_sim(pred_xy, target_xy)`
- `L_azimuth_cls = CE` (8 bins)
- `L_side = focal_CE + w_side_kl * KL(U || mean_batch_probs)`
- `L_total = w_d*L_d + w_h*L_h + w_a*L_a + w_ac*L_ac + w_s*L_s`

For this run:

- `w_distance=1.0`
- `w_height=1.0`
- `w_azimuth=2.0`
- `w_azimuth_cls=1.0`
- `w_side=4.0`
- `w_side_kl=0.5`

## Key Results

- Best validation total loss: `7.0512` at epoch `30`
- Best side accuracy: `0.2750` at epoch `8`
- Best azimuth circular MAE: `2.9827 deg` at epoch `30`
- Best azimuth class accuracy: `1.0000` at epoch `24`
- Best distance MAE: `1.0561 m` at epoch `30`
- Best height MAE: `1.9716 m` at epoch `30`

## End-of-Run Snapshot (Epoch 30)

- `val_total`: `7.0512`
- `val_dist_mae`: `1.0561 m`
- `val_height_mae`: `1.9716 m`
- `val_az_mae`: `2.9827 deg`
- `val_az_cls_acc`: `0.9953`
- `val_side_acc`: `0.2516`
- `val_side_focal`: `0.8964`
- `val_side_kl`: `0.0004`
- `side_prior_mix`: `0.2738`

## Late-Stage Stability (Mean of Epochs 26-30)

- `val_total`: `7.1409`
- `val_dist_mae`: `1.0928 m`
- `val_height_mae`: `2.0197 m`
- `val_az_mae`: `3.1900 deg`
- `val_az_cls_acc`: `0.9966`
- `val_side_acc`: `0.2531`
- `val_side_focal`: `0.8963`
- `val_side_kl`: `0.0004`
- `side_prior_mix`: `0.2736`

## Interpretation

- Azimuth learning is strong in this setup (low angular MAE and near-saturated azimuth class accuracy).
- Distance/height underperform compared with simpler Mel-only models in this repo.
- Side remains near-random for 4 classes (`~0.25`), indicating persistent side-collapse.
- Side losses changed very little late in training (`val_side_focal` and `val_side_kl` nearly flat), consistent with weak side learning.

## Comparison vs Paper Benchmark

Paper primary benchmark (44.1 kHz, FFT=1024, hop=256, Mel=256):

- Distance MAE: `0.36 m`
- Height MAE: `0.30 m`
- Azimuth MAE: `0.94 deg`
- Side MAE: `9.50 deg`

This run (`exp01`) best observed values:

- Distance MAE: `1.056 m` (worse)
- Height MAE: `1.972 m` (worse)
- Azimuth MAE: `2.983 deg` (worse)
- Side accuracy: `0.275` (metric differs from paper side MAE)
- Side MAE estimate from confusion matrix:
  - best-side epoch 8: `~86.34 deg`
  - best-total/last epoch 30: `~90.70 deg`

Note: side MAE estimate assumes class-angle mapping Front/Right/Back/Left = `0/90/180/270 deg`.

## Comparison vs `multitask_resnet_klbalance` (`exp01`)

- `twostream_gcc_mel` is better on azimuth MAE (`~2.98 deg` vs `~5.55 deg` best).
- `twostream_gcc_mel` is substantially worse on distance/height and side.
- For balanced overall localization quality on this dataset, `multitask_resnet_klbalance` remains the stronger model.

## Saved Artifacts

- `best.pt` (epoch `30`)
- `last.pt` (epoch `30`)
- `metrics.csv`
- `side_confusion_epoch_001.csv` ... `side_confusion_epoch_030.csv`
- `config.json`

