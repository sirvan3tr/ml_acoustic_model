# Experiment Report: `multitask_resnet_kendall_v4` (`default`)

## Run Command

```bash
uv run models/multitask_resnet_kendall_v4/train.py --device cuda:1
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

- Input feature stack: `24` channels
- Feature composition:
  - `8` log-mel channels
  - `8` relative mel channels (channel-minus-array-mean)
  - `8` phase IPD channels (mel-phase relative to array mean phase)
- Backbone: small ResNet with residual stages `32 -> 64 -> 128 -> 256`
- Stem stride set to `1` to preserve early time-frequency resolution
- Four task heads from shared embedding:
  - distance scalar
  - height scalar
  - azimuth `[sin, cos]`
  - side logits (`Front/Right/Back/Left`)

### Loss design

Per-task losses:

- `L_distance = L1`
- `L_height = L1`
- `L_azimuth = 1 - cos_sim(pred_xy, target_xy)`
- `L_side = cross_entropy`

Combined with Kendall homoscedastic weighting:

- `L_total = sum(exp(-s_i) * L_i + s_i)`
- Learned `s_i`: distance, height, azimuth, side
- `side_weight_floor=1.0` used to prevent side weight collapse

## Key Results

- Best validation total loss: `-2.1084` at epoch `30`
- Best distance MAE: `0.0290 m` at epoch `26`
- Best height MAE: `0.0421 m` at epoch `30`
- Best azimuth circular MAE: `3.6836 deg` at epoch `30`
- Best side accuracy: `0.6204` at epoch `27`

## End-of-Run Snapshot (Epoch 30)

- `val_total`: `-2.1084`
- `val_dist_mae`: `0.0442 m`
- `val_height_mae`: `0.0421 m`
- `val_az_mae`: `3.6836 deg`
- `val_side_acc`: `0.6055`
- `s_distance`: `-0.8794`
- `s_height`: `-0.7422`
- `s_azimuth`: `-1.6012`
- `s_side`: `-0.0539`

Note: Kendall total loss can go negative because of the additive `+ s_i` terms, so `val_total` is useful for model selection within this run but is not directly comparable to non-Kendall runs.

## Late-Stage Stability (Mean of Epochs 26-30)

- `val_total`: `-2.0324`
- `val_dist_mae`: `0.0353 m`
- `val_height_mae`: `0.0507 m`
- `val_az_mae`: `3.8514 deg`
- `val_side_acc`: `0.6043`
- `s_distance`: `-0.8581`
- `s_height`: `-0.7209`
- `s_azimuth`: `-1.5801`
- `s_side`: `-0.0374`

## Side Performance Detail

- Side confusion matrices show useful learning but uneven class recall.
- At epoch `30`, estimated side angular MAE (class-angle mapping `0/90/180/270`) is `~44.30 deg`.
- Best-side epoch `27` has similar side-angle estimate: `~43.83 deg`.

## Comparison vs Paper Benchmark

Paper best Mel config (44.1 kHz, FFT=1024, hop=256, Mel=256):

- Distance MAE: `0.36 m`
- Height MAE: `0.30 m`
- Azimuth MAE: `0.94 deg`
- Side MAE: `9.50 deg`

This run best observed values:

- Distance MAE: `0.029 m` (better, likely helped by easy within-recording split)
- Height MAE: `0.042 m` (better, likely helped by easy within-recording split)
- Azimuth MAE: `3.684 deg` (worse)
- Side MAE estimate: `~43.8 to 44.3 deg` (worse)

## Comparison vs `multitask_resnet_klbalance` (`exp01`)

- `kendall_v4` improves distance/height strongly.
- `kendall_v4` improves azimuth from `~5.55 deg` to `~3.68 deg`.
- `kendall_v4` is slightly lower on side accuracy (`~0.62` vs `~0.65` best).

## Saved Artifacts

- `best.pt` (epoch `30`)
- `last.pt` (epoch `30`)
- `metrics.csv`
- `side_confusion_epoch_001.csv` ... `side_confusion_epoch_030.csv`
- `config.json`
