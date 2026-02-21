# Original Baseline Model

This directory contains the exact replication of the standard benchmark model detailed in the original Acoustic UAV dataset paper.

## Architecture specifications
- **Input Feature**: Log-Mel Spectrogram (8 channels directly stacked) at 44.1 kHz, 1024 FFT, 256 Hop, 256 Mels.
- **Model**: Custom 8-stage ResNet using Residual skip-connections.
- **Normalisation**: GroupNorm and GELU (rather than BatchNorm and ReLU).
- **Target Mapping**: The outputs map into 6 continuous regressors representing Distance, Height, and circular components for Azimuth $[sin, cos]$ and Side $[sin, cos]$.
- **Loss**: Simple continuous uniform MSE loss. No multi-task weighting is applied.
- **Optimizer**: Adam with learning rate `0.0002` and betas `(0.5, 0.9)`.
- **Scheduler**: ReduceLROnPlateau.

## Running
To train the baseline, run the following:

```bash
uv run models/original/train.py --device cuda:0
```
