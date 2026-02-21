# Improved Original Baseline Model

This directory contains a fortified version of the paper's original 6-target generic ResNet model. 

## Improvements
1. **Explicit Phase Extraction**: We extract the Inter-channel phase differences (IPD) from the spectrogram and append them to the Mel features (creating 16 input channels instead of 8). This gives the model direct spatial cues based on time-difference of arrival (TDOA).
2. **Early Spatial Downsampling Delay**: The original model aggressively halves the spatial dimensions starting at `conv2`. We changed `conv2_x` to preserve stride 1 initially to allow the phase features deeper representation before pooling.
3. **Kendall's Homoscedastic Weighting**: Replaced the uniform MSE loss block applied across 6 wildly different parameters (which caused severe gradient fighting) with Kendall's learnable uncertainty framework. The model dynamically balances Distance, Height, and Angle loss values itself during training.
4. **Stable Optimization**: Migrated the legacy GAN-style optimizer mapping (Adam betas 0.5, 0.9 + ReduceLROnPlateau) to standard `AdamW` and `CosineAnnealingLR` schedules for stabler minimums.

## Running
To train the improved baseline, run the following:

```bash
uv run models/original_improved/train.py --device cuda:0
```
