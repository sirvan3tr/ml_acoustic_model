
- This repo uses `uv` for package management. Run python code with `uv`, e.g. `uv run script.py`
- DO NOT use git add or commit.
- DO NOT venture out of the project's folder.
- We will have multiple models, so place the code for each model in a different folder

## Shared Data Pipeline (PyTorch)

- Use `acoustic_ml.data.AcousticUAVDataset` as the canonical dataset for all model folders.
- Use `acoustic_ml.data.create_dataloader` to build loaders (defaults to train-shuffle behavior).
- Do not duplicate raw data parsing in each model folder unless a model has a strict requirement that cannot be handled via dataset `transform`.
- Base dataset input is waveform tensor `[channels, time]` from `output.wav` (8-channel, 96 kHz).
- Default target fields exposed by the dataset:
  - `regression`: `[distance_m, height_m, azimuth_deg]`
  - `azimuth_sin_cos`: circular azimuth encoding for angle-aware losses
  - `distance_class`, `height_class`, `azimuth_class`, `rotation_class`
- Dataset split behavior should remain per-file temporal split (`75% train / 25% test`) unless explicitly changed for an experiment.

We are training a model using the following dataset.

## Dataset Overview

**Source:** Recordings of a single DJI Mavic 3 Cine UAV captured by a custom 8-microphone array ("Acoustic Head").

## Audio Specifications

- **Channels:** 8 (circular array, two levels of 4 mics each)
- **Sample Rate:** 96 kHz
- **Bit Depth:** 32-bit float
- **Format:** WAV (primary), MP3 (optional)
- **Microphone Type:** Røde NTG2 supercardioid shotgun mics — important because these are *directional*, not omnidirectional, which introduces gain variation depending on angle

## Dataset Scale

| Source | # Recordings | Size | Duration |
|---|---|---|---|
| Ambient Noise | 4 | 1.19 GB | 416 s |
| UAV | 128 | 14.61 GB | 5,120 s |
| Secondary (Zoom H4) | 18 | 16.15 GB | 33,090 s |

**Train/test split per file:** 75% train (30 s) / 25% test (10 s) — pre-split at the recording level, not randomized across samples.

## Spatial Parameters / Labels (Ground Truth)

Each recording has 4 labeled outputs — what any new model needs to predict:

| Label | Unit | Values in Dataset |
|---|---|---|
| **Distance (a)** | meters | 10 m, 20 m |
| **Height (b)** | meters | 10 m, 20 m |
| **Azimuth (c)** | degrees | 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° (8 orientations, 45° resolution) |
| **Side/Orientation (d)** | categorical (encoded as degrees) | Front, Back, Left, Right |

So the full combinatorial space is: **2 heights × 2 distances × 8 azimuths × 4 sides = 128 unique configurations** — which matches the 128 recorded UAV files exactly.

Labels are stored in paired `.json` files alongside each `.wav`. The JSON also includes weather metadata (temperature, humidity, wind speed/direction, barometric pressure) and exact microphone positions in WGS84 coordinates.


## Key Structural Considerations for Model Design

**1. Input shape:** Each sample is 8 channels × time. The existing baseline model treats each channel's feature map as a separate 256×256 image, stacking them as an 8-channel input tensor to a CNN.

**2. Task type:** The authors treated all 4 outputs as *continuous regression* (using MSE loss), including "side" which is really a 4-class classification problem. Your engineer may want to treat side as classification (CrossEntropy) separately from the regression outputs.

**3. Label imbalance:** The dataset has only 2 distinct values for both distance and height. This means regression models may effectively learn binary classification for those parameters. Consider whether a classification head makes more sense for distance and height given the limited variation.

**4. The azimuth ambiguity problem:** Azimuth is circular — 0° and 360° are the same. Standard MSE doesn't handle this; circular loss functions (e.g., von Mises loss) or sine/cosine encoding of the angle are recommended.


## Baseline Model Performance (for benchmarking)

The authors used a deep residual CNN (ResNet-inspired, 8 conv blocks, ~512 channels at deepest layer). Best results came from **Mel Spectrogram** features:

| Metric | Distance (m) | Height (m) | Azimuth (°) | Side (°) |
|---|---|---|---|---|
| **MAE** | ~0.36–0.57 | ~0.29–0.61 | ~0.81–1.39 | ~8.41–16.35 |
| **RMSE** | ~0.49–0.84 | ~0.38–0.90 | ~1.06–1.99 | ~25–34 |

Best single config: 44.1 kHz, FFT 1024, hop 512 (75% overlap), 256 Mel coefficients → MAE of 0.36 m / 0.30 m / 0.94° / 9.5° for distance/height/azimuth/side.

Side prediction has notably high RMSE (~25–34°) despite reasonable MAE, suggesting the model struggles with certain orientations — worth investigating with confusion matrices if reframing as classification.

## Paper Benchmarks (Table 4/5 Reference)

Use the following as primary comparison targets from the paper.

### Table 4 — Best result per feature type

| Feature | Sample Rate | MAE Distance (m) | MAE Height (m) | MAE Azimuth (°) | MAE Side (°) |
|---|---|---|---|---|---|
| MFCC | 96 kHz | 0.57 | 0.41 | 1.43 | 18.77 |
| Mel Spectrogram | 44.1 kHz | 0.39 | 0.41 | 0.91 | 8.61 |
| STFT | 44.1 kHz | 0.58 | 0.59 | 1.08 | 11.08 |
| LFCC | 44.1 kHz | 0.65 | 0.66 | 1.39 | 18.40 |
| Bark Spectrogram | 16 kHz | 0.41 | 0.47 | 1.04 | 14.04 |

### Table 5 — Best Mel Spectrogram configurations

| Sample Rate | FFT | Hop | Mel Coeffs | MAE Dist (m) | MAE Height (m) | MAE Azimuth (°) | MAE Side (°) |
|---|---|---|---|---|---|---|---|
| 44.1 kHz | 1024 | 256 | 256 | 0.36 | 0.30 | 0.94 | 9.50 |
| 44.1 kHz | 2048 | 1024 | 256 | 0.45 | 0.39 | 0.95 | 10.34 |
| 96 kHz | 4096 | 2048 | 128 | 0.38 | 0.47 | 0.81 | 10.24 |
| 96 kHz | 4096 | 2048 | 256 | 0.37 | 0.47 | 0.92 | 9.04 |

### Primary benchmark to beat

- Main target: **44.1 kHz, FFT=1024, hop=256, Mel=256** with MAE:
  - distance `0.36 m`
  - height `0.30 m`
  - azimuth `0.94°`
  - side `9.50°`

### Comparison caveats

- Compare both MAE and RMSE, especially for side (paper side RMSE is much higher than side MAE).
- Keep split protocols consistent when comparing numbers.
- If side is modeled as classification, convert predictions to an angle-space metric (or provide both classification and angle metrics).


## Feature Recommendations (from paper's ablation)

All five features were tested. Ranked by overall performance:
1. **Mel Spectrogram** — best overall
2. **Bark Spectrogram** — competitive, especially at low sample rates
3. **STFT** — good for azimuth/side
4. **LFCC** — moderate
5. **MFCC** — worst at low sample rates, catches up at 96 kHz

**Key hyperparameter findings:**
- 64 Mel/Bark/LFC coefficients consistently underperforms — use 128 or 256
- 75% hop length overlap (hop = FFT/4) tends to help distance prediction
- 50% overlap (hop = FFT/2) tends to help height prediction
- Going from 16 kHz → 44.1 kHz is a bigger gain than 44.1 kHz → 96 kHz


## Limitations of the dataset

- **Only one UAV model** (DJI Mavic 3 Cine) — generalization to other drones is untested
- **Only 2 distance and 2 height values** — very coarse; models will essentially learn a binary classifier for these, not a true continuous regressor
- **45° azimuth resolution** — relatively coarse for fine localization
- **Max range: 20 m** — short range; real-world utility at longer distances is unknown
- **Static UAV only** — no flight dynamics or motion
- **Single environment** — urban background noise, one location
- **No raw de-noised data** — recordings are unprocessed, wind/ambient noise included (weather data provided to help with this)
- **Directional mics** — TDoA-based classical methods are complicated by the supercardioid pattern; AI approaches are better suited
