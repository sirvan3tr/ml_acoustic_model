# Multitask ResNet Kendall V4

This model improves upon `v3` by addressing key limitations discovered during evaluation, specifically focused on spatial audio phase localization and early spatial dimensionality retention.

## What's New?

1. **Inter-Channel Phase Differences (IPD):**
   Prior models implicitly relied purely on energy differences (Inter-channel Level Differences, ILD). This model extracts the phase matrix during STFT, maps it to Mel filterbanks, and calculates the circular difference from the array-wide mean phase. This grants the network pseudo-TDOA (Time Difference of Arrival) features for localization, a critical feature class in acoustics for low-to-mid frequency localization.
2. **Early Stem Spatial Preservation:**
   By modifying the input layer of the initial block to `stride=1` (instead of `stride=2`), the high-resolution features and temporal fidelity are maintained early in the network, allowing the task heads more accurate representations to work with.
3. **Larger Feature Embedding Size:**
   Increased the final stage output logic from `128` channels to `256` channels. This provides dense context for the heavily parameterized MLP heads.

## Architecture Structure
We map the input waveform (8-channels over 96kHz) into `24` channels by cascading:
* 8x Normalized Mel Spectrograms
* 8x Relative Mel Channel Sensitivities
* 8x Normalised Phase IPDs

The model outputs remain standard multitask metrics with Kendall-homoscedastic weighted losses.

## How to Run
By default, the training script checks for multiple GPUs and utilizes the second GPU for load balancing so other models can run on the first.

```bash
# Optional: Ensure data isn't missing
# uv run -m scripts.download_dataset

uv run models/multitask_resnet_kendall_v4/train.py --device cuda:1
```

If you are only running to test it initially or have a 1-GPU workflow, it will utilize `cuda:0` securely.

```bash
# Pass explicitly, or remove device block
uv run models/multitask_resnet_kendall_v4/train.py --device cuda:0
```


## Intuition

1. Feature Engineering: Phase is the Signal, Magnitude is just the Envelope
The Problem with Standard Spectrograms: The baseline model (and v3) computed the Short-Time Fourier Transform (STFT) and immediately discarded the complex phase component by taking the absolute power (power = spec.abs().pow(2.0)). In psychoacoustics and signal processing, taking the magnitude gives you the Inter-channel Level Difference (ILD)—which microphone heard it the loudest. However, ILD is a terrible localization feature for low-frequency sounds because the wavelengths are larger than the microphone array itself, meaning the signal diffracts around the array with almost no loss of amplitude.

The Fix (v4 IPD): The true physics-based cue for localization is the Time Difference of Arrival (TDOA). Sound hits the closest microphone milliseconds before it hits the furthest one. In the frequency domain, this physical time delay manifests mathematically as a Phase Shift. By extracting the phase angle (spec.angle()), mapping it through the Mel filterbanks, and computing the circular difference relative to the array's mean phase (Inter-channel Phase Difference, IPD), we hand the network explicit TDOA proxy features. We are changing the problem from "learn the physics of sound from scratch using only volume" to "here are the exact phase correlations; build a manifold."

2. Architectural Receptive Fields: Don't Stride the Phase
The Problem with Standard ResNets: ResNet and typical CNN backbones are heavily optimized for ImageNet, where the dominant features are large, spatial, translation-invariant textures (like a dog's fur or a car tire). To optimize for this, typical stems start with a massive, aggressive downsampling layer (e.g., stride=2, pool=2, etc.).

The Fix (v4 Stride=1): In spatial audio processing—especially when phase is introduced—translation invariance is actually a hindrance. The exact micro-pixel alignment between channel 1's phase and channel 8's phase represents physical centimeters of distance. By reducing the initial convolutional stem from stride=2 to stride=1, we delay spatial downsampling. This forces the early convolutional filters to learn cross-channel correlations at maximum temporal and spectral resolution before we start pooling information away. We trade slightly higher FLOPs in the stem for preserving high-frequency phase alignment.

3. Loss Landscape: The Kendall Multi-Task Formulation
The Problem with Multi-Task Learning: We are predicting four fundamentally different target manifolds simultaneously:

Distance (Meters, L1 Loss)
Height (Meters, L1 Loss)
Azimuth (Continuous Circular Degrees, Cosine Embedding Loss)
Side (Categorical, Cross-Entropy Loss)
If you simply sum these losses (Total = L1 + L_cos + L_ce), the gradients will violently fight each other. The magnitude and scale of an L1 error on distance has zero mathematical relation to the probability scale of a Cross-Entropy loss.

The Fix (Kendall's Homoscedastic Uncertainty): Instead of manually tuning hyperparameters (e.g., Distance * 0.1 + Side * 5.0), we treat the task weights as learnable parameters. Using Kendall's formulation, we define a trainable log-variance parameter $\sigma_i$ for each task $i$. The loss for task $i$ becomes: $\frac{L_i}{e^{\sigma_i}} + \sigma_i$

If a task is noisy or difficult at the current epoch, the network increases $\sigma$ to reduce that task's gradient contribution (the dividing term), but pays a linear penalty (the additive $+ \sigma$ term) for doing so. It forces the network to dynamically balance the four loss gradients over the course of training without human intervention.

4. Continuous Azimuth Embedding
The Problem with Angle Regression: If a drone moves from $359^\circ \to 1^\circ$, the physical change is $2^\circ$, but standard MSE tries to penalize a massive $358^\circ$ error. The loss manifold has a severe discontinuity at $0^\circ / 360^\circ$ that causes gradient explosions.

The Fix: We map the scalar angle to a 2D Euclidean coordinate space [sin(θ), cos(θ)]. The network predicts this 2D vector, and we use Cosine Similarity loss to measure the angle between the prediction and the ground truth vector. This creates a perfectly smooth, continuous loss topology on the unit circle.
