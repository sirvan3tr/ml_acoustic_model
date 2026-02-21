# Acoustic ML

Shared tooling for UAV acoustic localization experiments.

## PyTorch Dataloader

Use `acoustic_ml.data.AcousticUAVDataset` for multi-channel windows from `mic_array_data`.

```python
from acoustic_ml.data import AcousticUAVDataset, create_dataloader

train_ds = AcousticUAVDataset(
    root_dir="mic_array_data",
    split="train",
    sources=("Drone",),
    window_seconds=1.0,
    hop_seconds=0.5,
)
train_loader = create_dataloader(train_ds, batch_size=16, num_workers=4)

batch = next(iter(train_loader))
audio = batch["audio"]                     # [B, C, T]
regression = batch["targets"]["regression"]  # [B, 3] -> distance, height, azimuth
```

`targets` includes:

- `regression`: continuous `[distance_m, height_m, azimuth_deg]` (`NaN` for missing labels)
- `azimuth_sin_cos`: circular encoding for azimuth
- `distance_class`, `height_class`, `azimuth_class`, `rotation_class`: class indices (`-1` if unavailable)

Each item also includes `metadata` with recording id, source, sample rate, and window offsets.
