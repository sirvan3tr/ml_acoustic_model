# Acoustic ML

Shared tooling for UAV acoustic localization experiments.

## Model Comparison (From REPORTs)

Best observed metrics per run from:
- `models/original_improved/runs/default/REPORT.md`
- `models/multitask_resnet_kendall_v4/runs/default/REPORT.md`
- `models/multitask_resnet_klbalance/runs/exp01/REPORT.md`
- `models/twostream_gcc_mel/runs/exp01/REPORT.md`

| Method | Run | Dist MAE (m) | Height MAE (m) | Azimuth MAE (deg) | Side Acc | Side MAE est (deg) |
|---|---|---:|---:|---:|---:|---:|
| `paper_baseline_mel` | `44.1k/1024/256/256` | 0.360 | 0.300 | 0.940 | - | 9.50 |
| `original_improved` | `default` | 0.713 | 0.770 | 2.489 | 0.818 | 18.11 |
| `multitask_resnet_kendall_v4` | `default` | 0.029 | 0.042 | 3.684 | 0.620 | 43.83 |
| `multitask_resnet_klbalance` | `exp01` | 0.112 | 0.059 | 5.546 | 0.648 | 40.08 |
| `twostream_gcc_mel` | `exp01` | 1.056 | 1.972 | 2.983 | 0.275 | 86.34 |

Paper benchmark reference (44.1 kHz, FFT=1024, hop=256, Mel=256): Distance `0.36 m`, Height `0.30 m`, Azimuth `0.94 deg`, Side `9.50 deg` MAE.

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
