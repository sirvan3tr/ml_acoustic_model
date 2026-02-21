from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acoustic_ml.data import AcousticUAVDataset, create_dataloader
from models.twostream_gcc_mel.augment import WaveformAugmenter
from models.twostream_gcc_mel.features import TwoStreamFeatureBuilder
from models.twostream_gcc_mel.losses import MultiTaskRobustLoss
from models.twostream_gcc_mel.model import TwoStreamGCCMelNet


@dataclass
class TrainConfig:
    data_root: str = "mic_array_data"
    output_dir: str = "models/twostream_gcc_mel/runs/default"
    seed: int = 42

    epochs: int = 30
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    window_seconds: float = 3.0
    hop_seconds: float = 1.5
    split_ratio: float = 0.75
    side_label_smoothing: float = 0.05
    w_distance: float = 1.0
    w_height: float = 1.0
    w_azimuth: float = 2.0
    w_azimuth_cls: float = 1.0
    w_side: float = 4.0
    w_side_kl: float = 0.5
    side_focal_gamma: float = 1.5
    azimuth_label_smoothing: float = 0.05

    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 256
    mel_height: int = 256
    mel_width: int = 256
    max_lag: int = 64
    include_gcc_std: bool = True

    mic_dropout_prob: float = 0.25
    mic_dropout_max: int = 1
    ambient_noise_prob: float = 0.30
    ambient_snr_db_min: float = 5.0
    ambient_snr_db_max: float = 20.0
    global_gain_prob: float = 0.30
    global_gain_db_max: float = 6.0
    per_channel_gain_prob: float = 0.15
    per_channel_gain_db_max: float = 1.5
    circular_shift_prob: float = 0.0
    disable_augment: bool = False

    max_train_steps: int | None = None
    max_val_steps: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def circular_mae_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    delta = torch.remainder(pred_deg - target_deg + 180.0, 360.0) - 180.0
    return delta.abs().mean()


def azimuth_xy_to_deg(azimuth_xy: torch.Tensor) -> torch.Tensor:
    unit_xy = torch.nn.functional.normalize(azimuth_xy, dim=1, eps=1e-6)
    radians = torch.atan2(unit_xy[:, 0], unit_xy[:, 1])
    degrees = torch.rad2deg(radians)
    return torch.remainder(degrees + 360.0, 360.0)


def batch_sample_rate(batch: dict[str, object]) -> int:
    sr = batch["metadata"]["sample_rate"]
    if isinstance(sr, torch.Tensor):
        return int(sr[0].item())
    if isinstance(sr, list):
        return int(sr[0])
    return int(sr)


def move_to_device(
    batch: dict[str, object], device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    audio = batch["audio"].to(device)
    regression = batch["targets"]["regression"].to(device)
    azimuth_xy = batch["targets"]["azimuth_sin_cos"].to(device)
    azimuth_class = batch["targets"]["azimuth_class"].to(device)
    side_class = batch["targets"]["rotation_class"].to(device)

    targets = {
        "distance": regression[:, 0],
        "height": regression[:, 1],
        "azimuth_deg": regression[:, 2],
        "azimuth_xy": azimuth_xy,
        "azimuth_class": azimuth_class,
        "side_class": side_class,
    }
    return audio, targets


def train_one_epoch(
    model: nn.Module,
    criterion: MultiTaskRobustLoss,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    feature_builder: TwoStreamFeatureBuilder,
    augmenter: WaveformAugmenter | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.train()
    running: dict[str, float] = {}
    count = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        audio, targets = move_to_device(batch, device)
        if augmenter is not None:
            audio, targets = augmenter(audio, targets)

        features = feature_builder.build(audio, sample_rate=batch_sample_rate(batch))
        outputs = model(features["mel"], features["gcc"], features["channel_energy"])
        loss, scalars = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        for key, value in scalars.items():
            running[key] = running.get(key, 0.0) + value
        count += 1

    if count == 0:
        raise RuntimeError("No training steps executed.")
    return {key: value / count for key, value in running.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: MultiTaskRobustLoss,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    feature_builder: TwoStreamFeatureBuilder,
    max_steps: int | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    running: dict[str, float] = {}
    count = 0
    side_confusion = np.zeros((4, 4), dtype=np.int64)

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        audio, targets = move_to_device(batch, device)
        features = feature_builder.build(audio, sample_rate=batch_sample_rate(batch))
        outputs = model(features["mel"], features["gcc"], features["channel_energy"])
        _, loss_scalars = criterion(outputs, targets)

        pred_side = outputs["side_logits"].argmax(dim=1)
        pred_azimuth_class = outputs["azimuth_class_logits"].argmax(dim=1)
        pred_azimuth_deg = azimuth_xy_to_deg(outputs["azimuth_xy"])
        metrics = {
            "distance_mae": float(
                torch.nn.functional.l1_loss(outputs["distance"], targets["distance"])
                .detach()
                .cpu()
            ),
            "height_mae": float(
                torch.nn.functional.l1_loss(outputs["height"], targets["height"])
                .detach()
                .cpu()
            ),
            "azimuth_circular_mae_deg": float(
                circular_mae_deg(pred_azimuth_deg, targets["azimuth_deg"]).detach().cpu()
            ),
            "azimuth_class_acc": float(
                (pred_azimuth_class == targets["azimuth_class"]).float().mean().detach().cpu()
            ),
            "side_accuracy": float(
                (pred_side == targets["side_class"]).float().mean().detach().cpu()
            ),
        }
        scalars = {**loss_scalars, **metrics}

        for key, value in scalars.items():
            running[key] = running.get(key, 0.0) + value
        count += 1

        true_side = targets["side_class"].detach().cpu().numpy()
        pred_side_np = pred_side.detach().cpu().numpy()
        for t, p in zip(true_side.tolist(), pred_side_np.tolist()):
            if 0 <= t < 4 and 0 <= p < 4:
                side_confusion[t, p] += 1

    if count == 0:
        raise RuntimeError("No validation steps executed.")
    return {key: value / count for key, value in running.items()}, side_confusion


def build_dataloaders(
    cfg: TrainConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_ds = AcousticUAVDataset(
        root_dir=cfg.data_root,
        split="train",
        sources=("Drone",),
        split_ratio=cfg.split_ratio,
        window_seconds=cfg.window_seconds,
        hop_seconds=cfg.hop_seconds,
        transform=None,
    )
    val_ds = AcousticUAVDataset(
        root_dir=cfg.data_root,
        split="test",
        sources=("Drone",),
        split_ratio=cfg.split_ratio,
        window_seconds=cfg.window_seconds,
        hop_seconds=cfg.hop_seconds,
        transform=None,
    )
    train_loader = create_dataloader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def write_metrics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_side_confusion_csv(path: Path, matrix: np.ndarray) -> None:
    labels = ["Front", "Right", "Back", "Left"]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true_pred", *labels])
        for idx, label in enumerate(labels):
            writer.writerow([label, *matrix[idx].tolist()])


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train two-stream GCC-PHAT + Mel multitask UAV model."
    )
    parser.add_argument("--data-root", default="mic_array_data")
    parser.add_argument("--output-dir", default="models/twostream_gcc_mel/runs/default")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=1.5)
    parser.add_argument("--split-ratio", type=float, default=0.75)
    parser.add_argument("--side-label-smoothing", type=float, default=0.05)
    parser.add_argument("--w-distance", type=float, default=1.0)
    parser.add_argument("--w-height", type=float, default=1.0)
    parser.add_argument("--w-azimuth", type=float, default=2.0)
    parser.add_argument("--w-azimuth-cls", type=float, default=1.0)
    parser.add_argument("--w-side", type=float, default=4.0)
    parser.add_argument("--w-side-kl", type=float, default=0.5)
    parser.add_argument("--side-focal-gamma", type=float, default=1.5)
    parser.add_argument("--azimuth-label-smoothing", type=float, default=0.05)

    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=256)
    parser.add_argument("--mel-height", type=int, default=256)
    parser.add_argument("--mel-width", type=int, default=256)
    parser.add_argument("--max-lag", type=int, default=64)
    parser.add_argument(
        "--no-gcc-std",
        action="store_false",
        dest="include_gcc_std",
        help="Disable GCC temporal std channels.",
    )
    parser.set_defaults(include_gcc_std=True)

    parser.add_argument("--mic-dropout-prob", type=float, default=0.25)
    parser.add_argument("--mic-dropout-max", type=int, default=1)
    parser.add_argument("--ambient-noise-prob", type=float, default=0.30)
    parser.add_argument("--ambient-snr-db-min", type=float, default=5.0)
    parser.add_argument("--ambient-snr-db-max", type=float, default=20.0)
    parser.add_argument("--global-gain-prob", type=float, default=0.30)
    parser.add_argument("--global-gain-db-max", type=float, default=6.0)
    parser.add_argument("--per-channel-gain-prob", type=float, default=0.15)
    parser.add_argument("--per-channel-gain-db-max", type=float, default=1.5)
    parser.add_argument("--circular-shift-prob", type=float, default=0.0)
    parser.add_argument("--disable-augment", action="store_true")

    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        split_ratio=args.split_ratio,
        side_label_smoothing=args.side_label_smoothing,
        w_distance=args.w_distance,
        w_height=args.w_height,
        w_azimuth=args.w_azimuth,
        w_azimuth_cls=args.w_azimuth_cls,
        w_side=args.w_side,
        w_side_kl=args.w_side_kl,
        side_focal_gamma=args.side_focal_gamma,
        azimuth_label_smoothing=args.azimuth_label_smoothing,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        mel_height=args.mel_height,
        mel_width=args.mel_width,
        max_lag=args.max_lag,
        include_gcc_std=args.include_gcc_std,
        mic_dropout_prob=args.mic_dropout_prob,
        mic_dropout_max=args.mic_dropout_max,
        ambient_noise_prob=args.ambient_noise_prob,
        ambient_snr_db_min=args.ambient_snr_db_min,
        ambient_snr_db_max=args.ambient_snr_db_max,
        global_gain_prob=args.global_gain_prob,
        global_gain_db_max=args.global_gain_db_max,
        per_channel_gain_prob=args.per_channel_gain_prob,
        per_channel_gain_db_max=args.per_channel_gain_db_max,
        circular_shift_prob=args.circular_shift_prob,
        disable_augment=args.disable_augment,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        device=args.device,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(asdict(cfg), fp, indent=2)

    train_loader, val_loader = build_dataloaders(cfg)
    feature_builder = TwoStreamFeatureBuilder(
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        mel_size=(cfg.mel_height, cfg.mel_width),
        max_lag=cfg.max_lag,
        num_channels=8,
        include_gcc_std=cfg.include_gcc_std,
    )

    gcc_channels = len(feature_builder.mic_pairs) * (2 if cfg.include_gcc_std else 1)
    model = TwoStreamGCCMelNet(
        mel_channels=8,
        gcc_channels=gcc_channels,
        energy_channels=8,
    ).to(device)
    criterion = MultiTaskRobustLoss(
        w_distance=cfg.w_distance,
        w_height=cfg.w_height,
        w_azimuth=cfg.w_azimuth,
        w_azimuth_cls=cfg.w_azimuth_cls,
        w_side=cfg.w_side,
        w_side_kl=cfg.w_side_kl,
        side_focal_gamma=cfg.side_focal_gamma,
        side_label_smoothing=cfg.side_label_smoothing,
        azimuth_label_smoothing=cfg.azimuth_label_smoothing,
    ).to(device)

    augmenter: WaveformAugmenter | None = None
    if not cfg.disable_augment:
        augmenter = WaveformAugmenter(
            data_root=cfg.data_root,
            mic_dropout_prob=cfg.mic_dropout_prob,
            mic_dropout_max=cfg.mic_dropout_max,
            ambient_noise_prob=cfg.ambient_noise_prob,
            ambient_snr_db_min=cfg.ambient_snr_db_min,
            ambient_snr_db_max=cfg.ambient_snr_db_max,
            global_gain_prob=cfg.global_gain_prob,
            global_gain_db_max=cfg.global_gain_db_max,
            per_channel_gain_prob=cfg.per_channel_gain_prob,
            per_channel_gain_db_max=cfg.per_channel_gain_db_max,
            circular_shift_prob=cfg.circular_shift_prob,
        )

    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.1)

    best_score = math.inf
    history: list[dict[str, float | int]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            feature_builder=feature_builder,
            augmenter=augmenter,
            max_steps=cfg.max_train_steps,
        )
        val_stats, side_confusion = evaluate(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            feature_builder=feature_builder,
            max_steps=cfg.max_val_steps,
        )
        scheduler.step()

        row: dict[str, float | int] = {"epoch": epoch}
        for key, value in train_stats.items():
            row[f"train_{key}"] = value
        for key, value in val_stats.items():
            row[f"val_{key}"] = value
        history.append(row)
        write_side_confusion_csv(out_dir / f"side_confusion_epoch_{epoch:03d}.csv", side_confusion)

        score = float(val_stats["total"])
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "criterion_state_dict": criterion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "val_total": score,
                },
                out_dir / "best.pt",
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
                "val_total": score,
            },
            out_dir / "last.pt",
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_total={train_stats['total']:.4f} | "
            f"val_total={val_stats['total']:.4f} | "
            f"val_dist_mae={val_stats['distance_mae']:.3f} | "
            f"val_height_mae={val_stats['height_mae']:.3f} | "
            f"val_az_mae={val_stats['azimuth_circular_mae_deg']:.3f} deg | "
            f"val_az_cls_acc={val_stats['azimuth_class_acc']:.3f} | "
            f"val_side_acc={val_stats['side_accuracy']:.3f} | "
            f"val_side_focal={val_stats['side_focal']:.3f} | "
            f"val_side_kl={val_stats['side_kl']:.3f} | "
            f"side_prior_mix={val_stats['side_prior_mix']:.3f}"
        )

    write_metrics_csv(out_dir / "metrics.csv", history)
    print(f"Training complete. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
