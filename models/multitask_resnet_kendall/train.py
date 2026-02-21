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
from models.multitask_resnet_kendall.features import MelSpectrogramTransform
from models.multitask_resnet_kendall.losses import KendallMultiTaskLoss
from models.multitask_resnet_kendall.model import MultiTaskResNet


@dataclass
class TrainConfig:
    data_root: str = "mic_array_data"
    output_dir: str = "models/multitask_resnet_kendall/runs/default"
    seed: int = 42

    epochs: int = 30
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    window_seconds: float = 1.5
    hop_seconds: float = 0.75
    split_ratio: float = 0.75

    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 256
    mel_height: int = 256
    mel_width: int = 256

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


def move_to_device(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    audio = batch["audio"].to(device)
    regression = batch["targets"]["regression"].to(device)
    side_class = batch["targets"]["rotation_class"].to(device)

    targets = {
        "distance": regression[:, 0],
        "height": regression[:, 1],
        "azimuth_deg": regression[:, 2],
        "side_class": side_class,
    }
    return audio, targets


def train_one_epoch(
    model: nn.Module,
    criterion: KendallMultiTaskLoss,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.train()
    running: dict[str, float] = {}
    count = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        audio, targets = move_to_device(batch, device)
        outputs = model(audio)
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
    criterion: KendallMultiTaskLoss,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.eval()
    running: dict[str, float] = {}
    count = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        audio, targets = move_to_device(batch, device)
        outputs = model(audio)
        _, loss_scalars = criterion(outputs, targets)

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
                circular_mae_deg(outputs["azimuth_deg"], targets["azimuth_deg"])
                .detach()
                .cpu()
            ),
            "side_accuracy": float(
                (outputs["side_logits"].argmax(dim=1) == targets["side_class"])
                .float()
                .mean()
                .detach()
                .cpu()
            ),
        }
        scalars = {**loss_scalars, **metrics}

        for key, value in scalars.items():
            running[key] = running.get(key, 0.0) + value
        count += 1

    if count == 0:
        raise RuntimeError("No validation steps executed.")
    return {key: value / count for key, value in running.items()}


def build_dataloaders(cfg: TrainConfig) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = MelSpectrogramTransform(
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        out_size=(cfg.mel_height, cfg.mel_width),
    )

    train_ds = AcousticUAVDataset(
        root_dir=cfg.data_root,
        split="train",
        sources=("Drone",),
        split_ratio=cfg.split_ratio,
        window_seconds=cfg.window_seconds,
        hop_seconds=cfg.hop_seconds,
        transform=transform,
    )
    val_ds = AcousticUAVDataset(
        root_dir=cfg.data_root,
        split="test",
        sources=("Drone",),
        split_ratio=cfg.split_ratio,
        window_seconds=cfg.window_seconds,
        hop_seconds=cfg.hop_seconds,
        transform=transform,
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


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train multitask UAV model with Kendall weighting.")
    parser.add_argument("--data-root", default="mic_array_data")
    parser.add_argument("--output-dir", default="models/multitask_resnet_kendall/runs/default")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--window-seconds", type=float, default=1.5)
    parser.add_argument("--hop-seconds", type=float, default=0.75)
    parser.add_argument("--split-ratio", type=float, default=0.75)

    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=256)
    parser.add_argument("--mel-height", type=int, default=256)
    parser.add_argument("--mel-width", type=int, default=256)

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
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        mel_height=args.mel_height,
        mel_width=args.mel_width,
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

    model = MultiTaskResNet(in_channels=8).to(device)
    criterion = KendallMultiTaskLoss().to(device)

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
            max_steps=cfg.max_train_steps,
        )
        val_stats = evaluate(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            max_steps=cfg.max_val_steps,
        )
        scheduler.step()

        row: dict[str, float | int] = {"epoch": epoch}
        for key, value in train_stats.items():
            row[f"train_{key}"] = value
        for key, value in val_stats.items():
            row[f"val_{key}"] = value
        history.append(row)

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
            f"val_side_acc={val_stats['side_accuracy']:.3f} | "
            f"s=[{val_stats['s_distance']:.3f}, {val_stats['s_height']:.3f}, "
            f"{val_stats['s_azimuth']:.3f}, {val_stats['s_side']:.3f}]"
        )

    write_metrics_csv(out_dir / "metrics.csv", history)
    print(f"Training complete. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
