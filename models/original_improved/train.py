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
from models.original_improved.features import OriginalImprovedMelTransform
from models.original_improved.losses import ImprovedKendallMSELoss
from models.original_improved.model import ImprovedOriginalResNet


@dataclass
class TrainConfig:
    data_root: str = "mic_array_data"
    output_dir: str = "models/original_improved/runs/default"
    seed: int = 42

    epochs: int = 30
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 3e-4
    window_seconds: float = 1.5
    hop_seconds: float = 0.75
    split_ratio: float = 0.75

    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 256
    mel_height: int = 256
    mel_width: int = 256

    max_train_steps: int | None = None
    max_val_steps: int | None = None
    device: str = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def circular_mae_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    delta = torch.remainder(pred_deg - target_deg + 180.0, 360.0) - 180.0
    return delta.abs().mean()


def azimuth_xy_to_deg(azimuth_xy: torch.Tensor) -> torch.Tensor:
    unit_xy = torch.nn.functional.normalize(azimuth_xy, dim=1, eps=1e-6)
    radians = torch.atan2(unit_xy[:, 0], unit_xy[:, 1])
    degrees = torch.rad2deg(radians)
    return torch.remainder(degrees + 360.0, 360.0)


def side_class_to_xy(side_class: torch.Tensor) -> torch.Tensor:
    angles = side_class.float() * (math.pi / 2.0)
    return torch.stack([torch.sin(angles), torch.cos(angles)], dim=1)


def move_to_device(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    audio = batch["audio"].to(device)
    regression = batch["targets"]["regression"].to(device)
    azimuth_xy = batch["targets"]["azimuth_sin_cos"].to(device)
    side_class = batch["targets"]["rotation_class"].to(device)

    side_xy = side_class_to_xy(side_class)

    targets_6d = torch.stack([
        regression[:, 0] / 20.0,
        regression[:, 1] / 20.0,
        azimuth_xy[:, 0],
        azimuth_xy[:, 1],
        side_xy[:, 0],
        side_xy[:, 1],
    ], dim=1)

    eval_targets = {
        "distance": regression[:, 0],
        "height": regression[:, 1],
        "azimuth_deg": regression[:, 2],
        "side_class": side_class,
    }
    return audio, targets_6d, eval_targets


def train_one_epoch(
    model: nn.Module,
    criterion: ImprovedKendallMSELoss,
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

        audio, targets_6d, _ = move_to_device(batch, device)
        outputs = model(audio)
        loss, scalars = criterion(outputs, targets_6d)

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
    criterion: ImprovedKendallMSELoss,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_steps: int | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    running: dict[str, float] = {}
    count = 0
    side_confusion = np.zeros((4, 4), dtype=np.int64)

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        audio, targets_6d, eval_targets = move_to_device(batch, device)
        outputs_6d = model(audio)
        _, loss_scalars = criterion(outputs_6d, targets_6d)
        
        pred_distance = outputs_6d[:, 0] * 20.0
        pred_height = outputs_6d[:, 1] * 20.0
        pred_az_xy = outputs_6d[:, 2:4]
        pred_side_xy = outputs_6d[:, 4:6]

        pred_azimuth_deg = azimuth_xy_to_deg(pred_az_xy)
        pred_side_deg = azimuth_xy_to_deg(pred_side_xy)
        pred_side = torch.round(pred_side_deg / 90.0).long() % 4

        metrics = {
            "distance_mae": float(
                torch.nn.functional.l1_loss(pred_distance, eval_targets["distance"])
                .detach()
                .cpu()
            ),
            "height_mae": float(
                torch.nn.functional.l1_loss(pred_height, eval_targets["height"])
                .detach()
                .cpu()
            ),
            "azimuth_circular_mae_deg": float(
                circular_mae_deg(pred_azimuth_deg, eval_targets["azimuth_deg"])
                .detach()
                .cpu()
            ),
            "side_accuracy": float(
                (pred_side == eval_targets["side_class"])
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

        true_side = eval_targets["side_class"].detach().cpu().numpy()
        pred_side_np = pred_side.detach().cpu().numpy()
        for t, p in zip(true_side.tolist(), pred_side_np.tolist()):
            if 0 <= t < 4 and 0 <= p < 4:
                side_confusion[t, p] += 1

    if count == 0:
        raise RuntimeError("No validation steps executed.")
    return {key: value / count for key, value in running.items()}, side_confusion


def write_side_confusion_csv(path: Path, matrix: np.ndarray) -> None:
    labels = ["Front", "Right", "Back", "Left"]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true_pred", *labels])
        for idx, label in enumerate(labels):
            writer.writerow([label, *matrix[idx].tolist()])


def build_dataloaders(cfg: TrainConfig) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = OriginalImprovedMelTransform(
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        out_size=(cfg.mel_height, cfg.mel_width)
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


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train improved original paper baseline model.")
    parser.add_argument("--data-root", default="mic_array_data")
    parser.add_argument("--output-dir", default="models/original_improved/runs/default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)

    args = parser.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
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

    model = ImprovedOriginalResNet(in_channels=16).to(device)
    criterion = ImprovedKendallMSELoss().to(device)

    # Replaced unusual paper Adam params with standard AdamW + Scheduler
    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.learning_rate,
        weight_decay=1e-4
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
        val_stats, side_confusion = evaluate(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            max_steps=cfg.max_val_steps,
        )
        
        scheduler.step()
        score = float(val_stats["total_loss"])

        row: dict[str, float | int] = {"epoch": epoch}
        for key, value in train_stats.items():
            row[f"train_{key}"] = value
        for key, value in val_stats.items():
            row[f"val_{key}"] = value
        history.append(row)
        write_side_confusion_csv(out_dir / f"side_confusion_epoch_{epoch:03d}.csv", side_confusion)

        if score < best_score:
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "val_total": score,
                },
                out_dir / "best.pt",
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['total_loss']:.4f} | "
            f"val_loss={val_stats['total_loss']:.4f} | "
            f"val_dist_mae={val_stats['distance_mae']:.3f} | "
            f"val_height_mae={val_stats['height_mae']:.3f} | "
            f"val_az_mae={val_stats['azimuth_circular_mae_deg']:.3f} deg | "
            f"val_side_acc={val_stats['side_accuracy']:.3f} | "
            f"s_dist={val_stats['s_dist']:.3f} s_h={val_stats['s_height']:.3f} s_azx={val_stats['s_sin_az']:.3f}"
        )

    print(f"Training complete. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
