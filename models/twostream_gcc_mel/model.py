from __future__ import annotations

import torch
from torch import nn


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MelStream(nn.Module):
    def __init__(self, in_channels: int = 8, embedding_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            Conv2dBlock(in_channels, 32, stride=1),
            Conv2dBlock(32, 64, stride=2),
            Conv2dBlock(64, 96, stride=2),
            Conv2dBlock(96, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.proj(x)


class GCCStream(nn.Module):
    def __init__(self, in_channels: int = 28, embedding_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            Conv1dBlock(in_channels, 64, stride=1),
            Conv1dBlock(64, 96, stride=1),
            Conv1dBlock(96, 128, stride=1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.proj(x)


class EnergyStream(nn.Module):
    def __init__(self, in_features: int = 8, embedding_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeadMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskTower(nn.Module):
    def __init__(self, in_features: int, out_features: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoStreamGCCMelNet(nn.Module):
    def __init__(
        self,
        mel_channels: int = 8,
        gcc_channels: int = 28,
        energy_channels: int = 8,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.mel_stream = MelStream(in_channels=mel_channels, embedding_dim=embedding_dim)
        self.gcc_stream = GCCStream(in_channels=gcc_channels, embedding_dim=embedding_dim)
        self.energy_stream = EnergyStream(in_features=energy_channels, embedding_dim=32)

        fused_dim = embedding_dim * 2 + 32
        self.range_tower = TaskTower(fused_dim, out_features=192)
        self.direction_tower = TaskTower(fused_dim, out_features=192)

        self.distance_head = HeadMLP(192, 1)
        self.height_head = HeadMLP(192, 1)
        self.azimuth_head = HeadMLP(192, 2)  # [sin, cos].
        self.azimuth_class_head = HeadMLP(192, 8)  # 8-way azimuth bins.
        self.side_head = HeadMLP(192, 4)  # Front/Right/Back/Left logits.
        self.side_energy_prior = nn.Linear(32, 4)
        self.side_prior_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self, mel: torch.Tensor, gcc: torch.Tensor, channel_energy: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        mel_feat = self.mel_stream(mel)
        gcc_feat = self.gcc_stream(gcc)
        energy_feat = self.energy_stream(channel_energy)
        fused = torch.cat([mel_feat, gcc_feat, energy_feat], dim=1)
        range_feat = self.range_tower(fused)
        direction_feat = self.direction_tower(fused)
        side_main = self.side_head(direction_feat)
        side_prior = self.side_energy_prior(energy_feat)
        side_mix = torch.sigmoid(self.side_prior_logit)
        return {
            "distance": self.distance_head(range_feat).squeeze(-1),
            "height": self.height_head(range_feat).squeeze(-1),
            "azimuth_xy": self.azimuth_head(direction_feat),
            "azimuth_class_logits": self.azimuth_class_head(direction_feat),
            "side_logits": side_main + side_mix * side_prior,
            "side_prior_mix": side_mix,
        }
