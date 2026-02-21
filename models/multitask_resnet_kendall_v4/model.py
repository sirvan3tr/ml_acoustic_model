from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ResNetBackboneV4(nn.Module):
    def __init__(self, in_channels: int = 24) -> None:
        super().__init__()
        # V4 CHANGE: Stride reduced to 1 from 2 to keep spatial temporal relationships intact early
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(32, 32, stride=1)
        self.stage2 = self._make_stage(32, 64, stride=2)
        self.stage3 = self._make_stage(64, 128, stride=2)
        # V4 CHANGE: Increasing stage 4 channels from 128 to 256 for capacity
        self.stage4 = self._make_stage(128, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride),
            ResidualBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskResNet(nn.Module):
    """Shared backbone + four task heads with V4 features."""

    def __init__(self, in_channels: int = 24) -> None:
        super().__init__()
        # 8 Base Mel + 8 Relative Channels + 8 Phase IPDs = 24 Channels
        self.backbone = ResNetBackboneV4(in_channels=in_channels)
        features = 256  # V4 backbone output features

        self.distance_head = MLPHead(features, 1)
        self.height_head = MLPHead(features, 1)
        self.azimuth_head = MLPHead(features, 2)  # Output [sin, cos].
        self.side_head = MLPHead(features, 4)  # Front/Right/Back/Left logits.

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {
            "distance": self.distance_head(feat).squeeze(-1),
            "height": self.height_head(feat).squeeze(-1),
            "azimuth_xy": self.azimuth_head(feat),
            "side_logits": self.side_head(feat),
        }
