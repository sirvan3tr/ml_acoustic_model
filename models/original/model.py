from __future__ import annotations

import torch
from torch import nn


class ResNetBlockOriginal(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        # Ensure num_groups divides channels evenly
        groups_in = min(32, in_channels)
        groups_out = min(32, out_channels)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels)
        self.gelu = nn.GELU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=groups_out, num_channels=out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out = out + identity
        out = self.gelu(out)
        return out


class OriginalPaperResNet(nn.Module):
    """
    Original 8-stage ResNet baseline.
    Parameters:
      GroupNorm, GELU, and specific stride patterns.
    """

    def __init__(self, in_channels: int = 8) -> None:
        super().__init__()

        # Conv1: 32 channels. stride 1. [B, 32, 256, 256]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, 32), num_channels=32),
            nn.GELU()
        )

        # Stage 2: 64 channels. 1, then 2. --> [B, 64, 128, 128]
        self.conv2_x = self._make_stage(32, 64, stride=2)
        
        # Stage 3: 128 channels. 1, then 2. --> [B, 128, 64, 64]
        self.conv3_x = self._make_stage(64, 128, stride=2)
        
        # Stage 4: 192 channels. 1, then 2. --> [B, 192, 32, 32]
        self.conv4_x = self._make_stage(128, 192, stride=2)
        
        # Stage 5: 256 channels. 1, then 2. --> [B, 256, 16, 16]
        self.conv5_x = self._make_stage(192, 256, stride=2)
        
        # Stage 6: 512 channels. 1, then 2. --> [B, 512, 8, 8]
        self.conv6_x = self._make_stage(256, 512, stride=2)
        
        # Stage 7: 512 channels. 1, then 2. --> [B, 512, 4, 4]
        self.conv7_x = self._make_stage(512, 512, stride=2)
        
        # Stage 8: 512 channels. stride 1. --> [B, 512, 4, 4]
        self.conv8_x = nn.Sequential(
            ResNetBlockOriginal(512, 512, stride=1),
            ResNetBlockOriginal(512, 512, stride=1)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Head: fc1(512->256), Dropout, fc2(256->6)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 6)
        )

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        # Paper says: layer 1 at stride 1, layer 2 at stride 2
        return nn.Sequential(
            ResNetBlockOriginal(in_channels, out_channels, stride=1),
            ResNetBlockOriginal(out_channels, out_channels, stride=stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.conv6_x(x)
        x = self.conv7_x(x)
        x = self.conv8_x(x)
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
