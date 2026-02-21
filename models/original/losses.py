from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class OriginalMSELoss(nn.Module):
    """
    Original paper loss: Uniform MSE across all 6 continuous outputs.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # outputs: [B, 6]
        # targets: [B, 6]
        
        loss = F.mse_loss(outputs, targets)

        # Break out individual MAEs for monitoring (optional, but helps matching v3/v4 logs)
        with torch.no_grad():
            l1_dist = F.l1_loss(outputs[:, 0], targets[:, 0]).item()
            l1_height = F.l1_loss(outputs[:, 1], targets[:, 1]).item()
            l1_sin_az = F.l1_loss(outputs[:, 2], targets[:, 2]).item()
            l1_cos_az = F.l1_loss(outputs[:, 3], targets[:, 3]).item()
            l1_sin_side = F.l1_loss(outputs[:, 4], targets[:, 4]).item()
            l1_cos_side = F.l1_loss(outputs[:, 5], targets[:, 5]).item()

        scalars = {
            "total_mse": loss.item(),
            "dist_mae": l1_dist,
            "height_mae": l1_height,
            "sin_az_mae": l1_sin_az,
            "cos_az_mae": l1_cos_az,
            "sin_side_mae": l1_sin_side,
            "cos_side_mae": l1_cos_side,
        }
        return loss, scalars
