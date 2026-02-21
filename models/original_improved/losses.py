from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ImprovedKendallMSELoss(nn.Module):
    """
    Improved loss: Uses Kendall's Homoscedastic formulation to dynamically
    balance the 6 continuous variables independently, rather than uniform MSE.
    outputs: [B, 6] (Distance, Height, Sin(Az), Cos(Az), Sin(Side), Cos(Side))
    """

    def __init__(self) -> None:
        super().__init__()
        # Learnable log variances for each of the 6 targets
        self.log_vars = nn.Parameter(torch.zeros(6))

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Calculate raw independent MSE for each target
        raw_losses = torch.stack([
            F.mse_loss(outputs[:, 0], targets[:, 0]),
            F.mse_loss(outputs[:, 1], targets[:, 1]),
            F.mse_loss(outputs[:, 2], targets[:, 2]),
            F.mse_loss(outputs[:, 3], targets[:, 3]),
            F.mse_loss(outputs[:, 4], targets[:, 4]),
            F.mse_loss(outputs[:, 5], targets[:, 5]),
        ])
        
        # Kendall Weighting
        weights = torch.exp(-self.log_vars)
        weighted = weights * raw_losses + self.log_vars
        total_loss = weighted.sum()

        with torch.no_grad():
            l1_dist = F.l1_loss(outputs[:, 0], targets[:, 0]).item()
            l1_height = F.l1_loss(outputs[:, 1], targets[:, 1]).item()
            l1_sin_az = F.l1_loss(outputs[:, 2], targets[:, 2]).item()

        scalars = {
            "total_loss": total_loss.item(),
            "dist_mae": l1_dist,
            "height_mae": l1_height,
            "s_dist": self.log_vars[0].item(),
            "s_height": self.log_vars[1].item(),
            "s_sin_az": self.log_vars[2].item(),
        }
        return total_loss, scalars
