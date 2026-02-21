from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class KendallMultiTaskLoss(nn.Module):
    """
    Learn task weights via log variances.

    Total: sum(exp(-s_i) * L_i + s_i)
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(4))

    @staticmethod
    def _azimuth_vector_loss(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
        pred_xy = F.normalize(pred_xy, dim=1, eps=1e-6)
        target_xy = F.normalize(target_xy, dim=1, eps=1e-6)
        cos_sim = (pred_xy * target_xy).sum(dim=1).clamp(-1.0, 1.0)
        return (1.0 - cos_sim).mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_distance = F.l1_loss(outputs["distance"], targets["distance"])
        loss_height = F.l1_loss(outputs["height"], targets["height"])
        loss_azimuth = self._azimuth_vector_loss(outputs["azimuth_xy"], targets["azimuth_xy"])
        loss_side = F.cross_entropy(outputs["side_logits"], targets["side_class"])

        raw_losses = torch.stack([loss_distance, loss_height, loss_azimuth, loss_side])
        weighted = torch.exp(-self.log_vars) * raw_losses + self.log_vars
        total = weighted.sum()

        scalars = {
            "total": float(total.detach().cpu()),
            "distance": float(loss_distance.detach().cpu()),
            "height": float(loss_height.detach().cpu()),
            "azimuth": float(loss_azimuth.detach().cpu()),
            "side": float(loss_side.detach().cpu()),
            "s_distance": float(self.log_vars[0].detach().cpu()),
            "s_height": float(self.log_vars[1].detach().cpu()),
            "s_azimuth": float(self.log_vars[2].detach().cpu()),
            "s_side": float(self.log_vars[3].detach().cpu()),
        }
        return total, scalars
