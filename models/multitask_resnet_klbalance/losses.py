from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiTaskKLBalancedLoss(nn.Module):
    """
    Fixed multitask weighting + KL class-balance regularizer for side logits.
    """

    def __init__(
        self,
        w_distance: float = 1.0,
        w_height: float = 1.0,
        w_azimuth: float = 1.0,
        w_side: float = 2.0,
        w_side_kl: float = 0.5,
        side_label_smoothing: float = 0.05,
        side_num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.w_distance = w_distance
        self.w_height = w_height
        self.w_azimuth = w_azimuth
        self.w_side = w_side
        self.w_side_kl = w_side_kl
        self.side_label_smoothing = side_label_smoothing
        self.side_num_classes = side_num_classes

    @staticmethod
    def _azimuth_vector_loss(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
        pred_xy = F.normalize(pred_xy, dim=1, eps=1e-6)
        target_xy = F.normalize(target_xy, dim=1, eps=1e-6)
        cos_sim = (pred_xy * target_xy).sum(dim=1).clamp(-1.0, 1.0)
        return (1.0 - cos_sim).mean()

    def _side_balance_kl(self, side_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(side_logits, dim=1)
        mean_probs = probs.mean(dim=0).clamp_min(1e-8)
        uniform = torch.full_like(mean_probs, 1.0 / self.side_num_classes)
        # KL(U || mean_pred) discourages class-collapse in batch predictions.
        return F.kl_div(mean_probs.log(), uniform, reduction="sum")

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_distance = F.l1_loss(outputs["distance"], targets["distance"])
        loss_height = F.l1_loss(outputs["height"], targets["height"])
        loss_azimuth = self._azimuth_vector_loss(outputs["azimuth_xy"], targets["azimuth_xy"])
        loss_side_ce = F.cross_entropy(
            outputs["side_logits"],
            targets["side_class"],
            label_smoothing=self.side_label_smoothing,
        )
        loss_side_kl = self._side_balance_kl(outputs["side_logits"])
        loss_side = loss_side_ce + self.w_side_kl * loss_side_kl

        total = (
            self.w_distance * loss_distance
            + self.w_height * loss_height
            + self.w_azimuth * loss_azimuth
            + self.w_side * loss_side
        )

        scalars = {
            "total": float(total.detach().cpu()),
            "distance": float(loss_distance.detach().cpu()),
            "height": float(loss_height.detach().cpu()),
            "azimuth": float(loss_azimuth.detach().cpu()),
            "side_ce": float(loss_side_ce.detach().cpu()),
            "side_kl": float(loss_side_kl.detach().cpu()),
            "side": float(loss_side.detach().cpu()),
        }
        return total, scalars

