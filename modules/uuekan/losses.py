"""
MedBoard — modules/uuekan/losses.py
Tri-Partite Supervision Loss for UUEKAN:
  L_total = L_seg (Deep Supervision Dice + BCE)
          + L_boundary (Sobel Boundary MSE)
          + L_uncertainty (Regularization + Consistency + Threshold)

Reference: Section 3.4 of Chen et al. (BSPC 2026).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Union


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy with Logits + Dice Loss."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred_logits, target)
        pred_prob = torch.sigmoid(pred_logits)

        b = target.size(0)
        p_flat = pred_prob.view(b, -1)
        t_flat = target.view(b, -1)

        intersection = (p_flat * t_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (p_flat.sum(dim=1) + t_flat.sum(dim=1) + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return 0.5 * bce + dice_loss


class BoundaryLoss(nn.Module):
    """
    Supervises object contours using Sobel filter MSE between prediction and ground truth.
    L_b = ||Sobel(sigma(y_hat)) - Sobel(y)||^2
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred_logits)

        pred_bx = F.conv2d(pred_prob, self.sobel_x, padding=1)
        pred_by = F.conv2d(pred_prob, self.sobel_y, padding=1)
        pred_boundary = torch.sqrt(pred_bx ** 2 + pred_by ** 2 + 1e-6)

        target_bx = F.conv2d(target, self.sobel_x, padding=1)
        target_by = F.conv2d(target, self.sobel_y, padding=1)
        target_boundary = torch.sqrt(target_bx ** 2 + target_by ** 2 + 1e-6)

        return F.mse_loss(pred_boundary, target_boundary)


class UncertaintyRegularizationLoss(nn.Module):
    """
    Correlates estimated uncertainty with prediction error:
      - Low uncertainty in correct regions
      - High uncertainty in incorrect regions
      - Moderate uncertainty (tau=0.6) along boundary contours
    """

    def __init__(self, boundary_weight: float = 0.5):
        super().__init__()
        self.boundary_weight = boundary_weight
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        uncertainty_map: torch.Tensor,
    ) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred_logits)
        error = torch.abs(pred_prob - target)

        target_bx = F.conv2d(target, self.sobel_x, padding=1)
        target_by = F.conv2d(target, self.sobel_y, padding=1)
        boundary_grad = torch.sqrt(target_bx ** 2 + target_by ** 2 + 1e-6)
        boundary_mask = (boundary_grad > 0.1).float()
        non_boundary_mask = 1.0 - boundary_mask

        # Correct region: low uncertainty
        correct_mask = (error < 0.2).float() * non_boundary_mask
        loss_correct = (uncertainty_map * correct_mask).mean()

        # Incorrect region: high uncertainty
        incorrect_mask = (error >= 0.2).float() * non_boundary_mask
        loss_incorrect = ((1.0 - uncertainty_map) * incorrect_mask * error).mean()

        # Boundary region: moderate uncertainty target (0.6)
        target_u = torch.ones_like(uncertainty_map) * 0.6
        loss_boundary = F.mse_loss(uncertainty_map * boundary_mask, target_u * boundary_mask)

        return loss_correct + 0.5 * loss_incorrect + self.boundary_weight * loss_boundary


class CrossScaleConsistencyLoss(nn.Module):
    """Enforces consistency across multi-scale uncertainty maps."""

    def __init__(self):
        super().__init__()

    def forward(self, uncertainty_maps: List[torch.Tensor]) -> torch.Tensor:
        if len(uncertainty_maps) <= 1:
            return torch.tensor(0.0, device=uncertainty_maps[0].device)

        loss = torch.tensor(0.0, device=uncertainty_maps[0].device)
        for i in range(len(uncertainty_maps) - 1):
            deeper = uncertainty_maps[i]
            shallower = uncertainty_maps[i + 1]
            upsampled_deep = F.interpolate(
                deeper, size=shallower.shape[2:], mode="bilinear", align_corners=False
            )
            loss = loss + F.mse_loss(upsampled_deep, shallower)

        return loss / (len(uncertainty_maps) - 1)


class DeepSupervisionLoss(nn.Module):
    """Computes weighted Dice+BCE loss across main and auxiliary decoder outputs."""

    def __init__(self, base_loss: nn.Module | None = None, weights: List[float] | None = None):
        super().__init__()
        self.base_loss = base_loss or BCEDiceLoss()
        self.weights = weights or [1.0, 0.8, 0.6, 0.4]

    def forward(
        self,
        main_output: torch.Tensor,
        aux_outputs: List[torch.Tensor] | None,
        target: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.weights[0] * self.base_loss(main_output, target)
        if aux_outputs:
            for idx, aux in enumerate(aux_outputs):
                w = self.weights[min(idx + 1, len(self.weights) - 1)]
                loss = loss + w * self.base_loss(aux, target)
        return loss


class CombinedUUEKANLoss(nn.Module):
    """
    Master Composite Loss Function for UUEKAN.
    L_total = L_seg + boundary_weight * L_boundary + uncertainty_weight * L_u
    """

    def __init__(
        self,
        use_boundary: bool = True,
        boundary_weight: float = 0.2,
        use_uncertainty_loss: bool = True,
        uncertainty_loss_weight: float = 0.1,
        deep_supervision_weights: List[float] | None = None,
        uncertainty_config: dict | None = None,
    ):
        super().__init__()
        self.use_boundary = use_boundary
        self.boundary_weight = boundary_weight
        self.use_uncertainty_loss = use_uncertainty_loss
        self.uncertainty_loss_weight = uncertainty_loss_weight

        self.seg_loss = DeepSupervisionLoss(weights=deep_supervision_weights)
        if use_boundary:
            self.boundary_loss = BoundaryLoss()

        if use_uncertainty_loss:
            u_cfg = uncertainty_config or {}
            self.u_reg_loss = UncertaintyRegularizationLoss(
                boundary_weight=u_cfg.get("boundary_weight", 0.5)
            )
            self.u_consistency_loss = CrossScaleConsistencyLoss()
            self.reg_weight = u_cfg.get("reg_weight", 1.0)
            self.consistency_weight = u_cfg.get("consistency_weight", 0.5)

    def forward(
        self,
        model_output: Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]], Dict[str, torch.Tensor]],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates total composite loss and returns detailed metric breakdown.
        """
        loss_dict: Dict[str, float] = {}

        if isinstance(model_output, dict):
            main_pred = model_output["pred"]
            aux_preds = model_output.get("aux_preds", [])
            unc_maps = model_output.get("uncertainty_maps", [])
        elif isinstance(model_output, (tuple, list)):
            main_pred = model_output[0]
            aux_preds = model_output[1] if len(model_output) > 1 else []
            unc_maps = model_output[2] if len(model_output) > 2 else []
        else:
            main_pred = model_output
            aux_preds = []
            unc_maps = []

        # 1. Segmentation Loss
        l_seg = self.seg_loss(main_pred, aux_preds, target)
        total_loss = l_seg
        loss_dict["loss_seg"] = l_seg.item()

        # 2. Boundary Loss
        if self.use_boundary:
            l_b = self.boundary_loss(main_pred, target)
            total_loss = total_loss + self.boundary_weight * l_b
            loss_dict["loss_boundary"] = l_b.item()

        # 3. Uncertainty Losses
        if self.use_uncertainty_loss and len(unc_maps) > 0:
            reg_loss = torch.tensor(0.0, device=main_pred.device)
            for u_map in unc_maps:
                if u_map.shape[2:] != main_pred.shape[2:]:
                    u_resized = F.interpolate(
                        u_map, size=main_pred.shape[2:], mode="bilinear", align_corners=False
                    )
                else:
                    u_resized = u_map
                reg_loss = reg_loss + self.u_reg_loss(main_pred, target, u_resized)
            reg_loss = reg_loss / len(unc_maps)

            con_loss = self.u_consistency_loss(unc_maps)
            l_u = self.reg_weight * reg_loss + self.consistency_weight * con_loss

            total_loss = total_loss + self.uncertainty_loss_weight * l_u
            loss_dict["loss_uncertainty"] = l_u.item()

        loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict
