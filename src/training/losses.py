"""
Loss functions for clinical risk prediction.

This module provides loss computation utilities including
weighted loss for handling class imbalance.
"""

from typing import Optional

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss for imbalanced risk prediction.

    Applies higher weights to positive samples to handle class imbalance
    common in clinical risk prediction tasks.

    Args:
        positive_weight: Weight multiplier for positive samples.
        reduction: How to reduce the loss ('none', 'mean', 'sum').
    """

    def __init__(self, positive_weight: float = 100.0, reduction: str = "none"):
        super().__init__()
        self.positive_weight = positive_weight
        self.base_loss = nn.MSELoss(reduction="none")
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute weighted MSE loss.

        Args:
            predictions: Model predictions [batch, seq, dim].
            targets: Ground truth labels [batch, seq, dim].
            mask: Optional padding mask [batch, seq] (True = padding).

        Returns:
            Loss tensor (scalar if reduction is 'mean' or 'sum').
        """
        raw_loss = self.base_loss(predictions, targets)

        # Apply class weights
        weight_pos = torch.tensor(
            self.positive_weight, dtype=predictions.dtype, device=predictions.device
        )
        weight_neg = torch.tensor(
            1.0, dtype=predictions.dtype, device=predictions.device
        )
        weights = torch.where(targets != 0, weight_pos, weight_neg)
        weighted_loss = raw_loss * weights

        # Apply padding mask if provided
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1)
            weighted_loss = weighted_loss.masked_fill(expanded_mask, 0.0)

        if self.reduction == "mean":
            if mask is not None:
                valid_elements = (~mask).sum() * predictions.shape[-1]
                return weighted_loss.sum() / valid_elements
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()

        return weighted_loss


def compute_masked_loss(
    loss_fn: nn.Module,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    output_dim: int,
) -> torch.Tensor:
    """Compute loss with masking for variable-length sequences.

    Args:
        loss_fn: Loss function (should have reduction='none').
        predictions: Model predictions [batch, seq, dim].
        targets: Ground truth labels [batch, seq, dim].
        mask: Padding mask [batch, seq] (True = padding).
        output_dim: Output dimension for computing valid elements.

    Returns:
        Mean loss over valid elements.
    """
    raw_loss = loss_fn(predictions, targets)

    # Mask padding positions
    expanded_mask = mask.unsqueeze(-1)
    raw_loss = raw_loss.masked_fill(expanded_mask, 0.0)

    # Compute mean over valid elements
    total_loss = raw_loss.sum()
    total_valid_steps = (~mask).sum()
    total_valid_elements = total_valid_steps * output_dim

    if total_valid_elements > 0:
        return total_loss / total_valid_elements

    return torch.tensor(0.0, device=predictions.device)
