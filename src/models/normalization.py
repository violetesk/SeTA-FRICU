"""
Normalization layers for transformer models.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm is a simplification of LayerNorm that removes the mean centering
    operation while maintaining the scaling benefits. This results in faster
    computation with comparable performance.

    Args:
        dim: The dimension of the input features.
        eps: A small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape [..., dim].

        Returns:
            Normalized tensor of the same shape.
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight
