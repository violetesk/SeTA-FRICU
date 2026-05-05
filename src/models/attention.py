"""
MCE-Aware Time-ALiBi Attention mechanism for temporal clinical data.

This module implements a specialized attention mechanism that combines:
- Multi-head self-attention
- Time-aware ALiBi (Attention with Linear Biases) positional encoding
- MCE (Multi-scale Clinical Event) prediction for dynamic attention modulation
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCEAwareTimeALiBiAttention(nn.Module):
    """Multi-head attention with MCE-aware time-based ALiBi biases.

    This attention mechanism learns to predict slope and peak parameters
    that modulate attention weights based on temporal distance between events.
    This is particularly useful for clinical time-series data where the
    relevance of past events varies with time.

    Args:
        d_model: The dimension of the model (embedding dimension).
        nhead: Number of attention heads.
        dropout: Dropout probability.
    """

    # Class-level constants for parameter bounds
    MAX_SLOPE_LIMIT: float = 2.5
    MAX_SLOPE: float = 2.0
    MIN_SLOPE: float = 0.2
    MAX_PEAK: float = 10.0

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # MCE predictor: predicts slope scale and peak offset
        self.mce_predictor = nn.Sequential(
            nn.Linear(self.head_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 2),  # [slope_scale, peak_offset]
        )

        # Static priors (learnable baselines)
        self.register_buffer("static_slope", torch.zeros(1, nhead, 1, 1))
        self.register_buffer("static_peak_bias", torch.zeros(1, nhead, 1, 1))

        # Cached values for analysis (only populated during eval)
        self.last_attn_weights: Optional[torch.Tensor] = None
        self.cached_slope: Optional[torch.Tensor] = None
        self.cached_peak: Optional[torch.Tensor] = None

        self._init_priors()

    def _init_priors(self) -> None:
        """Initialize prior values for slope and peak with neutral perturbation."""
        target_peaks = torch.linspace(0, self.MAX_PEAK * 0.95, self.nhead)

        with torch.no_grad():
            # Small random perturbation around 1.0 for slopes
            noise = (torch.rand(self.nhead) - 0.5) * 0.1
            target_slopes = torch.ones(self.nhead) + noise
            self.static_slope[0, :, 0, 0] = target_slopes

            # Convert peaks to logits
            peak_ratio = torch.clamp(target_peaks / self.MAX_PEAK, 0.05, 0.95)
            target_peak_logits = torch.log(peak_ratio / (1 - peak_ratio))
            self.static_peak_bias[0, :, 0, 0] = target_peak_logits

        # Zero-initialize predictor output layer for stable start
        nn.init.zeros_(self.mce_predictor[-1].weight)
        nn.init.zeros_(self.mce_predictor[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with time-aware attention.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            times: Timestamp tensor of shape [batch, seq_len].
            attention_mask: Optional mask tensor of shape [batch, 1, seq_len, seq_len].

        Returns:
            Output tensor of shape [batch, seq_len, d_model].
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        attn_score = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(self.head_dim)

        # Compute time-based biases
        time_mat = torch.abs(times.unsqueeze(2) - times.unsqueeze(1)).unsqueeze(1)
        log_time_diff = torch.log(time_mat / 60.0 + 1.0)

        # Predict MCE parameters from queries
        q_input = q.permute(0, 2, 1, 3)  # [B, S, H, D]
        mce_delta = self.mce_predictor(q_input).permute(0, 2, 1, 3)  # [B, H, S, 2]

        # Compute dynamic slope and peak
        slope_scale = torch.exp(mce_delta[..., 0:1])
        slope = (self.static_slope * slope_scale).clamp(1e-4, self.MAX_SLOPE_LIMIT)

        peak_logit = self.static_peak_bias + mce_delta[..., 1:2] * 4
        peak = torch.sigmoid(peak_logit) * self.MAX_PEAK

        # Compute MCE bias (tent function centered at peak)
        dist_to_peak = torch.abs(log_time_diff - peak)
        mce_bias = -slope * dist_to_peak

        # Combine scores
        combined_mask = attn_score + mce_bias

        if attention_mask is not None:
            combined_mask = combined_mask + attention_mask

        # Softmax and dropout
        probs = F.softmax(combined_mask, dim=-1)

        # Cache for analysis during evaluation
        if not self.training:
            self.last_attn_weights = probs.detach().cpu()
            self.cached_slope = slope.detach().cpu()
            self.cached_peak = peak.detach().cpu()

        probs = self.dropout(probs)

        # Apply attention to values
        out = torch.einsum("bhqk,bhkd->bhqd", probs, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, d_model)

        return self.out_proj(out)

    def get_cached_params(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached slope and peak parameters from last forward pass.

        Returns:
            Tuple of (cached_slope, cached_peak), or (None, None) if not available.
        """
        return self.cached_slope, self.cached_peak
