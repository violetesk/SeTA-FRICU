"""
Transformer architecture components for clinical risk prediction.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import MCEAwareTimeALiBiAttention
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    """A single transformer block with pre-norm architecture.

    This block consists of:
    1. RMSNorm + MCE-Aware Time-ALiBi Attention + Residual
    2. RMSNorm + SwiGLU FFN + Residual

    Args:
        d_model: The dimension of the model.
        nhead: Number of attention heads.
        dim_feedforward: Dimension of the feedforward network.
        dropout: Dropout probability.
    """

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1
    ):
        super().__init__()

        # Pre-norm for attention
        self.norm1 = RMSNorm(d_model)
        self.attn = MCEAwareTimeALiBiAttention(d_model, nhead, dropout)

        # Pre-norm for FFN
        self.norm2 = RMSNorm(d_model)

        # SwiGLU feedforward network
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(d_model, dim_feedforward)  # Gate
        self.fc3 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            times: Timestamp tensor of shape [batch, seq_len].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape [batch, seq_len, d_model].
        """
        # Attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, times, attention_mask)
        x = residual + self.dropout(x)

        # FFN with residual (SwiGLU)
        residual = x
        x = self.norm2(x)
        x = F.silu(self.fc1(x)) * self.fc2(x)  # SwiGLU activation
        x = self.fc3(x)
        x = residual + self.dropout(x)

        return x


class PatientRiskTransformer(nn.Module):
    """Transformer model for patient risk prediction.

    This model processes sequences of clinical event embeddings and predicts
    risk scores for multiple time horizons at each timestep.

    Args:
        d_model: Dimension of input embeddings (default: 4096 for LLM embeddings).
        output_dim: Number of output dimensions (e.g., 1440 for minute-level predictions).
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dim_feedforward: Dimension of FFN hidden layer.
        dropout: Dropout probability.
        use_gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency.
    """

    def __init__(
        self,
        d_model: int = 4096,
        output_dim: int = 1440,
        nhead: int = 32,
        num_layers: int = 6,
        dim_feedforward: int = 11008,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.output_dim = output_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.final_norm = RMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights with Xavier uniform."""
        for name, param in self.named_parameters():
            # Skip MCE predictor parameters (they have custom initialization)
            if "mce_predictor" in name or "head_init_bias" in name:
                continue
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _create_attention_mask(
        self, times: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Create combined padding and causal attention mask.

        Args:
            times: Timestamp tensor of shape [batch, seq_len].
            device: Target device.
            dtype: Target dtype.

        Returns:
            Attention mask of shape [batch, 1, seq_len, seq_len].
        """
        batch_size, seq_len = times.shape
        mask_value = torch.finfo(dtype).min

        # Padding mask: positions where time == 0 are padding
        padding_mask = (times == 0).unsqueeze(1).unsqueeze(2)

        # Initialize attention bias
        attention_bias = torch.zeros(
            batch_size, 1, seq_len, seq_len, device=device, dtype=dtype
        )
        attention_bias = attention_bias.masked_fill(padding_mask, mask_value)

        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        attention_bias = attention_bias.masked_fill(causal_mask, mask_value)

        return attention_bias

    def forward(self, src: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            src: Input embeddings of shape [batch, seq_len, d_model].
            times: Timestamps of shape [batch, seq_len].

        Returns:
            Risk predictions of shape [batch, seq_len, output_dim].
        """
        # Ensure times has correct dtype
        if times.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            times = times.to(dtype=src.dtype)

        # Create attention mask
        attention_bias = self._create_attention_mask(times, src.device, src.dtype)

        # Process through transformer layers
        x = src
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, times, attention_bias, use_reentrant=False)
            else:
                x = layer(x, times, attention_bias)

        # Final norm and projection
        x = self.final_norm(x)
        output = self.output_projection(x)

        return output

    @property
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
