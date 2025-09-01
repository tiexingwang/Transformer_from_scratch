import torch
import torch.nn as nn

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.layers import MultiHeadAttention, LayerNorm, PositionwiseFeedForward

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        """
        Defines a Transformer Encoder Block consisting of:
        - Multi-Head Attention
        - Add & LayerNorm
        - Feedforward Network
        - Add & LayerNorm

        Args:
            d_model (int): Dimensionality of input/output
            num_heads (int): Number of attention heads
            d_ff (int): Hidden layer size in feedforward network
            dropout (float): Dropout rate (0.0–1.0)
        """
        super().__init__()

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input of shape [batch_size, seq_len, d_model]

        Returns:
            Tensor: Output of shape [batch_size, seq_len, d_model]
        """
        # Multi-head attention + residual + norm
        attn_output = self.mha(x, x, x)
        x = x + self.dropout1(attn_output)  # Residual connection first
        x = self.layer_norm1(x)             # Then layer normalization

        # Feedforward + residual + norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)    # Residual connection first
        x = self.layer_norm2(x)             # Then layer normalization

        return x

# Sanity check for EncoderBlock
if __name__ == "__main__":
    d_model = 64
    num_heads = 4
    batch_size = 2
    seq_len = 10
    d_ff = 4 * d_model
    dropout = 0.1

    x = torch.randn(batch_size, seq_len, d_model)
    encoder_block = EncoderBlock(d_model, num_heads, d_ff, dropout)
    out = encoder_block(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")

    

