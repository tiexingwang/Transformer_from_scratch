import torch
import torch.nn as nn

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.layers import MultiHeadAttention, LayerNorm, PositionwiseFeedForward

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.mha_cross = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, 
                encoder_output: torch.Tensor, 
                decoder_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input of shape [batch_size, seq_len, d_model]
            encoder_output (Tensor): Output of the encoder of shape [batch_size, seq_len, d_model]
            decoder_mask (Tensor, optional): Mask of shape [batch_size, seq_len, seq_len] or [batch_size, num_heads, seq_len, seq_len]
        Returns:
            Tensor: Output of shape [batch_size, seq_len, d_model]
        """
        # Multi-head self-attention + residual + norm
        attn_output = self.mha(query=x, key=x, value=x, mask=decoder_mask)
        x = x + self.dropout1(attn_output)  # Residual connection first
        x = self.layer_norm1(x)             # Then layer normalization

        # Multi-head cross-attention + residual + norm
        # For cross-attention, we don't need a mask since we're attending to encoder output
        attn_output = self.mha_cross(query=x, key=encoder_output, value=encoder_output)
        x = x + self.dropout2(attn_output)  # Residual connection first
        x = self.layer_norm2(x)             # Then layer normalization

        # Feedforward + residual + norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)    # Residual connection first
        x = self.layer_norm3(x)             # Then layer normalization

        return x
    
# Sanity check for the DecoderBlock module
if __name__ == "__main__":
    d_model = 64
    num_heads = 4
    d_ff = 256
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    x = torch.randn(batch_size, seq_len, d_model)
    encoder_output = torch.randn(batch_size, seq_len, d_model)
    decoder_mask = torch.ones(batch_size, seq_len, seq_len)  # No masking for testing
    decoder_block = DecoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
    out = decoder_block(x, encoder_output, decoder_mask)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")