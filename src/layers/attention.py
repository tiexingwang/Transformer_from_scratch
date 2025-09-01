# src/layers/attention.py
# Multi-head attention layer with explicit dimension names.

import torch
import torch.nn as nn
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """Initializes the MultiHeadAttention module.

        Args:
            d_model (int): The dimensionality of the model.
            num_heads (int): The number of attention heads.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Splits the last dimension into (num_heads, head_dim) and permutes.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_len, d_model].

        Returns:
            torch.Tensor: Tensor of shape [batch_size, num_heads, sequence_len, head_dim].
        """
        batch_size, sequence_len, d_model = x.size()
        assert d_model == self.d_model, "Input tensor's last dimension must match d_model."

        # Reshape to [batch_size, sequence_len, num_heads, head_dim]
        x = x.view(batch_size, sequence_len, self.num_heads, self.head_dim)

        # Permute to [batch_size, num_heads, sequence_len, head_dim]
        return x.permute(0, 2, 1, 3).contiguous()

    def concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenates the heads into the last dimension.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_heads, sequence_len, head_dim].

        Returns:
            torch.Tensor: Tensor of shape [batch_size, sequence_len, d_model].
        """
        batch_size, num_heads, sequence_len, head_dim = x.size()

        # Permute to [batch_size, sequence_len, num_heads, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()

        # Reshape to [batch_size, sequence_len, d_model]
        return x.view(batch_size, sequence_len, self.d_model)

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, num_heads, sequence_len, head_dim].
            key (torch.Tensor): Key tensor of shape [batch_size, num_heads, sequence_len, head_dim].
            value (torch.Tensor): Value tensor of shape [batch_size, num_heads, sequence_len, head_dim].
            mask (Optional[torch.Tensor]): Optional mask tensor of shape [batch_size, seq_len, seq_len] 
                or [batch_size, num_heads, seq_len, seq_len].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_heads, query_len, head_dim].
        """
        head_dim = query.size(-1)

        # Compute attention scores: [batch_size, num_heads, query_len, key_len]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)

        if mask is not None:
            # Handle mask broadcasting for multi-head attention
            if mask.dim() == 2:
                # mask shape: [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # There are two seq_len dimensions because in attention mechanisms,
                # we have a "query sequence length" (query_len) and a "key/value sequence length" (key_len).
                # Usually, for self-attention, query, key, and value all have the same length,
                # but in some cases (like cross-attention in the decoder), query_len and key_len can be different.
                # Therefore, the mask's last two dimensions are typically [query_len, key_len],
                # indicating which positions in the key/value can be attended to by each query position.
                # For example: mask.shape = [batch_size, num_heads, query_len, key_len]
                # This design allows the attention mechanism to flexibly support different input and output sequence lengths.
                if mask.size(0) == query.size(0) * query.size(1):
                    # Reshape from [batch_size * num_heads, seq_len, seq_len] to [batch_size, num_heads, seq_len, seq_len]
                    batch_size, num_heads = query.size(0), query.size(1)
                    mask = mask.view(batch_size, num_heads, mask.size(1), mask.size(2))
                else:
                    # mask shape: [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
                    mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                # mask shape: [batch_size, num_heads, seq_len, seq_len] - already correct
                pass
            else:
                raise ValueError(f"Mask must have 2, 3, or 4 dimensions, got {mask.dim()}")
            
            # Apply mask: mask should be broadcastable to scores shape
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights: [batch_size, num_heads, query_len, key_len]
        attention = torch.softmax(scores, dim=-1)

        # Weighted sum of values: [batch_size, num_heads, query_len, head_dim]
        return attention @ value

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for multi-head attention.

        Args:
            query (torch.Tensor): Input tensor of shape [batch_size, sequence_len, d_model].
            key (torch.Tensor): Key tensor of shape [batch_size, sequence_len, d_model].
            value (torch.Tensor): Value tensor of shape [batch_size, sequence_len, d_model].
            mask (Optional[torch.Tensor]): Optional mask tensor.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, sequence_len, d_model].
        """
        Q = self.split_heads(self.W_q(query))   # [batch_size, num_heads, sequence_len, head_dim]
        K = self.split_heads(self.W_k(key))     # [batch_size, num_heads, sequence_len, head_dim]
        V = self.split_heads(self.W_v(value))   # [batch_size, num_heads, sequence_len, head_dim]

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)  # [batch_size, num_heads, sequence_len, head_dim]

        concat_output = self.concat_heads(attention_output)  # [batch_size, sequence_len, d_model]

        return self.W_o(concat_output)  # [batch_size, sequence_len, d_model]


# Sanity check for the MultiHeadAttention module.
if __name__ == "__main__":
    d_model = 64
    num_heads = 4
    batch_size = 2
    sequence_len = 10

    input_tensor = torch.randn((batch_size, sequence_len, d_model))
    mha = MultiHeadAttention(d_model, num_heads)
    output_tensor = mha(input_tensor, input_tensor, input_tensor)

    print(f"input_tensor shape: {input_tensor.shape}")   # [2, 10, 64]
    print(f"output_tensor shape: {output_tensor.shape}") # [2, 10, 64]
