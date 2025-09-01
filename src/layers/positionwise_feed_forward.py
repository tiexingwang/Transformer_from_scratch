# -*- coding: utf-8 -*-
"""
Position-wise Feed-Forward Network for Transformer models.

This module implements the position-wise feed-forward network as described in
"Attention Is All You Need". It applies two linear transformations with an
activation function and dropout between them.
"""

import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network for Transformers.
    
    This module applies the following transformation to each position independently:
    Linear(d_model -> d_ff) -> Activation -> Dropout -> Linear(d_ff -> d_model) -> Dropout
    
    The network preserves the input shape [batch_size, time_steps, d_model] and
    processes each position independently.
    
    Attributes:
        fcn1 (nn.Linear): First linear transformation layer.
        activation (nn.Module): Activation function.
        dropout1 (nn.Dropout): First dropout layer.
        fcn2 (nn.Linear): Second linear transformation layer.
        dropout2 (nn.Dropout): Second dropout layer.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        activation: str | nn.Module = "gelu",
        bias: bool = True,
        final_dropout: bool | None = None,  # None => same as dropout
    ):
        """Initialize the PositionwiseFeedForward module.
        
        Args:
            d_model (int): Input and output dimension of the model.
            d_ff (int, optional): Hidden dimension of the feed-forward network.
                If None, defaults to 4 * d_model.
            dropout (float): Dropout probability for the first dropout layer.
            activation (str | nn.Module): Activation function. Can be a string
                ('gelu', 'relu', 'silu') or a custom nn.Module.
            bias (bool): Whether to use bias in linear layers.
            final_dropout (bool | None): Whether to apply dropout after the second
                linear layer. If None, uses the same value as dropout.
                
        Raises:
            ValueError: If activation string is not supported.
            TypeError: If activation is not a string or nn.Module.
        """
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        # Parse activation function
        if isinstance(activation, str):
            activation_map = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
            if activation.lower() not in activation_map:
                raise ValueError(f"Unsupported activation '{activation}'. Choose from {list(activation_map)} or pass an nn.Module.")
            activation_module = activation_map[activation.lower()]()
        elif isinstance(activation, nn.Module):
            activation_module = activation
        else:
            raise TypeError("activation must be a string or an nn.Module")

        final_dropout_prob = dropout if final_dropout is None else final_dropout

        # Initialize layers
        self.fcn1 = nn.Linear(d_model, d_ff, bias=bias)
        self.activation = activation_module
        self.dropout1 = nn.Dropout(dropout)
        self.fcn2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout2 = nn.Dropout(final_dropout_prob)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the position-wise feed-forward network.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape 
                [batch_size, time_steps, d_model].
                
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, time_steps, d_model].
        """
        # Apply first linear transformation
        hidden = self.fcn1(input_tensor)
        # Apply activation function
        hidden = self.activation(hidden)
        # Apply first dropout
        hidden = self.dropout1(hidden)
        # Apply second linear transformation
        output = self.fcn2(hidden)
        # Apply final dropout
        output = self.dropout2(output)
        return output
    
    
# Sanity check for the PositionwiseFeedForward module
if __name__ == "__main__":
    d_model = 64
    num_heads = 4
    batch_size = 2
    sequence_len = 10

    input_tensor = torch.randn((batch_size, sequence_len, d_model))
    position_ff = PositionwiseFeedForward(d_model, 4*d_model)
    
    output_tensor = position_ff(input_tensor)

    print(f"input_tensor shape: {input_tensor.shape}")   # [2, 10, 64]
    print(f"output_tensor shape: {output_tensor.shape}") # [2, 10, 64]
