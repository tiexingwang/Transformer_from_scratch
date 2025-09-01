# -*- coding: utf-8 -*-
"""
Layer Normalization implementation for Transformer models.

This module implements layer normalization as described in "Attention Is All You Need".
It normalizes the input across the last dimension and applies learnable scale and shift.
"""

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Layer Normalization module for Transformer architectures.
    
    This module normalizes the input tensor across the last dimension (d_model)
    and applies learnable scale (gamma) and shift (beta) parameters. Layer
    normalization helps stabilize training and improve convergence.
    
    Attributes:
        gamma (nn.Parameter): Learnable scale parameter.
        beta (nn.Parameter): Learnable shift parameter.
        eps (float): Small value to prevent division by zero.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        """Initialize the LayerNorm module.
        
        Args:
            d_model (int): Dimension of the model embeddings.
            eps (float, optional): Small value to prevent division by zero. 
                Defaults to 1e-5.
        """
        super().__init__()
        # Learnable scale (gamma) and shift (beta) parameters
        self.gamma = nn.Parameter(torch.randn(d_model))  # Scale factor
        self.beta = nn.Parameter(torch.randn(d_model))   # Shift factor
        self.eps = eps
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to the input tensor.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape 
                [batch_size, sequence_length, d_model].
                
        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        # Input tensor shape: [batch_size, sequence_length, d_model]
        
        # Calculate mean across the last dimension (d_model)
        mean = torch.mean(input_tensor, dim=-1, keepdim=True)
        
        # Calculate variance across the last dimension (d_model)
        variance = torch.mean((input_tensor - mean) ** 2, dim=-1, keepdim=True)
        
        # Normalize: (x - mean) / sqrt(variance + eps)
        normalized = (input_tensor - mean) / torch.sqrt(variance + self.eps)
        
        # Apply learnable scale and shift: gamma * normalized + beta
        return normalized * self.gamma + self.beta
