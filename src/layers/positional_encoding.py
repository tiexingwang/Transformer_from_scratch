# src/layers/positional_encoding.py
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in "Attention Is All You Need".
    
    This module precomputes a positional encoding buffer and adds it to the input
    tensor. The encoding uses sine and cosine functions of different frequencies
    to encode position information.
    
    Attributes:
        sequence_length (int): Maximum sequence length for positional encoding.
        d_model (int): Dimension of the model embeddings.
        wavelength (float): Base wavelength for frequency calculation.
        pe (torch.Tensor): Precomputed positional encoding buffer.
    """
    
    def __init__(self, wavelength: float, 
                 sequence_length: int, 
                 d_model: int, 
                 eps: float = 1e-9):
        """Initialize the PositionalEncoding module.
        
        Args:
            wavelength (float): Base wavelength for frequency calculation.
            sequence_length (int): Maximum sequence length for positional encoding.
            d_model (int): Dimension of the model embeddings.
            eps (float, optional): Small value to prevent division by zero. Defaults to 1e-9.
            
        Raises:
            AssertionError: If d_model is not even (required for sin/cos pairing).
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sin/cos pairing"
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.wavelength = float(wavelength)

        # Create position indices: [sequence_length, 1]
        position_indices = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        # Create frequency indices: [1, d_model//2]
        frequency_indices = torch.arange(d_model // 2, dtype=torch.float32).unsqueeze(0)

        # Calculate angles: position / wavelength^(2*frequency/d_model)
        # Shape: [sequence_length, d_model//2]
        angles = position_indices / (self.wavelength ** (2.0 * frequency_indices / d_model) + eps)

        # Initialize positional encoding tensor: [sequence_length, d_model]
        pos_encoding = torch.empty(sequence_length, d_model, dtype=torch.float32)
        pos_encoding[:, 0::2] = torch.sin(angles)  # Even columns: sine
        pos_encoding[:, 1::2] = torch.cos(angles)  # Odd columns: cosine

        # Register as buffer so it moves with .to(device) and isn't trainable
        # Shape: [1, sequence_length, d_model]
        self.register_buffer("pe", pos_encoding.unsqueeze(0))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input tensor.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape 
                [batch_size, time_steps, d_model].
                
        Returns:
            torch.Tensor: Input tensor with positional encoding added.
            
        Raises:
            AssertionError: If input d_model doesn't match positional encoding d_model.
            ValueError: If time_steps exceeds precomputed sequence_length.
        """
        batch_size, time_steps, d_model = input_tensor.shape
        assert d_model == self.d_model, "Input d_model does not match positional encoding d_model"
        if time_steps > self.sequence_length:
            raise ValueError(f"time_steps={time_steps} exceeds precomputed sequence_length={self.sequence_length}")
        return input_tensor + self.pe[:, :time_steps, :].to(dtype=input_tensor.dtype)
