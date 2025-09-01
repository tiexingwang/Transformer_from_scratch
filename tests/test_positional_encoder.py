# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 10:56:19 2025

@author: twang
"""

# tests/test_positional_encoding.py
import math
import torch
from torch.testing import assert_close
import sys
import os

# Adjust the import if your path/name differs
from src.layers import PositionalEncoding

def datacamp_formula_pe(time_steps: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """
    Recreate the PE table exactly like the DataCamp snippet:
        pe[:, 0::2] = sin(position * div_term)
        pe[:, 1::2] = cos(position * div_term)
    where div_term = exp(arange(0, d_model, 2) * -(log(base)/d_model))
    Returns shape [time_steps, d_model] on CPU, float32.
    """
    assert d_model % 2 == 0, "d_model must be even"
    pe = torch.empty(time_steps, d_model, dtype=torch.float32)
    position = torch.arange(0, time_steps, dtype=torch.float32).unsqueeze(1)            # [T,1]
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                         * -(math.log(base) / d_model))                                  # [D//2]
    angles = position * div_term                                                         # [T, D//2]
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe

def test_positional_encoding_matches_formula_cpu():
    torch.manual_seed(0)
    d_model = 64
    max_len = 128
    batch_size, time_steps = 2, 20  # time_steps <= max_len

    # Your implementation (set wavelength=10000.0 to match the paper/DataCamp)
    pe_layer = PositionalEncoding(wavelength=10000.0, sequence_length=max_len, d_model=d_model)
    pe_layer.eval()

    # If we feed zeros, the output should equal the PE itself (broadcasted over batch)
    x = torch.zeros(batch_size, time_steps, d_model, dtype=torch.float32)
    y = pe_layer(x)  # [B, T, D] = x + PE[:T]

    # Build expected PE for the first T steps
    expected = datacamp_formula_pe(time_steps, d_model)  # [T, D]

    # Compare (broadcast expected over batch)
    assert y.shape == (batch_size, time_steps, d_model)
    # dtype-aware tolerances
    if expected.dtype == torch.float32:
        rtol, atol = 2e-4, 5e-7     # covers your max_rel/max_abs diffs
    else:  # float64 or others
        rtol, atol = 1e-7, 1e-9
    assert_close(y, expected.unsqueeze(0).expand(batch_size, -1, -1), rtol=rtol, atol=atol)