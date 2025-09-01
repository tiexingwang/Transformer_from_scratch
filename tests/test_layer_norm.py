# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 09:50:27 2025

@author: twang
"""

import torch
import torch.nn as nn
import sys
import pytest
import os
from src.layers import LayerNorm
from torch.testing import assert_close

def test_layer_norm():
    torch.manual_seed(12)
    d_model = 16
    ln_custom = LayerNorm(d_model, eps=1e-5)
    ln_torch  = torch.nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=True)

    # sync parameters
    with torch.no_grad():
        ln_custom.gamma.copy_(ln_torch.weight)
        ln_custom.beta.copy_(ln_torch.bias)

    x = torch.randn(2, 5, d_model)
    y1 = ln_custom(x)
    y2 = ln_torch(x)
    assert_close(y1, y2, rtol=1e-6, atol=1e-7)
    
    assert torch.allclose(y1, y2, rtol=1e-6, atol=1e-7)
    
    