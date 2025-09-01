# tests/test_positionwise_fcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close
from src.layers import PositionwiseFeedForward

def _two_linears(m: nn.Module):
    linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert len(linears) >= 2, "Expected two Linear layers in PositionwiseFCN"
    return linears[0], linears[1]

def test_shape_and_dtype_device_preserved():
    torch.manual_seed(0)
    d_model, d_ff = 64, 256
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0, activation="gelu")
    x = torch.randn(3, 7, d_model, dtype=torch.float32)
    y = ffn(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device

def test_parity_with_closed_form_gelu_no_dropout():
    torch.manual_seed(0)
    d_model = 32
    d_ff = 4 * d_model
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0, activation="gelu").eval()
    x = torch.randn(2, 5, d_model)

    lin1, lin2 = _two_linears(ffn)
    z1 = x @ lin1.weight.T + lin1.bias           # [B,T,d_ff]
    a1 = F.gelu(z1)                               # GELU matches module's activation
    y_ref = a1 @ lin2.weight.T + lin2.bias        # [B,T,d_model]

    y = ffn(x)
    assert_close(y, y_ref, rtol=1e-6, atol=1e-7)

def test_parity_with_relu_when_configured():
    torch.manual_seed(0)
    d_model = 24
    d_ff = 3 * d_model
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0, activation="relu").eval()
    x = torch.randn(4, 3, d_model)

    lin1, lin2 = _two_linears(ffn)
    z1 = x @ lin1.weight.T + lin1.bias
    a1 = F.relu(z1)
    y_ref = a1 @ lin2.weight.T + lin2.bias

    y = ffn(x)
    assert_close(y, y_ref, rtol=1e-6, atol=1e-7)

def test_backward_gradients_flow():
    torch.manual_seed(0)
    d_model = 48
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=4*d_model, dropout=0.0, activation="gelu")
    ffn.train()
    x = torch.randn(2, 6, d_model, requires_grad=True)
    loss = ffn(x).sum()
    loss.backward()

    lin1, lin2 = _two_linears(ffn)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert lin1.weight.grad is not None and torch.isfinite(lin1.weight.grad).all()
    assert lin2.weight.grad is not None and torch.isfinite(lin2.weight.grad).all()

def test_custom_width_affects_parameter_shapes():
    d_model = 40
    d_ff = 5 * d_model + 8
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0, activation="gelu")
    lin1, lin2 = _two_linears(ffn)
    assert lin1.weight.shape == (d_ff, d_model)
    assert lin2.weight.shape == (d_model, d_ff)
