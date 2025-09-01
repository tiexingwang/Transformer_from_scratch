# tests/test_multi_head_attention.py compared to src/layers/attention.py
import torch
import torch.nn as nn
import pytest
import sys
import os
from src.layers import MultiHeadAttention
from torch.testing import assert_close

def sync_custom_from_torch(
    custom_mha: MultiHeadAttention,
    torch_mha: nn.MultiheadAttention,
) -> None:
    """
    Copy parameters from nn.MultiheadAttention to our custom layer so outputs can be compared 1:1.
    Assumes torch_mha has batch_first=True and default packed in-proj weights/bias.
    """
    d_model = custom_mha.d_model
    with torch.no_grad():
        # nn.MultiheadAttention packs Q/K/V into one matrix of shape [3*d_model, d_model]
        inW = torch_mha.in_proj_weight.detach()
        inb = torch_mha.in_proj_bias.detach()
        custom_mha.W_q.weight.copy_(inW[0:d_model, :]);        custom_mha.W_q.bias.copy_(inb[0:d_model])
        custom_mha.W_k.weight.copy_(inW[d_model:2*d_model, :]); custom_mha.W_k.bias.copy_(inb[d_model:2*d_model])
        custom_mha.W_v.weight.copy_(inW[2*d_model:3*d_model, :]); custom_mha.W_v.bias.copy_(inb[2*d_model:3*d_model])

        # Out projection
        custom_mha.W_o.weight.copy_(torch_mha.out_proj.weight.detach())
        custom_mha.W_o.bias.copy_(torch_mha.out_proj.bias.detach())

    # Sanity: verify parameters truly match
    assert_close(custom_mha.W_q.weight, inW[0:d_model, :])
    assert_close(custom_mha.W_k.weight, inW[d_model:2*d_model, :])
    assert_close(custom_mha.W_v.weight, inW[2*d_model:3*d_model, :])
    assert_close(custom_mha.W_q.bias,   inb[0:d_model])
    assert_close(custom_mha.W_k.bias,   inb[d_model:2*d_model])
    assert_close(custom_mha.W_v.bias,   inb[2*d_model:3*d_model])
    assert_close(custom_mha.W_o.weight, torch_mha.out_proj.weight)
    assert_close(custom_mha.W_o.bias,   torch_mha.out_proj.bias)


def test_split_concat_roundtrip():
    d_model = 64
    num_heads = 4
    batch_size = 3
    time_steps = 7

    x = torch.randn(batch_size, time_steps, d_model)
    mha = MultiHeadAttention(d_model, num_heads)
    y = mha.concat_heads(mha.split_heads(x))

    assert y.shape == x.shape
    assert_close(y, x)  # exact equality for view/permute paths in fp32

def test_matches_torch_mha_when_weights_synced():
    d_model = 64
    num_heads = 4
    batch_size = 2
    time_steps = 10

    queries = torch.rand(batch_size, time_steps, d_model)
    keys    = torch.rand(batch_size, time_steps, d_model)
    values  = torch.rand(batch_size, time_steps, d_model)

    custom = MultiHeadAttention(d_model, num_heads)
    torch_mha = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=0.0
    )

    # put both in eval just to be explicit (dropout=0 already)
    custom.eval()
    torch_mha.eval()

    # synchronize parameters
    sync_custom_from_torch(custom, torch_mha)

    out_custom = custom(queries, keys, values)      # [B,T,D]
    out_torch, _ = torch_mha(queries, keys, values) # [B,T,D]
    
    # dtype-aware tolerances
    if out_custom.dtype == torch.float32:
        rtol, atol = 2e-4, 5e-7     # covers your max_rel/max_abs diffs
    else:  # float64 or others
        rtol, atol = 1e-7, 1e-9

    # helpful diagnostics if this fails on some platforms
    try:
        assert_close(out_custom, out_torch, rtol=rtol, atol=atol)
    except AssertionError as e:
        diff = (out_custom - out_torch).detach()
        print("max_abs_diff:", diff.abs().max().item())
        rel = diff.abs() / out_torch.abs().clamp_min(1e-12)
        print("max_rel_diff:", rel.max().item())
        # You can relax to rtol=1e-4 if needed due to kernel/BLAS variance
        raise e

def test_mask_broadcast_and_effect():
    d_model = 32
    num_heads = 4
    batch_size = 2
    time_steps = 5

    x = torch.randn(batch_size, time_steps, d_model)
    custom = MultiHeadAttention(d_model, num_heads)

    # Build a mask that disables the last key timestep across the board
    # mask True/1=keep, False/0=mask. Shape broadcastable to [B,H,Tq,Tk]
    key_keep = torch.ones(batch_size, time_steps, dtype=torch.bool)
    key_keep[:, -1] = False
    mask = key_keep[:, None, None, :]  # [B,1,1,Tk]

    out_masked   = custom(x, x, x, mask=mask)
    out_unmasked = custom(x, x, x, mask=None)

    assert out_masked.shape == out_unmasked.shape
    assert not torch.allclose(out_masked, out_unmasked)


def test_raises_when_d_model_not_divisible_by_heads():
    with pytest.raises(AssertionError):
        _ = MultiHeadAttention(d_model=65, num_heads=4)




