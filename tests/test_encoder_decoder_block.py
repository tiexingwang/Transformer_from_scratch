# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 15:05:36 2025

@author: twang
"""
import torch
from src.blocks.encoder_block import EncoderBlock
from src.blocks.decoder_block import DecoderBlock
import torch.nn as nn
from torch.testing import assert_close

def test_encoder_block():
    encoder_block = EncoderBlock(d_model=64, num_heads=4, d_ff=256, dropout=0.1)
    x = torch.randn(2, 10, 64)
    output = encoder_block(x)
    assert output.shape == (2, 10, 64)
    assert output.is_cuda == x.is_cuda

def test_encoder_block_equals_nn_transformer_encoder_layer():
    torch.manual_seed(0)
    d_model = 64
    num_heads = 4
    d_ff = 256
    dropout = 0.1
    batch_size = 2
    seq_len = 10
    
    encoder_block = EncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
    nn_transformer_encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, 
        nhead=num_heads, 
        dim_feedforward=d_ff, 
        dropout=dropout,
        batch_first=True,
        activation="gelu"  # Changed from default "relu" to "gelu"
    )

    # Check activation functions
    print("=== ACTIVATION FUNCTION CHECK ===")
    print(f"Custom activation: {type(encoder_block.feed_forward.activation).__name__}")
    print(f"PyTorch activation: {type(nn_transformer_encoder_layer.activation).__name__}")
    print("=== END ACTIVATION CHECK ===\n")

    # Sync parameters for Multi-Head Attention
    with torch.no_grad():
        # Query, Key, Value weights and biases
        encoder_block.mha.W_q.weight.copy_(nn_transformer_encoder_layer.self_attn.in_proj_weight[0:d_model, :])
        encoder_block.mha.W_q.bias.copy_(nn_transformer_encoder_layer.self_attn.in_proj_bias[0:d_model])
        encoder_block.mha.W_k.weight.copy_(nn_transformer_encoder_layer.self_attn.in_proj_weight[d_model:2*d_model, :])
        encoder_block.mha.W_k.bias.copy_(nn_transformer_encoder_layer.self_attn.in_proj_bias[d_model:2*d_model])
        encoder_block.mha.W_v.weight.copy_(nn_transformer_encoder_layer.self_attn.in_proj_weight[2*d_model:3*d_model, :])
        encoder_block.mha.W_v.bias.copy_(nn_transformer_encoder_layer.self_attn.in_proj_bias[2*d_model:3*d_model])
        encoder_block.mha.W_o.weight.copy_(nn_transformer_encoder_layer.self_attn.out_proj.weight)
        encoder_block.mha.W_o.bias.copy_(nn_transformer_encoder_layer.self_attn.out_proj.bias)
        
        # Feed-forward network
        encoder_block.feed_forward.fcn1.weight.copy_(nn_transformer_encoder_layer.linear1.weight)
        encoder_block.feed_forward.fcn1.bias.copy_(nn_transformer_encoder_layer.linear1.bias)
        encoder_block.feed_forward.fcn2.weight.copy_(nn_transformer_encoder_layer.linear2.weight)
        encoder_block.feed_forward.fcn2.bias.copy_(nn_transformer_encoder_layer.linear2.bias)
        
        # Layer normalization
        encoder_block.layer_norm1.gamma.copy_(nn_transformer_encoder_layer.norm1.weight)
        encoder_block.layer_norm1.beta.copy_(nn_transformer_encoder_layer.norm1.bias)
        encoder_block.layer_norm2.gamma.copy_(nn_transformer_encoder_layer.norm2.weight)
        encoder_block.layer_norm2.beta.copy_(nn_transformer_encoder_layer.norm2.bias)

    # Sanity check - verify all parameters are properly synced
    assert_close(encoder_block.mha.W_q.weight, nn_transformer_encoder_layer.self_attn.in_proj_weight[0:d_model, :])
    assert_close(encoder_block.mha.W_q.bias, nn_transformer_encoder_layer.self_attn.in_proj_bias[0:d_model])
    assert_close(encoder_block.mha.W_k.weight, nn_transformer_encoder_layer.self_attn.in_proj_weight[d_model:2*d_model, :])
    assert_close(encoder_block.mha.W_k.bias, nn_transformer_encoder_layer.self_attn.in_proj_bias[d_model:2*d_model])
    assert_close(encoder_block.mha.W_v.weight, nn_transformer_encoder_layer.self_attn.in_proj_weight[2*d_model:3*d_model, :])
    assert_close(encoder_block.mha.W_o.weight, nn_transformer_encoder_layer.self_attn.out_proj.weight)
    assert_close(encoder_block.mha.W_o.bias, nn_transformer_encoder_layer.self_attn.out_proj.bias)
    assert_close(encoder_block.feed_forward.fcn1.weight, nn_transformer_encoder_layer.linear1.weight)
    assert_close(encoder_block.feed_forward.fcn1.bias, nn_transformer_encoder_layer.linear1.bias)
    assert_close(encoder_block.feed_forward.fcn2.weight, nn_transformer_encoder_layer.linear2.weight)
    assert_close(encoder_block.feed_forward.fcn2.bias, nn_transformer_encoder_layer.linear2.bias)
    assert_close(encoder_block.layer_norm1.gamma, nn_transformer_encoder_layer.norm1.weight)
    assert_close(encoder_block.layer_norm1.beta, nn_transformer_encoder_layer.norm1.bias)
    assert_close(encoder_block.layer_norm2.gamma, nn_transformer_encoder_layer.norm2.weight)
    assert_close(encoder_block.layer_norm2.beta, nn_transformer_encoder_layer.norm2.bias)

    # Test with same input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Set both to eval mode to disable dropout
    encoder_block.eval()
    nn_transformer_encoder_layer.eval()
    
    # Step-by-step comparison
    print("=== STEP-BY-STEP COMPARISON ===")
    
    # Step 1: Multi-head attention
    attn_custom = encoder_block.mha(x, x, x)
    attn_torch = nn_transformer_encoder_layer.self_attn(x, x, x)[0]  # [0] to get output, not attention weights
    print(f"Attention output diff: {(attn_custom - attn_torch).abs().max().item()}")
    
    # Step 2: First residual + norm
    residual1_custom = x + attn_custom
    residual1_torch = x + attn_torch
    print(f"First residual diff: {(residual1_custom - residual1_torch).abs().max().item()}")
    
    norm1_custom = encoder_block.layer_norm1(residual1_custom)
    norm1_torch = nn_transformer_encoder_layer.norm1(residual1_torch)
    print(f"First norm diff: {(norm1_custom - norm1_torch).abs().max().item()}")
    
    # Step 3: Feed-forward - DETAILED BREAKDOWN
    print("\n=== FEED-FORWARD DETAILED BREAKDOWN ===")
    
    # First linear layer
    linear1_custom = encoder_block.feed_forward.fcn1(norm1_custom)
    linear1_torch = nn_transformer_encoder_layer.linear1(norm1_torch)
    print(f"Linear1 output diff: {(linear1_custom - linear1_torch).abs().max().item()}")
    
    # Activation function
    activation_custom = encoder_block.feed_forward.activation(linear1_custom)
    activation_torch = nn_transformer_encoder_layer.activation(linear1_torch)
    print(f"Activation output diff: {(activation_custom - activation_torch).abs().max().item()}")
    
    # First dropout
    dropout1_custom = encoder_block.feed_forward.dropout1(activation_custom)
    dropout1_torch = nn_transformer_encoder_layer.dropout(activation_torch)
    print(f"Dropout1 output diff: {(dropout1_custom - dropout1_torch).abs().max().item()}")
    
    # Second linear layer
    linear2_custom = encoder_block.feed_forward.fcn2(dropout1_custom)
    linear2_torch = nn_transformer_encoder_layer.linear2(dropout1_torch)
    print(f"Linear2 output diff: {(linear2_custom - linear2_torch).abs().max().item()}")
    
    # Final dropout
    dropout2_custom = encoder_block.feed_forward.dropout2(linear2_custom)
    dropout2_torch = nn_transformer_encoder_layer.dropout(linear2_torch)
    print(f"Dropout2 output diff: {(dropout2_custom - dropout2_torch).abs().max().item()}")
    
    # Complete feed-forward
    ff_custom = encoder_block.feed_forward(norm1_custom)
    ff_torch = nn_transformer_encoder_layer.linear2(
        nn_transformer_encoder_layer.dropout(
            nn_transformer_encoder_layer.activation(
                nn_transformer_encoder_layer.linear1(norm1_torch)
            )
        )
    )
    print(f"Complete feed-forward diff: {(ff_custom - ff_torch).abs().max().item()}")
    print("=== END FEED-FORWARD BREAKDOWN ===\n")
    
    # Step 4: Second residual + norm
    residual2_custom = norm1_custom + ff_custom
    residual2_torch = norm1_torch + ff_torch
    print(f"Second residual diff: {(residual2_custom - residual2_torch).abs().max().item()}")
    
    norm2_custom = encoder_block.layer_norm2(residual2_custom)
    norm2_torch = nn_transformer_encoder_layer.norm2(residual2_torch)
    print(f"Second norm diff: {(norm2_custom - norm2_torch).abs().max().item()}")
    
    # Final outputs
    output = encoder_block(x)
    nn_output = nn_transformer_encoder_layer(x)
    
    print(f"Final output diff: {(output - nn_output).abs().max().item()}")
    print("=== END COMPARISON ===")
    
    # Use more lenient tolerances for comparison
    try:
        assert_close(output, nn_output, rtol=1e-3, atol=1e-5)
    except AssertionError as e:
        # Print diagnostic information
        diff = (output - nn_output).abs()
        print(f"Max absolute difference: {diff.max().item()}")
        print(f"Max relative difference: {(diff / nn_output.abs().clamp_min(1e-8)).max().item()}")
        print(f"Output shapes: {output.shape} vs {output.shape}")
        print(f"Output dtypes: {output.dtype} vs {nn_output.dtype}")
        raise e
    
    assert output.is_cuda == x.is_cuda

def test_decoder_block():
    decoder_block = DecoderBlock(d_model=64, num_heads=4, d_ff=256, dropout=0.1)
    x = torch.randn(2, 10, 64)
    encoder_output = torch.randn(2, 10, 64)
    decoder_mask = torch.ones(2, 10, 10)
    output = decoder_block(x, encoder_output, decoder_mask)
    assert output.shape == (2, 10, 64)
    assert output.is_cuda == x.is_cuda

def test_decoder_block_equals_nn_transformer_decoder_layer():
    torch.manual_seed(0)
    d_model = 64
    num_heads = 4
    d_ff = 256
    dropout = 0.1
    batch_size = 2
    seq_len = 10
    
    decoder_block = DecoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
    nn_transformer_decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model, 
        nhead=num_heads, 
        dim_feedforward=d_ff, 
        dropout=dropout,
        batch_first=True,
        activation="gelu"  # Changed from default "relu" to "gelu"
    )

    # sync parameters
    with torch.no_grad():
        decoder_block.mha.W_q.weight.copy_(nn_transformer_decoder_layer.self_attn.in_proj_weight[0:d_model, :])
        decoder_block.mha.W_q.bias.copy_(nn_transformer_decoder_layer.self_attn.in_proj_bias[0:d_model])
        decoder_block.mha.W_k.weight.copy_(nn_transformer_decoder_layer.self_attn.in_proj_weight[d_model:2*d_model, :])
        decoder_block.mha.W_k.bias.copy_(nn_transformer_decoder_layer.self_attn.in_proj_bias[d_model:2*d_model])
        decoder_block.mha.W_v.weight.copy_(nn_transformer_decoder_layer.self_attn.in_proj_weight[2*d_model:3*d_model, :])
        decoder_block.mha.W_v.bias.copy_(nn_transformer_decoder_layer.self_attn.in_proj_bias[2*d_model:3*d_model])
        decoder_block.mha.W_o.weight.copy_(nn_transformer_decoder_layer.self_attn.out_proj.weight)
        decoder_block.mha.W_o.bias.copy_(nn_transformer_decoder_layer.self_attn.out_proj.bias)
        
        # Cross-attention (encoder-decoder attention)
        decoder_block.mha_cross.W_q.weight.copy_(nn_transformer_decoder_layer.multihead_attn.in_proj_weight[0:d_model, :])
        decoder_block.mha_cross.W_q.bias.copy_(nn_transformer_decoder_layer.multihead_attn.in_proj_bias[0:d_model])
        decoder_block.mha_cross.W_k.weight.copy_(nn_transformer_decoder_layer.multihead_attn.in_proj_weight[d_model:2*d_model, :])
        decoder_block.mha_cross.W_k.bias.copy_(nn_transformer_decoder_layer.multihead_attn.in_proj_bias[d_model:2*d_model])
        decoder_block.mha_cross.W_v.weight.copy_(nn_transformer_decoder_layer.multihead_attn.in_proj_weight[2*d_model:3*d_model, :])
        decoder_block.mha_cross.W_v.bias.copy_(nn_transformer_decoder_layer.multihead_attn.in_proj_bias[2*d_model:3*d_model])
        decoder_block.mha_cross.W_o.weight.copy_(nn_transformer_decoder_layer.multihead_attn.out_proj.weight)
        decoder_block.mha_cross.W_o.bias.copy_(nn_transformer_decoder_layer.multihead_attn.out_proj.bias)   
        
        # Layer normalization
        decoder_block.layer_norm1.gamma.copy_(nn_transformer_decoder_layer.norm1.weight)
        decoder_block.layer_norm1.beta.copy_(nn_transformer_decoder_layer.norm1.bias)
        decoder_block.layer_norm2.gamma.copy_(nn_transformer_decoder_layer.norm2.weight)
        decoder_block.layer_norm2.beta.copy_(nn_transformer_decoder_layer.norm2.bias)
        decoder_block.layer_norm3.gamma.copy_(nn_transformer_decoder_layer.norm3.weight)
        decoder_block.layer_norm3.beta.copy_(nn_transformer_decoder_layer.norm3.bias)
        
        # Feed-forward network
        decoder_block.feed_forward.fcn1.weight.copy_(nn_transformer_decoder_layer.linear1.weight)
        decoder_block.feed_forward.fcn1.bias.copy_(nn_transformer_decoder_layer.linear1.bias)
        decoder_block.feed_forward.fcn2.weight.copy_(nn_transformer_decoder_layer.linear2.weight)
        decoder_block.feed_forward.fcn2.bias.copy_(nn_transformer_decoder_layer.linear2.bias)

    # Sanity check - verify all parameters are properly synced
    assert_close(decoder_block.mha.W_q.weight, nn_transformer_decoder_layer.self_attn.in_proj_weight[0:d_model, :])
    assert_close(decoder_block.mha.W_q.bias, nn_transformer_decoder_layer.self_attn.in_proj_bias[0:d_model])
    assert_close(decoder_block.mha.W_k.weight, nn_transformer_decoder_layer.self_attn.in_proj_weight[d_model:2*d_model, :])
    assert_close(decoder_block.mha.W_k.bias, nn_transformer_decoder_layer.self_attn.in_proj_bias[d_model:2*d_model])
    assert_close(decoder_block.mha.W_v.weight, nn_transformer_decoder_layer.self_attn.in_proj_weight[2*d_model:3*d_model, :])
    assert_close(decoder_block.mha.W_o.weight, nn_transformer_decoder_layer.self_attn.out_proj.weight)
    assert_close(decoder_block.mha.W_o.bias, nn_transformer_decoder_layer.self_attn.out_proj.bias)
    assert_close(decoder_block.mha_cross.W_q.weight, nn_transformer_decoder_layer.multihead_attn.in_proj_weight[0:d_model, :])
    assert_close(decoder_block.mha_cross.W_q.bias, nn_transformer_decoder_layer.multihead_attn.in_proj_bias[0:d_model])
    assert_close(decoder_block.mha_cross.W_k.weight, nn_transformer_decoder_layer.multihead_attn.in_proj_weight[d_model:2*d_model, :])
    assert_close(decoder_block.mha_cross.W_k.bias, nn_transformer_decoder_layer.multihead_attn.in_proj_bias[d_model:2*d_model])
    assert_close(decoder_block.mha_cross.W_v.weight, nn_transformer_decoder_layer.multihead_attn.in_proj_weight[2*d_model:3*d_model, :])
    assert_close(decoder_block.mha_cross.W_v.bias, nn_transformer_decoder_layer.multihead_attn.in_proj_bias[d_model:2*d_model])
    assert_close(decoder_block.mha_cross.W_o.weight, nn_transformer_decoder_layer.multihead_attn.out_proj.weight)
    assert_close(decoder_block.mha_cross.W_o.bias, nn_transformer_decoder_layer.multihead_attn.out_proj.bias)
    assert_close(decoder_block.layer_norm1.gamma, nn_transformer_decoder_layer.norm1.weight)
    assert_close(decoder_block.layer_norm1.beta, nn_transformer_decoder_layer.norm1.bias)
    assert_close(decoder_block.layer_norm2.gamma, nn_transformer_decoder_layer.norm2.weight)
    assert_close(decoder_block.layer_norm2.beta, nn_transformer_decoder_layer.norm2.bias)
    assert_close(decoder_block.layer_norm3.gamma, nn_transformer_decoder_layer.norm3.weight)
    assert_close(decoder_block.layer_norm3.beta, nn_transformer_decoder_layer.norm3.bias)
    assert_close(decoder_block.feed_forward.fcn1.weight, nn_transformer_decoder_layer.linear1.weight)
    assert_close(decoder_block.feed_forward.fcn1.bias, nn_transformer_decoder_layer.linear1.bias)
    assert_close(decoder_block.feed_forward.fcn2.weight, nn_transformer_decoder_layer.linear2.weight)
    assert_close(decoder_block.feed_forward.fcn2.bias, nn_transformer_decoder_layer.linear2.bias)

    # Test with same input
    x = torch.randn(batch_size, seq_len, d_model)
    encoder_output = torch.randn(batch_size, seq_len, d_model)
    
    # Create mask in PyTorch's expected format: [batch_size * num_heads, seq_len, seq_len]
    # PyTorch expects the mask to be flattened across attention heads
    decoder_mask = torch.ones(batch_size * num_heads, seq_len, seq_len)
    
    # Set both to eval mode to disable dropout
    decoder_block.eval()
    nn_transformer_decoder_layer.eval()

    output = decoder_block(x, encoder_output, decoder_mask)
    nn_output = nn_transformer_decoder_layer(x, encoder_output, decoder_mask)

    # assert_close
    assert_close(output, nn_output)
    assert output.is_cuda == x.is_cuda
    
    # Use more lenient tolerances for comparison
    try:
        assert_close(output, nn_output, rtol=1e-3, atol=1e-5)
    except AssertionError as e:
        # Print diagnostic information
        diff = (output - nn_output).abs()
        print(f"Max absolute difference: {diff.max().item()}")
        print(f"Max relative difference: {(diff / nn_output.abs().clamp_min(1e-8)).max().item()}")
        print(f"Output shapes: {output.shape} vs {nn_output.shape}")
        print(f"Output dtypes: {output.dtype} vs {nn_output.dtype}")
        raise e
    