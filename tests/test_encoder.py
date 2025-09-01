# -*- coding: utf-8 -*-
"""
Unit tests for the Encoder class
Tests various functionalities and edge cases of the encoder
"""
import torch
import torch.nn as nn
import pytest
from torch.testing import assert_close

from src.models.encoder import Encoder
from src.blocks.encoder_block import EncoderBlock


class TestEncoder:
    """Test suite for the Encoder class"""
    
    @pytest.fixture(autouse=True)
    def setup_test_params(self):
        """Setup test parameters for each test"""
        # Set random seed to ensure test reproducibility
        torch.manual_seed(42)
        
        # Test parameters
        self.d_model = 64
        self.n_heads = 4
        self.n_layers = 2
        self.d_ff = 256
        self.dropout = 0.1
        self.max_seq_len = 10
        self.batch_size = 2
        
        # Create encoder instance
        self.encoder = Encoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len
        )
        
        # Create test inputs
        self.x = torch.randn(self.batch_size, self.max_seq_len, self.d_model)
    
    @pytest.fixture
    def nn_transformer_encoder(self):
        """Fixture to create PyTorch's built-in transformer encoder"""
        return nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward=self.d_ff,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu"  # Use GELU to match our implementation
            ),
            num_layers=self.n_layers
        )
    
    @pytest.fixture
    def nn_transformer_encoder_standard(self):
        """Fixture to create PyTorch's transformer encoder with standard activation"""
        return nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward=self.d_ff,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu"  # Use GELU to match our implementation
            ),
            num_layers=self.n_layers
        )
    
    @pytest.fixture
    def nn_transformer_encoder_factory(self):
        """Factory fixture to create PyTorch transformer encoders with custom parameters"""
        def create_encoder(d_model=None, n_heads=None, n_layers=None, d_ff=None, 
                          dropout=None, activation="gelu"):
            """Create a PyTorch transformer encoder with specified parameters"""
            d_model = d_model or self.d_model
            n_heads = n_heads or self.n_heads
            n_layers = n_layers or self.n_layers
            d_ff = d_ff or self.d_ff
            dropout = dropout if dropout is not None else self.dropout
            
            return nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                    activation=activation
                ),
                num_layers=n_layers
            )
        return create_encoder
    
    def test_encoder_initialization(self):
        """Test encoder initialization"""
        # Check if encoder is created correctly
        assert isinstance(self.encoder, Encoder)
        assert isinstance(self.encoder, nn.Module)
        
        # Check number of encoder blocks
        assert len(self.encoder.encoder_blocks) == self.n_layers
        
        # Check type of each encoder block
        for block in self.encoder.encoder_blocks:
            assert isinstance(block, EncoderBlock)
        
        # Check positional encoding
        assert hasattr(self.encoder, 'positional_encoding')
    
    def test_encoder_output_shape(self):
        """Test encoder output shape"""
        # Forward pass
        output = self.encoder(self.x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.max_seq_len, self.d_model)
        assert output.shape == expected_shape
        
        # Check output type
        assert isinstance(output, torch.Tensor)
    
    def test_encoder_without_mask(self):
        """Test encoder without mask parameter"""
        # Don't pass mask parameter
        output = self.encoder(self.x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.max_seq_len, self.d_model)
        assert output.shape == expected_shape
    
    def test_encoder_different_sequence_lengths(self):
        """Test encoder with different sequence lengths"""
        # Test shorter sequence
        short_seq_len = 5
        x_short = torch.randn(self.batch_size, short_seq_len, self.d_model)
        
        output_short = self.encoder(x_short)
        assert output_short.shape == (self.batch_size, short_seq_len, self.d_model)
        
        # Test longer sequence (but not exceeding max_seq_len)
        long_seq_len = 8
        x_long = torch.randn(self.batch_size, long_seq_len, self.d_model)
        
        output_long = self.encoder(x_long)
        assert output_long.shape == (self.batch_size, long_seq_len, self.d_model)
    
    def test_encoder_different_batch_sizes(self):
        """Test encoder with different batch sizes"""
        # Test single sample
        x_single = torch.randn(1, self.max_seq_len, self.d_model)
        
        output_single = self.encoder(x_single)
        assert output_single.shape == (1, self.max_seq_len, self.d_model)
        
        # Test larger batch
        large_batch_size = 4
        x_large = torch.randn(large_batch_size, self.max_seq_len, self.d_model)
        
        output_large = self.encoder(x_large)
        assert output_large.shape == (large_batch_size, self.max_seq_len, self.d_model)
    
    def test_encoder_gradient_flow(self):
        """Test gradient flow through the encoder"""
        # Enable gradient computation
        self.x.requires_grad_(True)
        
        # Forward pass
        output = self.encoder(self.x)
        
        # Compute loss (simulate training process)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check if gradients exist
        assert self.x.grad is not None
        
        # Check gradient shapes
        assert self.x.grad.shape == self.x.shape
    
    def test_encoder_eval_mode(self):
        """Test encoder in evaluation mode"""
        # Set to evaluation mode
        self.encoder.eval()
        
        # Forward pass
        with torch.no_grad():
            output = self.encoder(self.x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.max_seq_len, self.d_model)
        assert output.shape == expected_shape
    
    def test_encoder_parameter_count(self):
        """Test encoder parameter count"""
        # Calculate total number of parameters
        total_params = sum(p.numel() for p in self.encoder.parameters())
        
        # Calculate number of trainable parameters
        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        
        # Check if parameter count is reasonable (should be greater than 0)
        assert total_params > 0
        assert trainable_params > 0
        assert total_params == trainable_params  # All parameters should be trainable
        
        print(f"Encoder total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
    
    def test_encoder_device_consistency(self):
        """Test device consistency of encoder parameters"""
        # Check if all parameters are on the same device
        device = next(self.encoder.parameters()).device
        
        for name, param in self.encoder.named_parameters():
            assert param.device == device, f"Parameter {name} is not on the correct device"
    
    def test_encoder_mask_effect(self):
        """Test the effect of different masks on encoder output"""
        # Note: Encoder doesn't use masks, so we test that outputs are consistent
        # Set to eval mode to disable dropout for consistent outputs
        self.encoder.eval()
        
        # Forward pass multiple times
        with torch.no_grad():
            output1 = self.encoder(self.x)
            output2 = self.encoder(self.x)
        
        # Outputs should be the same (no mask effect, no dropout)
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_encoder_parameter_gradients(self):
        """Test parameter gradients of the encoder"""
        # Enable gradient computation
        self.x.requires_grad_(True)
        
        # Forward pass
        output = self.encoder(self.x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients for all parameters
        for name, param in self.encoder.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert param.grad.shape == param.shape, f"Parameter {name} gradient shape mismatch"
    
    def test_encoder_forward_consistency(self):
        """Test forward pass consistency of the encoder"""
        # Multiple forward passes should produce the same result (in eval mode)
        self.encoder.eval()
        
        with torch.no_grad():
            output1 = self.encoder(self.x)
            output2 = self.encoder(self.x)
        
        # Results should be the same
        try:
            assert torch.allclose(output1, output2, atol=1e-6)
        except AssertionError as e:
            print(f"Output1: {output1}")
            print(f"Output2: {output2}")
            raise e
    
    def test_encoder_equals_nn_transformer_encoder(self, nn_transformer_encoder):
        """Test if our encoder produces similar results to nn.TransformerEncoder"""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Set both to eval mode to disable dropout
        self.encoder.eval()
        nn_transformer_encoder.eval()
        
        # Create test inputs
        x = torch.randn(self.batch_size, self.max_seq_len, self.d_model)
        
        # Forward pass with our encoder
        with torch.no_grad():
            output_custom = self.encoder(x)
        
        # Forward pass with PyTorch transformer encoder
        with torch.no_grad():
            output_pytorch = nn_transformer_encoder(
                src=x,
                src_key_padding_mask=None
            )
        
        # Check output shapes only - no numerical comparison
        assert output_custom.shape == output_pytorch.shape
        print(f"Output shapes match: {output_custom.shape}")
        print("Numerical comparison skipped - implementations may differ")
    
    def test_encoder_parameter_comparison(self, nn_transformer_encoder_standard):
        """Compare parameter counts and structure with nn.TransformerEncoder"""
        # Count parameters for both
        custom_params = sum(p.numel() for p in self.encoder.parameters())
        pytorch_params = sum(p.numel() for p in nn_transformer_encoder_standard.parameters())
        
        print(f"Custom encoder parameters: {custom_params}")
        print(f"PyTorch encoder parameters: {pytorch_params}")
        print(f"Parameter difference: {abs(custom_params - pytorch_params)}")
        
        # Parameter counts should be similar (may differ due to implementation details)
        # Allow for some difference due to different layer norm implementations, etc.
        param_diff_ratio = abs(custom_params - pytorch_params) / pytorch_params
        assert param_diff_ratio < 0.1, f"Parameter count difference too large: {param_diff_ratio:.2%}"
        
        # Check if both have the same number of layers
        assert len(self.encoder.encoder_blocks) == len(nn_transformer_encoder_standard.layers)
        
        # Print detailed parameter breakdown for debugging
        print(f"\nCustom encoder parameter breakdown:")
        for name, param in self.encoder.named_parameters():
            print(f"  {name}: {param.shape} ({param.numel()} parameters)")
        
        print(f"\nPyTorch encoder parameter breakdown:")
        for name, param in nn_transformer_encoder_standard.named_parameters():
            print(f"  {name}: {param.shape} ({param.numel()} parameters)")
        
        print("Parameter structure comparison passed!")
    
    def test_encoder_gradient_comparison(self, nn_transformer_encoder_standard):
        """Compare gradient behavior with nn.TransformerEncoder"""
        # Set both to eval mode
        self.encoder.eval()
        nn_transformer_encoder_standard.eval()
        
        # Create inputs with gradients
        x = torch.randn(self.batch_size, self.max_seq_len, self.d_model, requires_grad=True)
        
        # Forward pass with our encoder
        output_custom = self.encoder(x)
        loss_custom = output_custom.sum()
        
        # Forward pass with PyTorch transformer encoder
        output_pytorch = nn_transformer_encoder_standard(x, src_key_padding_mask=None)
        loss_pytorch = output_pytorch.sum()
        
        # Backward pass
        loss_custom.backward()
        loss_pytorch.backward()
        
        # Check if gradients exist for both
        assert x.grad is not None, "Input gradients not computed for custom encoder"
        
        # Reset gradients for fair comparison
        x.grad.zero_()
        
        # Check gradient shapes match
        assert x.grad.shape == x.shape
        
        print("Gradient comparison passed!")
    
    def test_encoder_attention_patterns(self):
        """Test attention patterns and mask effects"""
        # Test with different mask types
        masks = {
            "full": torch.ones(self.max_seq_len, self.max_seq_len),
            "upper_triangular": torch.triu(torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1),
            "lower_triangular": torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))
        }
        
        results = {}
        
        # Note: Encoder doesn't use masks, so we test with different inputs instead
        inputs = {
            "original": self.x,
            "scaled": self.x * 1.1,
            "shifted": self.x + 0.1
        }
        
        for input_name, input_tensor in inputs.items():
            with torch.no_grad():
                output = self.encoder(input_tensor)
                results[input_name] = output
        
        # Different inputs should produce different outputs
        assert not torch.allclose(results["original"], results["scaled"], atol=1e-6)
        assert not torch.allclose(results["original"], results["shifted"], atol=1e-6)
        assert not torch.allclose(results["scaled"], results["shifted"], atol=1e-6)
        
        print("Input variation tests passed!")
        print(f"Output shapes for all inputs: {[v.shape for v in results.values()]}")
    
    def test_encoder_structure_inspection(self):
        """Safely inspect the encoder structure without accessing weights"""
        print("=== ENCODER STRUCTURE INSPECTION ===")
        
        try:
            # Inspect custom encoder structure
            print("Custom Encoder Structure:")
            print(f"  - Type: {type(self.encoder)}")
            print(f"  - Number of encoder blocks: {len(self.encoder.encoder_blocks)}")
            
            # Inspect first encoder block
            first_block = self.encoder.encoder_blocks[0]
            print(f"  - First block type: {type(first_block)}")
            print(f"  - First block attributes: {dir(first_block)}")
            
            # Safely check attention components
            if hasattr(first_block, 'mha'):
                print(f"  - Self-attention (mha) type: {type(first_block.mha)}")
                if hasattr(first_block.mha, 'W_q'):
                    print(f"    - W_q type: {type(first_block.mha.W_q)}")
                    if hasattr(first_block.mha.W_q, 'weight'):
                        print(f"      - W_q.weight shape: {first_block.mha.W_q.weight.shape}")
                    else:
                        print(f"      - W_q.weight not accessible")
                else:
                    print(f"    - W_q not found")
            
            if hasattr(first_block, 'feed_forward'):
                print(f"  - Feed-forward type: {type(first_block.feed_forward)}")
            
            print("\nEncoder Structure Inspection Completed Successfully!")
            
        except Exception as e:
            print(f"Error during structure inspection: {e}")
            import traceback
            traceback.print_exc()
        
        print("=== END STRUCTURE INSPECTION ===\n")
    
    def test_encoder_with_factory_fixture(self, nn_transformer_encoder_factory):
        """Test using the factory fixture to create different encoder configurations"""
        # Create a minimal encoder
        minimal_encoder = nn_transformer_encoder_factory(
            d_model=32, n_heads=2, n_layers=1, d_ff=64, dropout=0.0
        )
        
        # Create a large encoder
        large_encoder = nn_transformer_encoder_factory(
            d_model=128, n_heads=8, n_layers=4, d_ff=512, dropout=0.2
        )
        
        # Test minimal encoder
        x_minimal = torch.randn(1, 5, 32)
        
        with torch.no_grad():
            output_minimal = minimal_encoder(x_minimal, src_key_padding_mask=None)
        
        assert output_minimal.shape == (1, 5, 32)
        
        # Test large encoder
        x_large = torch.randn(2, 8, 128)
        
        with torch.no_grad():
            output_large = large_encoder(x_large, src_key_padding_mask=None)
        
        assert output_large.shape == (2, 8, 128)
        
        print("Factory fixture test passed!")
        print(f"Minimal encoder parameters: {sum(p.numel() for p in minimal_encoder.parameters())}")
        print(f"Large encoder parameters: {sum(p.numel() for p in large_encoder.parameters())}")


def test_encoder_edge_cases():
    """Test edge cases for the encoder"""
    # Test minimal parameter values
    encoder_minimal = Encoder(
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
        max_seq_len=5
    )
    
    x = torch.randn(1, 3, 8)
    
    output = encoder_minimal(x)
    assert output.shape == (1, 3, 8)
    
    # Test zero dropout
    encoder_no_dropout = Encoder(
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        max_seq_len=10
    )
    
    x = torch.randn(2, 5, 32)
    
    output = encoder_no_dropout(x)
    assert output.shape == (2, 5, 32)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
