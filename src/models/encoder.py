import torch
import torch.nn as nn

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.blocks import EncoderBlock
from src.layers import PositionalEncoding
# implemtn a Bert style encoder
class Encoder(nn.Module):
    def __init__(self, d_model:int, 
                 n_heads:int, 
                 n_layers:int, 
                 d_ff:int,
                 dropout:float,
                 max_seq_len:int):
        super().__init__()

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.positional_encoding = PositionalEncoding(wavelength=max_seq_len, 
                                                      sequence_length=max_seq_len, 
                                                      d_model=d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x
    
if __name__ == "__main__":
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 256
    dropout = 0.1
    max_seq_len = 10
    encoder = Encoder(d_model, n_heads, n_layers, d_ff, dropout, max_seq_len)
    x = torch.randn(2, max_seq_len, d_model)
    output = encoder(x)
    print(output.shape)


