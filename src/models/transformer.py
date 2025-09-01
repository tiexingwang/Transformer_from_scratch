import torch
import torch.nn as nn

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.blocks import EncoderBlock, DecoderBlock
from src.layers import PositionalEncoding

# Transformer model with encoder and decoder
# Task for language translation English to Chinese

class Transformer(nn.Module):
    def __init__(self, 
                input_vocab_size:int,
                output_vocab_size:int,
                d_model:int, 
                n_heads:int, 
                n_layers:int, 
                d_ff:int, 
                dropout:float, 
                max_seq_len:int):
        super().__init__()
        # convert vocab size to embedding size
        self.input_embedding = nn.Embedding(input_vocab_size, d_model)
        self.output_embedding = nn.Embedding(output_vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
            ])
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
            ])
        self.positional_encoding = PositionalEncoding(wavelength=max_seq_len, 
                                                      sequence_length=max_seq_len, 
                                                      d_model=d_model)
        self.linear = nn.Linear(d_model, output_vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, src: torch.Tensor, 
                tgt: torch.Tensor, 
                src_mask: torch.Tensor = None, 
                tgt_mask: torch.Tensor = None, 
                src_padding_mask: torch.Tensor = None, 
                tgt_padding_mask: torch.Tensor = None) -> torch.Tensor:

        src = self.input_embedding(src)
        tgt = self.output_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        encoder_output = src
        for encoder_block in self.encoder:
            encoder_output = encoder_block(encoder_output)

        decoder_output = tgt
        for decoder_block in self.decoder:
            decoder_output = decoder_block(decoder_output, encoder_output, tgt_mask)

        output = self.linear(decoder_output)
        return self.softmax(output)

if __name__ == "__main__":
    input_vocab_size = 10000
    output_vocab_size = 10000
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 256
    dropout = 0.1
    max_seq_len = 10

    transformer = Transformer(input_vocab_size, output_vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_seq_len)
    src = torch.randint(0, input_vocab_size, (1, max_seq_len))
    tgt = torch.randint(0, output_vocab_size, (1, max_seq_len))
    output = transformer(src, tgt)
    print(output.shape)