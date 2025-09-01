import torch
import torch.nn as nn

def test_copy_weights():
    in_feature = 10
    out_feature = 5
    src = nn.Linear(in_features=in_feature, out_features=out_feature)
    target = nn.Linear(in_features=in_feature, out_features=out_feature)
    
    with torch.no_grad():
        target.weight.copy_(src.weight)
        target.bias.copy_(src.bias)
        
    assert torch.allclose(target.weight, src.weight)
    assert torch.allclose(target.bias, src.bias)

