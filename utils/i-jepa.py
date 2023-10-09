import torch

"""
Stochastically Dropping path (pathways) during training, keeping scales of summation of ouptut values
"""

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # print('shape : ', shape)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    # print('random tensor: ', random_tensor)
    output = x.div(keep_prob) * random_tensor
    # print('Random Masking on Path & Scale on the Kept Values to equal original Values')
    return output