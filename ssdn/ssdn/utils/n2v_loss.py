import torch
from torch import Tensor
import torch.nn as nn


def loss_mask_mse(
    masked_coords: Tensor,
    input: Tensor,
    target: Tensor,
    reduction: bool = 'none'
):
    mse = 0
    coords = masked_coords.tolist()[0]
    for coord in coords:
        x, y = coord
        diff = target[:, :, x, y] - input[:, :, x, y]
        mse += (diff ** 2)
    if reduction == 'mean':
        return mse / len(masked_coords)
    elif reduction == 'none':
        return mse
    else:
        raise ValueError('Invalid reduction method')
