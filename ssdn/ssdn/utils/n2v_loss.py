import torch
from torch import Tensor
import torch.nn as nn


def loss_mask_mse(
    masked_coords: Tensor,
    input: Tensor,
    target: Tensor,
    reduction: bool = 'mean'
):
    mse = 0
    coords = masked_coords.tolist()[0]
    for coord in coords:
        x, y = coord
        mse += ((target[:, :, x, y] - input[:, :, x, y]) ** 2).mean().item()
    if reduction == 'mean':
        return mse / len(masked_coords)
    elif reduction == 'none':
        return mse
    else:
        raise ValueError('Invalid reduction method')
