from torch import Tensor
import torch.nn as nn


def loss_mask_mse(
    masked_coords: list,
    input: torch.Tensor,
    target: Tensor,
    reduction: bool = 'mean'
):
    mse = 0
    for coord in masked_coords:
        x, y = coord
        # TODO: may not want to get mean here as do it in pipeline?
        mse += ((target[:, :, x, y] - input[:, :, x, y]) ** 2).mean().item()
    if reduction == 'mean':
        return mse / len(masked_coords)
    elif reduction == 'none':
        return mse
    else:
        raise ValueError('Invalid reduction method')
