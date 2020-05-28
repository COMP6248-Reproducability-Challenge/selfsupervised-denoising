import torch
from torch import Tensor
import torch.nn as nn


def loss_mask_mse(
    masked_coords: Tensor,
    input: Tensor,
    target: Tensor
):
    mse = 0
    coords = masked_coords.tolist()[0]
    for coord in coords:
        x, y = coord
        diff = target[:, :, x, y] - input[:, :, x, y]
        mse += (diff ** 2)
    return mse

