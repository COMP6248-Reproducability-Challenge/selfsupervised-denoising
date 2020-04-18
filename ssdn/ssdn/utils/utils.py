"""Generic utilities for use with package.
"""

__authors__ = "David Jones <dsj1n15@ecs.soton.ac.uk>"

import numpy as np
import torch

from nptyping import Array
from numbers import Number
from torch import Tensor

def set_color_channels(x: Array[Number], num_channels: int):
    # ## FROM NVIDIA SOURCE ## #
    assert x.shape[0] in [1, 3, 4]
    x = x[: min(x.shape[0], 3)]  # drop possible alpha channel
    if x.shape[0] == num_channels:
        return x
    elif x.shape[0] == 1:
        return np.tile(x, [3, 1, 1])
    y = np.mean(x, axis=0, keepdims=True)
    if np.issubdtype(x.dtype, np.integer):
        y = np.round(y).astype(x.dtype)
    return y


def compute_ramped_lrate(
    i: int,
    iteration_count: int,
    ramp_up_fraction: float,
    ramp_down_fraction: float,
    learning_rate: float,
) -> float:
    if ramp_up_fraction > 0.0:
        ramp_up_end_iter = iteration_count * ramp_up_fraction
        if i <= ramp_up_end_iter:
            t = (i / ramp_up_fraction) / iteration_count
            learning_rate = learning_rate * (0.5 - np.cos(t * np.pi) / 2)

    if ramp_down_fraction > 0.0:
        ramp_down_start_iter = iteration_count * (1 - ramp_down_fraction)
        if i >= ramp_down_start_iter:
            t = ((i - ramp_down_start_iter) / ramp_down_fraction) / iteration_count
            learning_rate = learning_rate * (0.5 + np.cos(t * np.pi) / 2) ** 2

    return learning_rate


def clip_img(img: Tensor, inplace: bool = False) -> Tensor:
    if not inplace:
        img = img.clone()
    if img.is_floating_point():
        c_min, c_max = (0, 1)
    else:
        c_min, c_max = (0, 255)
    return torch.clamp_(img, c_min, c_max)


if __name__ == "__main__":
    learning_rate = 3e-4
    rampup_fraction = 0.1
    rampdown_fraction = 0.3
    num_iter = 100
    minibatch_size = 2
    for n in range(0, num_iter + minibatch_size, minibatch_size):
        lr = compute_ramped_lrate(
            n, num_iter, rampup_fraction, rampdown_fraction, learning_rate
        )
        print(lr)
