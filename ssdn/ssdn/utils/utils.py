"""Generic utilities for use with package.
"""

__authors__ = "David Jones <dsj1n15@ecs.soton.ac.uk>"

import numpy as np

from nptyping import Array
from numbers import Number


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
