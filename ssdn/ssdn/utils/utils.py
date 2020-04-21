"""Generic utilities for use with package.
"""

__authors__ = "David Jones <dsj1n15@ecs.soton.ac.uk>"

import numpy as np
import torch

from nptyping import Array
from numbers import Number
from torch import Tensor

from ssdn.utils.data_format import DataFormat, DataDim, DATA_FORMAT_DIMS


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
    """Clip tensor data so that it is within a valid image range. That is
    between 0-1 for float images and 0-255 for int images. Values are clamped
    meaning any values outside this range are set to the limits, other values
    are not touched.

    Args:
        img (Tensor): Image or batch of images to clip.
        inplace (bool, optional): Whether to do the operation in place.
            Defaults to False; this will first clone the data.

    Returns:
        Tensor: Reference to input image or new image.
    """
    if not inplace:
        img = img.clone()
    if img.is_floating_point():
        c_min, c_max = (0, 1)
    else:
        c_min, c_max = (0, 255)
    return torch.clamp_(img, c_min, c_max)


def rotate(
    x: torch.Tensor, angle: int, data_format: str = DataFormat.BCHW
) -> torch.Tensor:
    """Rotate images by 90 degrees clockwise. Can handle any 2D data format.
    Args:
        x (Tensor): Image or batch of images.
        angle (int): Clockwise rotation angle in multiples of 90.
        data_format (str, optional): Format of input image data, e.g. BCHW,
            HWC. Defaults to BCHW.
    Returns:
        Tensor: Copy of tensor with rotation applied.
    """
    dims = DATA_FORMAT_DIMS[data_format]
    h_dim = dims[DataDim.HEIGHT]
    w_dim = dims[DataDim.WIDTH]

    if angle == 0:
        return x
    elif angle == 90:
        return x.transpose(h_dim, w_dim).flip(w_dim)
    elif angle == 180:
        return x.flip(h_dim)
    elif angle == 270:
        return x.transpose(h_dim, w_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")
