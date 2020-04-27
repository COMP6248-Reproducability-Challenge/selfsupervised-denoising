"""Generic utilities for use with package.
"""

__authors__ = "David Jones <dsj1n15@ecs.soton.ac.uk>"

import numpy as np
import torch
import torch.functional as F
import re
import os

from typing import Any, List
from torch import Tensor
from contextlib import contextmanager


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


def mse2psnr(mse: Tensor, float_imgs: bool = True):
    high_val = torch.tensor(1.0) if float_imgs else torch.tensor(255)
    return 20 * torch.log10(high_val) - 10 * torch.log10(mse)


def calculate_psnr(img: Tensor, ref: Tensor, axis: int = None):
    mse = F.reduce_mean((img - ref) ** 2, axis=axis)
    return mse2psnr(mse, img.is_floating_point())


def list_constants(clazz: Any, private: bool = False) -> List[Any]:
    """Fetch all values from variables formatted as constants in a class.

    Args:
        clazz (Any): Class to fetch constants from.

    Returns:
        List[Any]: List of values.
    """
    variables = [i for i in dir(clazz) if not callable(i)]
    regex = re.compile(r"^{}[A-Z0-9_]*$".format("" if private else "[A-Z]"))
    names = list(filter(regex.match, variables))
    values = [clazz.__dict__[name] for name in names]
    return values


@contextmanager
def cd(newdir: str):
    """Context manager for managing changes of directory where when the context is left
    the original directory is restored.

    Args:
        newdir (str): New directory to enter
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
