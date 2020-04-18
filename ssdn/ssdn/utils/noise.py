"""Contains noise methods for use with batched image inputs.
"""

import torch
import ssdn

from torch import Tensor
from torch.distributions import Uniform, Poisson
from numbers import Number
from typing import Union, Tuple

operation_seed_counter = 0


def get_seed():
    global operation_seed_counter
    operation_seed_counter += 1
    return operation_seed_counter


def set_seed(seed: int):
    global operation_seed_counter
    operation_seed_counter = seed


def add_gaussian(
    tensor: Tensor,
    std_dev: Union[Number, Tuple[Number, Number]],
    mean: Number = 0,
    inplace: bool = False,
) -> Tuple[Tensor, Union[Number, Tensor]]:
    """Adds Gaussian noise to a batch of input images.

    Args:
        tensor (Tensor): Tensor to add noise to; this should be in a B*** format, e.g. BCHW.
        std_dev (Union[Number, Tuple[Number, Number]]): Standard deviation of noise being
            added. If a Tuple is provided then a standard deviation pulled from the
            uniform distribution between the two value is used for each batched input (B***).
            If the input value(s) are integers they will be divided by 255 inline with the
            input image dynamic ranges.
        mean (Number, optional): Mean of noise being added. Defaults to 0.
        inplace (bool, optional): Whether to add the noise in-place. Defaults to False.

    Returns:
        Tuple[Tensor, Union[Number, Tensor]]: Tuple containing:
            * Copy of or reference to input tensor with noise added.
            * Standard deviation (SD) used for noise generation. This will be an array of
            the different SDs used if a range of SDs are being used.
    """
    if not inplace:
        tensor = tensor.clone()

    torch.manual_seed(get_seed())
    if isinstance(std_dev, (list, tuple)):
        if len(std_dev) == 1:
            std_dev = std_dev[0]
        else:
            assert len(std_dev) == 2
            (min_std_dev, max_std_dev) = std_dev
            if isinstance(min_std_dev, int):
                min_std_dev /= 255
            if isinstance(max_std_dev, int):
                max_std_dev /= 255
            uniform_generator = Uniform(min_std_dev, max_std_dev)
            std_dev = uniform_generator.sample((tensor.shape[0], 1, 1, 1))
    if isinstance(std_dev, int):
        std_dev = std_dev / 255
    tensor = tensor.add_(torch.randn(tensor.size()) * std_dev + mean)
    tensor = ssdn.utils.clip_img(tensor, inplace=True)

    return tensor, std_dev


def add_poisson(
    tensor: Tensor, lam: Union[Number, Tuple[Number, Number]], inplace: bool = False
) -> Tuple[Tensor, Union[Number, Tensor]]:
    """Adds Poisson noise to a batch of input images.

    Args:
        tensor (Tensor): Tensor to add noise to; this should be in a B*** format, e.g. BCHW.
        lam (Union[Number, Tuple[Number, Number]]): Distribution rate parameter (lambda) for
            noise being added. If a Tuple is provided then the lambda is pulled from the
            uniform distribution between the two value is used for each batched input (B***).
        inplace (bool, optional): Whether to add the noise in-place. Defaults to False.

    Returns:
        Tuple[Tensor, Union[Number, Tensor]]: Tuple containing:
            * Copy of or reference to input tensor with noise added.
            * Lambda used for noise generation. This will be an array of the different
            lambda used if a range of lambda are being used.
    """
    if not inplace:
        tensor = tensor.clone()

    torch.manual_seed(get_seed())
    if isinstance(lam, (list, tuple)):
        if len(lam) == 1:
            lam = lam[0]
        else:
            assert len(lam) == 2
            (min_lam, max_lam) = lam
            uniform_generator = Uniform(min_lam, max_lam)
            lam = uniform_generator.sample((tensor.shape[0], 1, 1, 1))
    tensor.mul_(lam)
    poisson_generator = Poisson(torch.tensor(1, dtype=float))
    noise = poisson_generator.sample(tensor.shape)
    tensor.add_(noise)
    tensor.div_(lam)
    tensor = ssdn.utils.clip_img(tensor, inplace=True)

    return tensor, lam


def add_style(
    images: Tensor, style: str, inplace: bool = False
) -> Tuple[Tensor, Union[Number, Tensor]]:
    """Adds a style using a string configuration in the format: 'gauss_{SD}',
    'gauss_{MIN_SD}_{MAX_SD}', 'poisson_{LAMBDA}', 'poisson_{MIN_LAMBDA}_{MAX_LAMBDA}'.
    If SD parameters contain a decimal point they are treated as floats. This means the
    underlying `add_gaussian` method will not attempt to scale them.

    Args:
        images (Tensor): Tensor to add noise to; this should be in a B*** format, e.g. BCHW.
        style (str): Style string. NotImplementedError will be thrown if the noise type is
            not valid.
        inplace (bool, optional): Whether to add the noise in-place. Defaults to False.

    Returns:
        Tuple[Tensor, Union[Number, Tensor]]: Tuple containing:
            * Copy or reference of input tensor with noise added.
            * Noise parameters from underlying noise generation.
    """
    # Gaussian noise with constant/variable std.dev
    if style.startswith("gauss"):
        params = [p for p in style.replace("gauss", "", 1).split("_")]
        use_floats = any(map(lambda x: "." in x, params))
        if use_floats:
            params = [float(p) for p in params]
        else:
            params = [int(p) for p in params]
        return add_gaussian(images, params, inplace=inplace)
    elif style.startswith("poisson"):  # Poisson noise with constant/variable lambda.
        params = [float(p) for p in style.replace("poisson", "", 1).split("_")]
        return add_poisson(images, params, inplace=inplace)
    else:
        raise NotImplementedError("Noise type not supported")
