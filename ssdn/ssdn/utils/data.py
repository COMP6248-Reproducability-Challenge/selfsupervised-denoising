import numpy as np
import torch

from torch import Tensor

from PIL import Image
from ssdn.utils.data_format import DataFormat, DataDim, DATA_FORMAT_DIM_INDEX, permute_tuple


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
    dims = DATA_FORMAT_DIM_INDEX[data_format]
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


def show_image(img: Tensor, data_format: str = DataFormat.CHW):
    np_img = img.detach().numpy()
    np_img = np.clip(np_img, 0, 1)
    np_img = np_img.transpose(*permute_tuple(data_format, DataFormat.WHC))
    print(np_img.shape)
    pil_img = Image.fromarray(np.uint8(np_img * 255))
    pil_img.show()
