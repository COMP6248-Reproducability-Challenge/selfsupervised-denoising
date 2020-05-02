import numpy as np
import torch
import torchvision
import torch.nn.functional as F

from torch import Tensor

from PIL import Image
from ssdn.utils.data_format import (
    DataFormat,
    DataDim,
    DATA_FORMAT_DIM_INDEX,
    permute_tuple,
    batch,
    unbatch,
)


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
        return x.flip(w_dim).transpose(h_dim, w_dim)
    elif angle == 180:
        return x.flip(w_dim).flip(h_dim)
    elif angle == 270:
        return x.flip(h_dim).transpose(h_dim, w_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")


def tensor2image(img: Tensor, data_format: str = DataFormat.CHW) -> Image:
    img = img.detach()
    # Create a grid of images if batched
    if isinstance(img, list) or len(img.shape) == 4:
        img = img.permute(permute_tuple(batch(data_format), DataFormat.BCHW))
        img = torchvision.utils.make_grid(img)
        img = img.permute(permute_tuple(DataFormat.CHW, unbatch(data_format)))

    np_img = img.numpy()
    np_img = np.clip(np_img, 0, 1)
    np_img = np_img.transpose(*permute_tuple(data_format, DataFormat.WHC))
    channels = np_img.shape[-1]
    if channels == 3:
        mode = "RGB"
    elif channels == 1:
        mode = "L"
        np_img = np.squeeze(np_img)
    else:
        raise NotImplementedError(
            "Cannot convert image with {} channels to PIL image.".format(channels)
        )
    return Image.fromarray(np.uint8(np_img * 255), mode=mode)


def mse2psnr(mse: Tensor, float_imgs: bool = True):
    high_val = torch.tensor(1.0) if float_imgs else torch.tensor(255)
    return 20 * torch.log10(high_val) - 10 * torch.log10(mse)


def calculate_psnr(img: Tensor, ref: Tensor, data_format: str = DataFormat.BCHW):
    dim_indexes = dict(DATA_FORMAT_DIM_INDEX[data_format])  # shallow copy
    dim_indexes.pop(DataDim.BATCH, None)
    dims = tuple(dim_indexes.values())
    mse = F.mse_loss(img, ref, reduction="none")
    mse = torch.mean(mse, dim=dims)
    return mse2psnr(mse, img.is_floating_point())

def show_tensor_image(img: Tensor, data_format: str = DataFormat.CHW):
    pil_img = tensor2image(img, data_format=data_format)
    pil_img.show()


def save_tensor_image(img: Tensor, path: str, data_format: str = DataFormat.CHW):
    pil_img = tensor2image(img, data_format=data_format)
    pil_img.save(path)


def set_color_channels(img: Image, channels: int) -> Image:
    cur_channels = len(img.getbands())
    if cur_channels != channels:
        if channels == 1:
            return img.convert("L")
        if channels == 3:
            return img.convert("RGB")
    return img
