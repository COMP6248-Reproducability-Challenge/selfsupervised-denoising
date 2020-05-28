import torch.nn as nn

from torch import Tensor
from typing import Tuple


class Crop2d(nn.Module):
    """Crop input using slicing. Assumes BCHW data.

    Args:
        crop (Tuple[int, int, int, int]): Amounts to crop from each side of the image.
            Tuple is treated as [left, right, top, bottom]/
    """

    def __init__(self, crop: Tuple[int, int, int, int]):
        super().__init__()
        self.crop = crop
        assert len(crop) == 4

    def forward(self, x: Tensor) -> Tensor:
        (left, right, top, bottom) = self.crop
        x0, x1 = left, x.shape[-1] - right
        y0, y1 = top, x.shape[-2] - bottom
        return x[:, :, y0:y1, x0:x1]


class Shift2d(nn.Module):
    """Shift an image in either or both of the vertical and horizontal axis by first
    zero padding on the opposite side that the image is shifting towards before
    cropping the side being shifted towards.

    Args:
        shift (Tuple[int, int]): Tuple of vertical and horizontal shift. Positive values
            shift towards right and bottom, negative values shift towards left and top.
    """

    def __init__(self, shift: Tuple[int, int]):
        super().__init__()
        self.shift = shift
        vert, horz = self.shift
        y_a, y_b = abs(vert), 0
        x_a, x_b = abs(horz), 0
        if vert < 0:
            y_a, y_b = y_b, y_a
        if horz < 0:
            x_a, x_b = x_b, x_a
        # Order : Left, Right, Top Bottom
        self.pad = nn.ZeroPad2d((x_a, x_b, y_a, y_b))
        self.crop = Crop2d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x: Tensor) -> Tensor:
        return self.shift_block(x)
