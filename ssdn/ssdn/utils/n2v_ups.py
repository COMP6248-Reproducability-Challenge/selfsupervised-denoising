import numpy as np
import math
from random import randint
from torch import Tensor

def manipulate(
    image: Tensor,
    subpatch_size: int = 5,
    inplace: bool = False
):
    """
    TODO: might need to do this with the same mask for each image?
    Take sub-patches from within an image at random coordinates
    to replace a given percentage of central pixels with a random
    pixel within the sub-patch (aka Uniform Pixel Selection).

    Args:
        image (Tensor): a 64x64 image tensor.
        subpatch_size (Number): the size of the sub-patch.

    Returns:
        Tensor: the image after the Uniform Pixel Selection.
    """
    if subpatch_size % 2 == 0:
        raise ValueError("subpatch_size must be odd")

    if not inplace:
        image = image.clone()

    image_x = image.shape[2]
    image_y = image.shape[3]
    subpatch_radius = math.floor(subpatch_size / 2)
    coords = get_stratified_coords((image_x, image_y))

    masked_coords = []
    mask_tensor = torch.zeros(image_x, image_y, dtype=int)
    for coord in zip(*coords):
        x, y = coord
        masked_coords.append((x,y))
        mask_tensor[x, y] = 1

        min_x = min([x - subpatch_radius, 0])
        max_x = min([x + subpatch_radius, image_x - 1])
        min_y = min([y - subpatch_radius, 0])
        max_y = min([y + subpatch_radius, image_y -1])

        rand_x = rand_num_exclude(min_x, max_x, [x])
        rand_y = rand_num_exclude(min_y, max_y, [y])

        # Now replace pixel at x,y with pixel from rand_x,rand_y
        image[:, :, x, y] = image[:, :, rand_x, rand_y]
    return image, masked_coords, mask_tensor

def rand_num_exclude(_min: int, _max: int, exclude: list):
    """
    Get a random integer in a range but excluding any in the list
    or integers to be excluded.

    Args:
        _min (Number): minimum integer value.
        _max (Number): maximum integer value.
        exclude (list): list of integers to be excluded.
    """
    rand = randint(_min, _max)
    return rand_num_exclude(_min, _max, exclude) if rand in exclude else rand


def get_stratified_coords(shape):
    """
    Args:
        shape (Tuple[Number,Number]): the shape of the input patch
    """
    perc_pix = 1.5 # TODO put in config
    box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
    coord_gen = get_random_coords(box_size)

    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    x_coords = []
    y_coords = []
    for i in range(box_count_y):
        for j in range(box_count_x):
            y, x = next(coord_gen)
            y = int(i * box_size + y)
            x = int(j * box_size + x)
            if (y < shape[0] and x < shape[1]):
                y_coords.append(y)
                x_coords.append(x)
    return (y_coords, x_coords)


def get_random_coords(box_size):
    while True:
        # yield used so can call next() on this to get next random coords :D ez
        yield (np.random.rand() * box_size, np.random.rand() * box_size)
