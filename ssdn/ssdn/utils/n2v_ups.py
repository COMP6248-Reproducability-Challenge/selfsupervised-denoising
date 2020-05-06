import numpy as np
import math
from random import randint

def manipulate(image, subpatch_size = 5, inplace: bool = False):
    """
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

    coords = get_stratified_coords(tuple(image.shape))
    for coord in zip(*coords):
        x, y = coord
        min_x = min([x - subpatch_radius, 0])
        max_x = min([x + subpatch_radius, image_y])
        min_y = min([y - subpatch_radius, 0])
        max_y = min([y + subpatch_radius, image_y])

        rand_x = rand_num_exclude(min_x, max_x, [x])
        rand_y = rand_num_exclude(min_y, max_y, [y])

        # Now replace pixel at x,y with pixel from rand_x,rand_y
        image[:, :, x, y] = image[:, :, rand_x, rand_y]
    return image

def rand_num_exclude(_min, _max, exclude: list):
    rand = randint(_min, _max)
    return rand_num_exclude(_min, _max, exclude) if rand in exclude else rand

def pm_uniform_withCP(local_sub_patch_radius):
    def random_neighbor_withCP_uniform(patch, coords, dims):
        vals = []
        for coord in zip(*coords):
            # Take coord to be the center pixel.

            #
            sub_patch = get_subpatch(patch, coord,local_sub_patch_radius)
            rand_coords = [np.random.randint(0, s) for s in sub_patch.shape[0:dims]]
            vals.append(sub_patch[tuple(rand_coords)])
        return vals
    return random_neighbor_withCP_uniform




def get_subpatch(patch, coord, local_sub_patch_radius):
    start = np.maximum(0, np.array(coord) - local_sub_patch_radius)
    end = start + local_sub_patch_radius*2 + 1

    shift = np.minimum(0, patch.shape - end)

    start += shift
    end += shift

    slices = [ slice(s, e) for s, e in zip(start, end)]

    return patch[tuple(slices)]


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