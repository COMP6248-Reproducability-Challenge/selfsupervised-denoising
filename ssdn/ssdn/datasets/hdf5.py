"""Contains custom dataset for loading from files created by `dataset_tool_h5.py`.
"""
__authors__ = "David Jones <dsj1n15@ecs.soton.ac.uk>"

import h5py
import numpy as np
import torch
import ssdn.utils
import torchvision.transforms.functional as F

from torch.utils.data import Dataset
from ssdn.utils import Transform


class HDF5Dataset(Dataset):

    def __init__(self, file_path: str, transform: Transform = None):
        """Custom dataset for loading from a file stored in the HDF5 format. This is
        provided to mirror the file format created by the `dataset_tool_h5.py` used
        by the Tensorflow implementation. By default outputs are three channel tensors
        where channels are split at the top structure, i.e. [[[R, R]], [[G, G]], [[B, B, B]]].

        Args:
            file_path (str): HDF5 file to load from.
            transform (Transform, optional): A custom transform to apply after loading.
                A PIL input will be fed into this transform. A Tensor conversion operation
                will always occur after. Defaults to None.
        """
        super(HDF5Dataset, self).__init__()

        self.file_path = file_path
        with h5py.File(self.file_path, "r") as h5file:
            self.img_count = h5file["images"].shape[0]
        self.transform = transform

    def __getitem__(self, index: int):
        with h5py.File(self.file_path, "r") as h5file:
            img = h5file["images"][index]
            shp = h5file["shapes"][index]
        img = ssdn.utils.set_color_channels(np.reshape(img, shp), 3)
        # Apply custom transform, force to PIL format
        if self.transform:
            img = F.to_pil_image(img)
            img = self.transform(img)
        # Convert to tensor if this hasn't be done during the transform
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)
        return img

    def __len__(self):
        return self.img_count
