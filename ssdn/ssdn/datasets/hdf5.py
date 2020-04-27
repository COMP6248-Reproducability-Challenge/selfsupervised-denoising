"""Contains custom dataset for loading from files created by `dataset_tool_h5.py`.
"""
__authors__ = "David Jones <dsj1n15@ecs.soton.ac.uk>"

import h5py
import numpy as np
import torch
import ssdn
import torchvision.transforms.functional as F

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from ssdn.utils.transforms import Transform
from ssdn.utils.data_format import DataFormat, PIL_FORMAT, permute_tuple
from typing import Tuple


class HDF5Dataset(Dataset):
    def __init__(
        self,
        file_path: str,
        transform: Transform = None,
        h5_format: str = PIL_FORMAT,
        output_format: str = DataFormat.CHW,
        channels: int = 3,
    ):
        """Custom dataset for loading from a file stored in the HDF5 format. This is
        provided to mirror the file format created by the `dataset_tool_h5.py` used
        by the Tensorflow implementation. By default outputs are three channel tensors
        where channels are split at the top structure, i.e. [[[R, R]], [[G, G]], [[B, B, B]]].

        Args:
            file_path (str): HDF5 file to load from.
            transform (Transform, optional): A custom transform to apply after loading.
                A PIL input will be fed into this transform. A Tensor conversion operation
                will always occur after. Defaults to None.
            h5_format (str, optional): Format h5 data is stored in. Defaults to PIL_FORMAT.
            output_format (str, optional): Data format to output data in, if None the default
                format used by PyTorch will be used. Defaults to DataFormat.CHW.
            channels (int, optional): Number of output channels (1 or 3). If the loaded
                image is 1 channel and 3 channels are required the single channel
                will be copied across each channel. If 3 channels are loaded and 1 channel
                is required a weighted RGB to L conversion occurs.
        """
        super(HDF5Dataset, self).__init__()

        self.file_path = file_path
        with h5py.File(self.file_path, "r") as h5file:
            self.img_count = h5file["images"].shape[0]
        self.transform = transform
        self.output_format = output_format
        self.channels = channels
        self.h5_format = h5_format

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        with h5py.File(self.file_path, "r") as h5file:
            img = h5file["images"][index]
            shp = h5file["shapes"][index]
        img = np.reshape(img, shp)
        # Get actual PIL object for transforms to be applied to
        img = img.transpose(*permute_tuple(self.h5_format, "WHC"))
        img = Image.fromarray(img)
        img = ssdn.utils.set_color_channels(img, self.channels)
        # Apply custom transform
        if self.transform:
            img = self.transform(img)
        # Convert to tensor if this hasn't be done during the transform
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)
        if self.output_format is not None:
            img = img.permute(permute_tuple(PIL_FORMAT, self.output_format))

        return img, index

    def image_size(self, index: int, ignore_transform: bool = False) -> Tensor:
        """Quick method to check image size by accessing only shape field. Note that if a
        transform is in place then the data must be loaded directly from the dataset
        to ensure the transform has not changed the shape.

        Args:
            index (int): Index of image in dataset.
            ignore_transform (bool, optional): Whether the transform is known not to
                affect the output size. This will cause the true image size to always
                be returned. Defaults to False.

        Returns:
            Tensor: Shape tensor in output data format
        """
        # Check if quick method viable
        if self.transform is not None and not ignore_transform:
            return torch.tensor(self.__getitem__(index)[0].shape)
        # Can use quick method
        with h5py.File(self.file_path, "r") as h5file:
            shp = h5file["shapes"][index]
        shp = shp[list(permute_tuple(self.h5_format, self.output_format))]
        return torch.tensor(shp)

    def __len__(self):
        return self.img_count
