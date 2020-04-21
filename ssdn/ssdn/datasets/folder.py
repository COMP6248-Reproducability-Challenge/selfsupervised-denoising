"""Contains custom dataset for loading unlabelled images from a folder.
"""
__authors__ = "David Jones <dsj1n15@ecs.soton.ac.uk>"

import torch
import torchvision.transforms.functional as F
import os
import glob

from ssdn.utils.transforms import Transform, NoiseTransform
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS


class UnlabelledImageFolderDataset(Dataset):
    """Custom dataset using behaviour similar to Torchvision's `ImageFolder` with the
    difference of not expecting subfolders of labelled data. By default outputs are three
    channel tensors where channels are split at the top structure, i.e.
    [[[R, R]], [[G, G]], [[B, B, B]]].

    Args:
        dir_path (str): Root directory to load images from.
        extensions (str, optional): Image extensions to match on. Defaults to those used
            by Torchvision's `ImageFolder` dataset.
            transform (Transform, optional): A custom transform to apply after loading.
                A PIL input will be fed into this transform. A Tensor conversion operation
                will always occur after. Defaults to None.
        recursive (bool, optional): Whether to search folders recursively for images.
            Defaults to False.
    """

    def __init__(
        self,
        dir_path: str,
        extensions: str = IMG_EXTENSIONS,
        transform: Transform = None,
        recursive: bool = False,
    ):
        super(UnlabelledImageFolderDataset, self).__init__()
        self.dir_path = dir_path
        files = []
        for ext in extensions:
            files.extend(
                glob.glob(os.path.join(dir_path, "*" + ext), recursive=recursive)
            )
        self.files = sorted(files)
        self.loader = default_loader
        self.transform = transform

        if len(files) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in directory: " + self.dir_path + "\n"
                    "Supported extensions are: " + ",".join(extensions)
                )
            )

    def __getitem__(self, index: int):
        path = self.files[index]
        img = self.loader(path)
        # Apply custom transform
        if self.transform:
            img = self.transform(img)
        # Convert to tensor if this hasn't be done during the transform
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)
        return img, index

    def __len__(self):
        return len(self.files)
