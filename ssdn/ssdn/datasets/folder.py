"""Contains custom dataset for loading unlabelled images from a folder.
"""
__authors__ = "David Jones <dsj1n15@ecs.soton.ac.uk>"

import torch
import torchvision.transforms.functional as F
import os
import glob
import tempfile
import string

from ssdn.utils.transforms import Transform
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from typing import List


def is_fs_case_sensitive() -> bool:
    """Check if file system is case sensitive using a temporary file. The result will
    be cached for future calls. See `https://stackoverflow.com/a/36580834`.

    Returns:
        bool: Whether the file system is case insensitive.
    """
    if not hasattr(is_fs_case_sensitive, "case_sensitive"):
        with tempfile.NamedTemporaryFile(prefix="TmP") as tmp_file:
            setattr(
                is_fs_case_sensitive,
                "case_sensitive",
                not os.path.exists(tmp_file.name.lower()),
            )
    return is_fs_case_sensitive.case_sensitive


def case_insensitive_extensions(extensions: List[str]) -> List[str]:
    """In the case that the current file system is case sensitive, this method will
    map the provided list of extensions from the form ".PNG" to ".[pP][nN][gG]". When
    used with glob case will be ignored on any file system. Any duplicated match patterns
    generated will be removed to ensure files will not be matched twice.

    Args:
        extensions (List[str]): List of file extensions.

    Returns:
        List[str]: Extensions formatted for case insensitivity. The original order may
            not be preserved. If the file system is not case sensitive the original list
            is returned untouched.
    """
    def mp(char: str) -> str:
        if char not in string.ascii_letters:
            return char
        return "[{}{}]".format(char.lower(), char.upper())

    extensions = [ext.lower() for ext in extensions]
    # Add both lower and uppercase versions when file system case sensitive
    if is_fs_case_sensitive():
        cs_extensions = []
        for extension in extensions:
            cs_extensions += ["".join(map(mp, extension))]
        extensions = cs_extensions
    # Remove duplicates
    return list(set(extensions))


def find_files(
    dir_path: str, extensions: List[str], recursive: bool, case_insensitive: bool = True
) -> List[str]:
    """Structured glob match for finding files ending with a given extension. These
    extensions by default will be treated as case insensitive on all file systems.

    Args:
        dir_path (str): Root directory path to search from.
        extensions (List[str]): List of extensions to match. These can be prefixed with
            a '.' but this is not required.
        recursive (bool): Whether to search folders recursively for matches.
        case_insensitive (bool, optional): Whether to force the file system to ignore
            case. Defaults to True.

    Returns:
        List[str]: List of matched files.
    """
    if case_insensitive:
        extensions = case_insensitive_extensions(extensions)
    star_match = os.path.join("**", "*") if recursive else "*"
    files = []
    for ext in extensions:
        if ext[0] != ".":
            ext = "." + ext
        path = os.path.join(dir_path, star_match + ext)
        files.extend(glob.glob(path, recursive=recursive))
    return files


class UnlabelledImageFolderDataset(Dataset):
    """Custom dataset using behaviour similar to Torchvision's `ImageFolder` with the
    difference of not expecting subfolders of labelled data. By default outputs are three
    channel tensors where channels are split at the top structure, i.e.
    [[[R, R]], [[G, G]], [[B, B, B]]].

    Args:
        dir_path (str): Root directory to load images from.
        extensions (str, optional): Image extensions to match on. Defaults to those used
            by Torchvision's `ImageFolder` dataset. These will always be matched in a case
            insensitive manner. Duplicate extensions will be removed.
            transform (Transform, optional): A custom transform to apply after loading.
                A PIL input will be fed into this transform. A Tensor conversion operation
                will always occur after. Defaults to None.
        recursive (bool, optional): Whether to search folders recursively for images.
            Defaults to False.
    """

    def __init__(
        self,
        dir_path: str,
        extensions: List = IMG_EXTENSIONS,
        transform: Transform = None,
        recursive: bool = False,
    ):
        super(UnlabelledImageFolderDataset, self).__init__()
        self.dir_path = dir_path
        self.files = sorted(find_files(dir_path, extensions, recursive))
        self.loader = default_loader
        self.transform = transform

        if len(self.files) == 0:
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
