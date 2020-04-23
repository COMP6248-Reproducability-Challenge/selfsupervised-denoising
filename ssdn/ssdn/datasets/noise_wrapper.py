import torch
import numpy as np
import ssdn

from torch import Tensor

from torch.utils.data import Dataset
from ssdn.params import NoiseAlgorithm

from enum import Enum, auto
from typing import Union, Dict, Tuple
from numbers import Number
from ssdn.utils.data_format import DataFormat, DATA_FORMAT_DIM_INDEX, DataDim


class NoisyDataset(Dataset):
    INPUT = 0
    REFERENCE = 1
    METADATA = 2

    class Metadata(Enum):
        INDEXES = auto()
        INPUT_NOISE_VALUES = auto()
        REFERENCE_NOISE_VALUES = auto()
        INPUT_SHAPE = auto()
        REFERENCE_SHAPE = auto()

    def __init__(
        self,
        child: Dataset,
        noise_style: str,
        algorithm: NoiseAlgorithm,
        enable_metadata: bool = True,
        pad_uniform: bool = True,
        pad_multiple: int = 32,
        square: bool = False,
        data_format: str = DataFormat.CHW
    ):
        self.child = child
        self.enable_metadata = enable_metadata
        self.noise_style = noise_style
        self.algorithm = algorithm
        self.pad_uniform = pad_uniform
        self.pad_multiple = pad_multiple
        self.square = square
        self._max_image_size = None
        self.data_format = data_format

    def __getitem__(self, index: int):
        data = self.child.__getitem__(index)
        img = data[0]
        if self.enable_metadata:
            # Create metadata object
            metadata = {}
            metadata[NoisyDataset.Metadata.INDEXES] = index
        else:
            metadata = None
        # Noisify and create appropriate reference
        (inp, ref, metadata) = self.prepare_input(img, metadata)

        if self.enable_metadata:
            return (inp, ref, metadata)
        else:
            return (inp, ref)

    def __len__(self) -> int:
        return self.child.__len__()

    def prepare_input(
        self, clean: Tensor, metadata: Dict = {}
    ) -> Tuple[Tensor, Tensor, Dict]:
        # Helper function to fix coefficient shape to [n, 1, 1, 1] shape
        def broadcast_coeffs(imgs: Tensor, coeffs: Union[Tensor, Number]):
            return torch.zeros((imgs.shape[0], 1, 1, 1)) + coeffs

        # Create the noisy input images
        noisy_in, noisy_in_coeff = ssdn.utils.noise.add_style(clean, self.noise_style)
        inp, inp_coeff = noisy_in, noisy_in_coeff

        # N2C requires noisy input and clean reference images
        if self.algorithm == NoiseAlgorithm.NOISE_TO_CLEAN:
            ref, ref_coeff = clean, 0
        # N2N requires noisy input and noisy reference images
        elif self.algorithm == NoiseAlgorithm.NOISE_TO_NOISE:
            ref, ref_coeff = ssdn.utils.noise.add_style(clean, self.noise_style)
        # SSDN requires noisy input and no reference images
        elif self.algorithm == NoiseAlgorithm.SELFSUPERVISED_DENOISING:
            ref, ref_coeff = torch.zeros(0), 0
        # SSDN mean only requires noisy input and same image as noisy input reference
        elif self.algorithm == NoiseAlgorithm.SELFSUPERVISED_DENOISING_MEAN_ONLY:
            ref, ref_coeff = noisy_in, noisy_in_coeff
        else:
            raise NotImplementedError("Denoising algorithm not supported")

        # Fill metdata dictionary
        if metadata is not None:
            metadata[NoisyDataset.Metadata.INPUT_NOISE_VALUES] = broadcast_coeffs(
                inp, inp_coeff
            )
            metadata[NoisyDataset.Metadata.REFERENCE_NOISE_VALUES] = broadcast_coeffs(
                ref, ref_coeff
            )
            metadata[NoisyDataset.Metadata.INPUT_SHAPE] = inp.shape
            metadata[NoisyDataset.Metadata.REFERENCE_SHAPE] = ref.shape

        # Original implementation pads before adding noise, here it is done after as it
        # reduces the false scenario of adding structured noise across the full image
        inp, ref = self.pad_to_output_size(inp), self.pad_to_output_size(ref)

        return (inp, ref, metadata)


    @property
    def max_image_size(self):
        if self._max_image_size is None:
            # Get maximum image size in order CHW
            df = DATA_FORMAT_DIM_INDEX[self.data_format]
            dims = (df[DataDim.CHANNEL], df[DataDim.HEIGHT], df[DataDim.WIDTH])
            max_image_size = [max([data[0].shape[axis] for data in self.child]) for axis in dims]
            # Convert back into data format used by dataset
            labelled = zip(max_image_size, dims)
            ordered = [size for size, _ in sorted(labelled, key=lambda x: x[1])]
            self._max_image_size = ordered
        return self._max_image_size


    def get_output_size(self, image: Tensor) -> Tensor:
        df = DATA_FORMAT_DIM_INDEX[self.data_format]
        # Use largest image size in dataset if returning uniform sized tensors
        if self.pad_uniform:
            image_size = self.max_image_size
        else:
            image_size = image.shape
        image_size = list(image_size)

        # Pad width and height axis up to a supported multiple
        if self.pad_multiple:
            pad = self.pad_multiple
            for dim in [DataDim.HEIGHT, DataDim.WIDTH]:
                image_size[df[dim]] = (image_size[df[dim]] + pad - 1) // pad * pad

        # Pad to be a square
        if self.square:
            size = max(image_size[df[DataDim.HEIGHT]], image_size[df[DataDim.WIDTH]])
            image_size[df[DataDim.HEIGHT]] = size
            image_size[df[DataDim.WIDTH]] = size

        return torch.Size(image_size)

    def pad_to_output_size(self, image: Tensor) -> Tensor:
        supported = [DataFormat.CHW, DataFormat.CWH, DataFormat.BCHW, DataFormat.BCWH]
        if self.data_format not in supported:
            raise NotImplementedError("Padding not supported by data format")

        df = DATA_FORMAT_DIM_INDEX[self.data_format]
        output_size = self.get_output_size(image)
        # Already correct, do not pad
        if output_size == image.shape:
            return image
        left, top = 0, 0
        right = output_size[df[DataDim.WIDTH]] - image.shape[df[DataDim.WIDTH]]
        bottom = output_size[df[DataDim.HEIGHT]] - image.shape[df[DataDim.HEIGHT]]
        # Pad Width/Height ignoring other axis
        pad_matrix = [[0, 0]] * len(self.data_format)
        pad_matrix[df[DataDim.WIDTH]] = [left, right]
        pad_matrix[df[DataDim.HEIGHT]] = [top, bottom]
        # PyTorch methods expect PIL images so fallback to Numpy for padding
        np_padded = np.pad(image, pad_matrix, mode='reflect')
        # Convert back to Tensor
        return torch.tensor(np_padded, device=image.device, requires_grad=image.requires_grad)
