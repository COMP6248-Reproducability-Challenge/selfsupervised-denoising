import torch
import ssdn

from torch import Tensor

from torch.utils.data import Dataset
from ssdn.params import NoiseAlgorithm

from enum import Enum, auto
from typing import Union, Dict, Tuple
from numbers import Number


class NoisyDataset(Dataset):
    INPUT = 0
    REFERENCE = 1
    METADATA = 2

    class Metadata(Enum):
        INDEXES = auto()
        INPUT_NOISE_VALUES = auto()
        REFERENCE_NOISE_VALUES = auto()

    def __init__(
        self,
        child: Dataset,
        noise_style: str,
        algorithm: NoiseAlgorithm,
        enable_metadata: bool = True,
    ):
        self.child = child
        self.enable_metadata = enable_metadata
        self.noise_style = noise_style
        self.algorithm = algorithm

    def __getitem__(self, index: int):
        # Assume image is always in the first argument
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

        return (inp, ref, metadata)
