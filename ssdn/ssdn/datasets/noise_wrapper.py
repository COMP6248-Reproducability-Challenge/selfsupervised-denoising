import torch
import numpy as np
import ssdn

from torch import Tensor

from torch.utils.data import Dataset
from ssdn.params import NoiseAlgorithm

from enum import Enum, auto
from typing import Union, Dict, Tuple, List, Optional
from numbers import Number
from ssdn.utils.data_format import DataFormat, DATA_FORMAT_DIM_INDEX, DataDim

NULL_IMAGE = torch.zeros(0)


class NoisyDataset(Dataset):
    """Wrapper for a child dataset for creating inputs for training denoising
    algorithms. This involves adding noise to data and creating appropriate
    references for the algorithm being trained. Data can be padded to match
    requirements of input network; this can be unpadded again using information
    provided in the metadata dictionary. Metadata includes shapes of the inputs
    before padding, the index of the data returned, and noise coefficients used.

    Args:
        child (Dataset): Child dataset to load data from. It is expected that an
            unlabelled image is the first element of any returned data.
        noise_style (str): The noise style to use in string representation.
        algorithm (NoiseAlgorithm): The algorithm the loader should prepare data for.
            This will dictate the appropriate reference images created.
        enable_metadata (bool, optional): Whether to return a dictionary containing
            information about data creation. When False only two values are returned.
            Defaults to True.
        pad_uniform (bool, optional): Whether to pad returned images to the same size
            as the largest image. This may cause very slow initialisation for large
            datasets. Defaults to False.
        pad_multiple (int, optional): Whether to pad the width and height of returned
                images to a divisor. Ignored if None. Defaults to None.
        square (bool, optional): Whether to pad such that width and height are equal.
            Defaults to False.
        data_format (str, optional): Format of data from underlying dataset.
            Defaults to DataFormat.CHW.
    """

    INPUT = 0
    REFERENCE = 1
    METADATA = 2
    """ Indexes for returned data."""

    def __init__(
        self,
        child: Dataset,
        noise_style: str,
        algorithm: NoiseAlgorithm,
        enable_metadata: bool = True,
        pad_uniform: bool = False,
        pad_multiple: int = None,
        square: bool = False,
        data_format: str = DataFormat.CHW,
    ):
        self.child = child
        self.enable_metadata = enable_metadata
        self.noise_style = noise_style
        self.algorithm = algorithm
        self.pad_uniform = pad_uniform
        self.pad_multiple = pad_multiple
        self.square = square
        self.data_format = data_format

        # Initialise max image size property, this will load all data
        self._max_image_size = None
        if self.pad_uniform:
            _ = self.max_image_size

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Optional[Dict]]:
        data = self.child.__getitem__(index)
        img = data[0]

        if self.enable_metadata:
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
        """Translate clean reference into training input and reference. The algorithm
        being trained dictates the reference used, e.g. Noise2Noise will create a
        noisy input and noisy reference.

        Args:
            clean (Tensor): Clean input image to create noisy input and reference from.
            metadata (Dict, optional): Dictionary to fill with metadata. Defaults to
                creating a new dictionary.

        Returns:
            Tuple[Tensor, Tensor, Dict]: Input, Reference, Metadata Dictionary
        """
        # Helper function to fix coefficient shape to [1, 1, 1] shape, batcher will
        # automatically elevate to [n, 1, 1, 1] shape if required
        def broadcast_coeffs(imgs: Tensor, coeffs: Union[Tensor, Number]):
            return torch.zeros((1, 1, 1)) + coeffs

        # Track the true shape as batching may lead to padding distorting this shape
        image_shape = clean.shape

        # Create the noisy input images
        noisy_in, noisy_in_coeff = ssdn.utils.noise.add_style(clean, self.noise_style)
        noisy_in = ssdn.utils.n2v_ups.manipulate(noisy_in, 5) # TODO use config for neighbourhood radius
        inp, inp_coeff = noisy_in, noisy_in_coeff

        # N2C requires noisy input and clean reference images
        if self.algorithm == NoiseAlgorithm.NOISE_TO_CLEAN:
            ref, ref_coeff = clean, 0
        # N2N requires noisy input and noisy reference images
        elif self.algorithm == NoiseAlgorithm.NOISE_TO_NOISE:
            ref, ref_coeff = ssdn.utils.noise.add_style(clean, self.noise_style)
        elif self.algorithm == NoiseAlgorithm.NOISE_TO_VOID:
            ref, ref_coeff = ssdn.utils.noise.add_style(clean, self.noise_style)
        # SSDN requires noisy input and no reference images
        elif self.algorithm == NoiseAlgorithm.SELFSUPERVISED_DENOISING:
            ref, ref_coeff = NULL_IMAGE, 0
        # SSDN mean only requires noisy input and same image as noisy input reference
        elif self.algorithm == NoiseAlgorithm.SELFSUPERVISED_DENOISING_MEAN_ONLY:
            ref, ref_coeff = noisy_in, noisy_in_coeff
        else:
            raise NotImplementedError("Denoising algorithm not supported")

        # Original implementation pads before adding noise, here it is done after as it
        # reduces the false scenario of adding structured noise across the full image
        inp = self.pad_to_output_size(inp)
        if ref is not NULL_IMAGE:
            ref = self.pad_to_output_size(ref)

        # Fill metdata dictionary
        if metadata is not None:
            metadata[NoisyDataset.Metadata.CLEAN] = self.pad_to_output_size(clean)
            metadata[NoisyDataset.Metadata.IMAGE_SHAPE] = torch.tensor(image_shape)
            metadata[NoisyDataset.Metadata.INPUT_NOISE_VALUES] = broadcast_coeffs(
                inp, inp_coeff
            )
            metadata[NoisyDataset.Metadata.REFERENCE_NOISE_VALUES] = broadcast_coeffs(
                ref, ref_coeff
            )

        return (inp, ref, metadata)

    @property
    def max_image_size(self) -> List[int]:
        """ Find the maximum image size in the dataset. Will try calling `image_size` method
        first in case a fast method for checking size has been implemented. Will fall back
        to loading images from the dataset as normal and checking their shape. Once this
        method has been called once the maximum size will be cached for subsequent calls.
        """
        if self._max_image_size is None:
            try:
                image_sizes = [self.child.image_size(i) for i in range(len(self.child))]
            except AttributeError:
                image_sizes = [torch.tensor(data[0].shape) for data in self.child]

            image_sizes = torch.stack(image_sizes)
            max_image_size = torch.max(image_sizes, dim=0).values
            self._max_image_size = max_image_size
        return self._max_image_size

    def get_output_size(self, image: Tensor) -> Tensor:
        """Calculate output size of an image using the current padding configuration.
        """
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

        return torch.tensor(image_size)

    def pad_to_output_size(self, image: Tensor) -> Tensor:
        """ Apply reflection padding to the image to meet the current padding
        configuration. Note that padding is handled by Numpy.
        """

        supported = [DataFormat.CHW, DataFormat.CWH, DataFormat.BCHW, DataFormat.BCWH]
        if self.data_format not in supported:
            raise NotImplementedError("Padding not supported by data format")

        df = DATA_FORMAT_DIM_INDEX[self.data_format]
        output_size = self.get_output_size(image)
        # Already correct, do not pad
        if all(output_size == torch.tensor(image.shape)):
            return image
        left, top = 0, 0
        right = output_size[df[DataDim.WIDTH]] - image.shape[df[DataDim.WIDTH]]
        bottom = output_size[df[DataDim.HEIGHT]] - image.shape[df[DataDim.HEIGHT]]
        # Pad Width/Height ignoring other axis
        pad_matrix = [[0, 0]] * len(self.data_format)
        pad_matrix[df[DataDim.WIDTH]] = [left, right]
        pad_matrix[df[DataDim.HEIGHT]] = [top, bottom]
        # PyTorch methods expect PIL images so fallback to Numpy for padding
        np_padded = np.pad(image, pad_matrix, mode="reflect")
        # Convert back to Tensor
        return torch.tensor(
            np_padded, device=image.device, requires_grad=image.requires_grad
        )

    @staticmethod
    def _unpad_single(image: Tensor, shape: Tensor) -> Tensor:
        # Create slice list extracting from 0:n for each shape axis
        slices = list(map(lambda x: slice(*x), (zip([0] * len(shape), shape))))
        return image[slices]

    @staticmethod
    def _unpad(image: Tensor, shape: Tensor) -> Union[Tensor, List[Tensor]]:
        if len(image.shape) <= shape.shape[-1]:
            return NoisyDataset._unpad_single(image, shape)
        return [NoisyDataset._unpad_single(i, s) for i, s in zip(image, shape)]

    @staticmethod
    def unpad(
        image: Tensor, metadata: Dict, batch_index: int = None
    ) -> Union[Tensor, List[Tensor]]:
        """For a padded image or batch of padded images, undo padding. It is
        assumed that the original image is positioned in the top left and
        that the channel count has not changed.

        Args:
            image (Tensor): Single image or batch of images.
            metadata (Tensor): Metadata dictionary associated with images to
                unpad.

        Returns:
            Union[Tensor, List[Tensor]]: Unpadded image tensor if not batched.
                List of unpadded images if batched.
        """
        inp_shape = metadata[NoisyDataset.Metadata.IMAGE_SHAPE]
        if batch_index is not None:
            image = image[batch_index]
            inp_shape = inp_shape[batch_index]
        return NoisyDataset._unpad(image, inp_shape)

    class Metadata(Enum):
        """ Enumeration of fields that can be contained in the metadata dictionary.
        """

        CLEAN = auto()
        IMAGE_SHAPE = auto()
        INDEXES = auto()
        INPUT_NOISE_VALUES = auto()
        REFERENCE_NOISE_VALUES = auto()
