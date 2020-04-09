"""Contains custom sampler to allow repeated use of the same data with fair extraction.
"""
__authors__ = "David Jones <dsj1n15@ecs.soton.ac.uk>"

import torch

from torch.utils.data import Sampler, Dataset
from typing import Generator


class FixedLengthSampler(Sampler):
    """Sample in either sequential or a random order for the given number of samples. If the
    number of requested samples execeds the dataset, the dataset will loop. Unlike standard
    sampling with replacement this means a sample will only ever be used once more than any
    other sample.

    There is no option for fully random selection with replacement, use PyTorch's
    `RandomSampler` if this behaviour is desired.

    Args:
        data_source (Dataset): Dataset to load samples from.
        num_samples (int, optional): The number of samples to be returned by the dataset.
            Defaults to None; this is equivalent to the length of the dataset.
        shuffled (bool, optional): Whether to randomise order. Defaults to False.
    """

    def __init__(
        self, data_source: Dataset, num_samples: int = None, shuffled: bool = False,
    ):
        self.data_source = data_source
        self._num_samples = num_samples
        self.shuffled = shuffled

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        else:
            return self._num_samples

    def sampler(self) -> Generator[int, None, None]:
        """Iterator handling both shuffled and non-shuffled behaviour.

        Yields:
            Generator[int, None, None]: Next index to sample.
        """
        remaining = self.num_samples
        if self.shuffled:
            while remaining > 0:
                n = min(remaining, len(self.data_source))
                for idx in torch.randperm(len(self.data_source))[0 : n]:
                    yield int(idx)
                remaining -= n
        else:
            current_idx = None
            while remaining > 0:
                if current_idx is None or current_idx >= len(self.data_source):
                    current_idx = 0
                yield current_idx
                current_idx += 1
                remaining -= 1

    def __iter__(self):
        return self.sampler()

    def __len__(self):
        return self.num_samples
