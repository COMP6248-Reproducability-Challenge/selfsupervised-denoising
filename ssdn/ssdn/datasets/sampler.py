from __future__ import annotations

"""Contains custom sampler to allow repeated use of the same data with fair extraction.
"""
__authors__ = "David Jones <dsj1n15@ecs.soton.ac.uk>"

import torch

from torch.utils.data import Sampler, Dataset
from typing import Generator, List, Dict


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
        self._next_iter = None
        self._last_iter = None

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
                for idx in torch.randperm(len(self.data_source))[0:n]:
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

    def __iter__(self) -> Generator[int, None, None]:
        if self._next_iter is None:
            sample_order = list(self.sampler())
            self._last_iter = SamplingOrder(sample_order)
            return self._last_iter
        else:
            return self._next_iter

    def __len__(self) -> int:
        return self.num_samples

    def for_next_iter(self, iter_order: SamplingOrder):
        self._next_iter = iter_order

    def last_iter(self) -> Generator[int, None, None]:
        return self._last_iter

class SamplingOrder:
    def __init__(self, order: List[int], index: int = 0):
        self.order = order
        self.index = index

    def __iter__(self) -> Generator[int, None, None]:
        return self

    def __next__(self) -> int:
        if self.index < len(self.order):
            value = self.order[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration()

    def set_read_count(self, read_count: int):
        self.index = read_count

    def state_dict(self) -> Dict:
        state_dict = {"order": self.order, "index": self.index}
        return state_dict

    @staticmethod
    def from_state_dict(state_dict: Dict) -> SamplingOrder:
        return SamplingOrder(state_dict["order"], state_dict["index"])
