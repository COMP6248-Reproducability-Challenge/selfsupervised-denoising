from torch.utils.data import DataLoader, Dataset
from ssdn.datasets import FixedLengthSampler, SamplingOrder
from typing import List, Any

num_workers = 4


class MockIndexDataset(Dataset):
    """Simple dataset which just returns the index that was read for
    tracking read order.

    Args:
        length (int): Maximum index to return.
    """

    def __init__(self, length: int):
        self.length = length

    def __getitem__(self, index: int):
        return index

    def __len__(self):
        return self.length


def read_dataloader(dataloader: DataLoader) -> List[int]:
    return [data for data in dataloader]


def collapse_index_batches(index_batches: List[List[Any]]) -> List[Any]:
    indexes = []
    for batch in index_batches:
        indexes.extend(batch)
    return indexes


def split_list(list: List, n: int) -> List[List[Any]]:
    return [list[i * n : (i + 1) * n] for i in range((len(list) + n - 1) // n)]


def check_sequential_with_reset(indexes: List[int], max_index: int):
    expect = 0
    for index in indexes:
        if index != expect:
            raise AssertionError("Got: {}, Expected: {}".format(index, expect))
        if index + 1 >= max_index:
            expect = 0
        else:
            expect = index + 1


def check_full_batches(index_batches: List[List[Any]], batch_size: int):
    if any(map(lambda x: len(x) != batch_size, index_batches)):
        raise AssertionError("Expected '{}' length index sets ".format(batch_size))


def check_index_repetition(indexes: List[int], max_index: int):
    # Ensure no repeats within shuffled subsection
    index_sets = split_list(indexes, max_index)
    if any(map(lambda x: len(set(x)) != len(x), index_sets)):
        raise AssertionError("An index was repeated within a subsection")


def _test_sequential(dataset_length: int, num_samples: int):
    dataset = MockIndexDataset(dataset_length)
    sampler = FixedLengthSampler(dataset, num_samples=num_samples, shuffled=False)
    loader_params = {
        "batch_size": 1,
        "num_workers": num_workers,
        "drop_last": False,
        "sampler": sampler,
    }
    dataloader = DataLoader(dataset, **loader_params)
    indexes = read_dataloader(dataloader)
    check_sequential_with_reset(indexes, dataset_length)


def test_sequential_oversampled():
    _test_sequential(5, 8)


def test_sequential_undersampled():
    _test_sequential(20, 8)


def test_sequential_source_length():
    _test_sequential(20, None)


def test_sequential_batched():
    dataset_length = 5
    batch_size = 2
    dataset = MockIndexDataset(dataset_length)
    sampler = FixedLengthSampler(dataset, num_samples=9, shuffled=False)
    loader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": True,
        "sampler": sampler,
    }
    dataloader = DataLoader(dataset, **loader_params)
    index_batches = read_dataloader(dataloader)
    # Note drop_last == True ensures we only get full batches back
    check_full_batches(index_batches, batch_size)
    indexes = collapse_index_batches(index_batches)
    check_sequential_with_reset(indexes, dataset_length)


def _test_shuffled(dataset_length: int, num_samples: int):
    dataset = MockIndexDataset(dataset_length)
    sampler = FixedLengthSampler(dataset, num_samples=num_samples, shuffled=True)
    loader_params = {
        "batch_size": 1,
        "num_workers": num_workers,
        "drop_last": False,
        "sampler": sampler,
    }
    dataloader = DataLoader(dataset, **loader_params)
    indexes = read_dataloader(dataloader)
    check_index_repetition(indexes, dataset_length)


def test_shuffled_oversampled():
    _test_shuffled(5, 8)


def test_shuffled_undersampled():
    _test_shuffled(20, 8)


def test_shuffled_source_length():
    _test_shuffled(20, None)


def test_shuffled_batched():
    dataset_length = 5
    batch_size = 2
    dataset = MockIndexDataset(dataset_length)
    sampler = FixedLengthSampler(dataset, num_samples=8, shuffled=True)
    loader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": True,
        "sampler": sampler,
    }
    dataloader = DataLoader(dataset, **loader_params)
    index_batches = read_dataloader(dataloader)
    # Note drop_last == True ensures we only get full batches back
    check_full_batches(index_batches, batch_size)
    indexes = collapse_index_batches(index_batches)
    check_index_repetition(indexes, dataset_length)


def _test_state_saving():
    dataset_length = 5
    batch_size = 2
    pause_batch = 2
    dataset = MockIndexDataset(dataset_length)
    sampler = FixedLengthSampler(dataset, num_samples=9, shuffled=False)
    loader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": False,
        "sampler": sampler,
    }
    dataloader = DataLoader(dataset, **loader_params)
    index_batches_ref = read_dataloader(dataloader)
    indexes_ref = collapse_index_batches(index_batches_ref)

    index_batches = []
    for i, data in enumerate(dataloader):
        index_batches += [data]
        if i == pause_batch:
            break

    # State save, must update actual read count in case data has been loaded
    # that has not yet been used
    saved_iterator = sampler.last_iter()
    read_count = len(collapse_index_batches(index_batches))
    sampler_state_dict = saved_iterator.state_dict()
    sampler_state_dict["index"] = read_count

    # Attempt to resume
    dataset = MockIndexDataset(dataset_length)
    dataloader = DataLoader(dataset, **loader_params)
    loaded_iterator = SamplingOrder.from_state_dict(sampler_state_dict)
    sampler.for_next_iter(loaded_iterator)
    for data in dataloader:
        index_batches += [data]

    # Resumed should match reference read in one iteration
    indexes_ref = collapse_index_batches(index_batches_ref)
    indexes = collapse_index_batches(index_batches)
    zipped = zip(index_batches_ref, index_batches)
    if not indexes_ref == indexes:
        raise AssertionError("Got: {}, Expected: {}".format(indexes, indexes_ref))

    zipped = zip(index_batches_ref, index_batches)
    if not all(map(lambda x: all(x[0] == x[1]), zipped)):
        raise AssertionError(
            "Got: {}, Expected: {}".format(index_batches, index_batches_ref)
        )


def test_state_saving():
    # Run test multiple times to account for multiple workers
    for i in range(5):
        _test_state_saving()
