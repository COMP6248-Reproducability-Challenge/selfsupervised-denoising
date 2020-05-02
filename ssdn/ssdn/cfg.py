import os

from ssdn.params import ConfigValue, DatasetType
from typing import Dict


def base():
    return {
        ConfigValue.TRAIN_ITERATIONS: 100,
        ConfigValue.TRAIN_MINIBATCH_SIZE: 2,
        ConfigValue.TEST_MINIBATCH_SIZE: 2,
        ConfigValue.IMAGE_CHANNELS: 3,
        ConfigValue.TRAIN_PATCH_SIZE: 256,
        ConfigValue.LEARNING_RATE: 3e-4,
        ConfigValue.LR_RAMPDOWN_FRACTION: 0.1,
        ConfigValue.LR_RAMPUP_FRACTION: 0.3,
        ConfigValue.EVAL_INTERVAL: 10,
        ConfigValue.PRINT_INTERVAL: 10,
        ConfigValue.SNAPSHOT_INTERVAL: 10,
        ConfigValue.DATALOADER_WORKERS: 8,
        ConfigValue.PIN_DATA_MEMORY: False,
        ConfigValue.DIAGONAL_COVARIANCE: False,
        ConfigValue.TRAIN_DATASET_TYPE: None,
        ConfigValue.TEST_DATASET_TYPE: None,
        ConfigValue.TRAIN_DATASET_NAME: None,
        ConfigValue.TEST_DATASET_NAME: None,
    }


class DatasetName:
    BSD = "bsd"
    IMAGE_NET = "ilsvrc"
    KODAK = "kodak"
    SET14 = "set14"


def infer_datasets(cfg: Dict):
    """For training and test dataset parameters infer from the path the name of
    the dataset being targetted and whether or not the data should be loaded as
    a h5 file or a folder.

    Args:
        cfg (Dict): [description]
    """

    def infer_dname(path: str):
        # Look for part of dataset name in path for guessing dataset
        dataset_dict = {
            "BSDS300": DatasetName.BSD,
            "ILSVRC": DatasetName.IMAGE_NET,
            "KODAK": DatasetName.KODAK,
            "SET14": DatasetName.SET14,
        }
        potentials = []
        for key, name in dataset_dict.items():
            if key.lower() in path.lower():
                potentials += [name]
        if len(potentials) == 0:
            raise ValueError("Could not infer dataset from path.")
        if len(potentials) > 1:
            raise ValueError("Matched multiple datasets with dataset path.")
        return potentials[0]

    def infer_dtype(path: str):
        # Treat files as HDF5 and directories as folders
        dtype = DatasetType.FOLDER if os.path.isdir(path) else DatasetType.HDF5
        return dtype

    # Infer for training set
    if cfg[ConfigValue.TRAIN_DATA_PATH]:
        if cfg.get(ConfigValue.TRAIN_DATASET_NAME, None) is None:
            cfg[ConfigValue.TRAIN_DATASET_NAME] = infer_dname(
                cfg[ConfigValue.TRAIN_DATA_PATH]
            )
        if cfg.get(ConfigValue.TRAIN_DATASET_TYPE, None) is None:
            cfg[ConfigValue.TRAIN_DATASET_TYPE] = infer_dtype(
                cfg[ConfigValue.TRAIN_DATA_PATH]
            )
    # Infer for testing/validation set
    if cfg[ConfigValue.TEST_DATA_PATH]:
        if cfg.get(ConfigValue.TEST_DATASET_NAME, None) is None:
            cfg[ConfigValue.TEST_DATASET_NAME] = infer_dname(
                cfg[ConfigValue.TEST_DATA_PATH]
            )
        if cfg.get(ConfigValue.TEST_DATASET_TYPE, None) is None:
            cfg[ConfigValue.TEST_DATASET_TYPE] = infer_dtype(
                cfg[ConfigValue.TEST_DATA_PATH]
            )


def test_length(dataset_name: str) -> int:
    """To give meaningful PSNR results similar amounts of data should be evaluated.
    Return the test length based on image size and image count. Note that for all
    datasets it is assumed the test dataset is being used.

    Args:
        dataset_name (str): Name of the dataset (BSD...),

    Returns:
        int: Image count to test for. When higher than the dataset length existing
            images should be reused.
    """
    mapping = {
        DatasetName.BSD: 300,  # 3  x Testset Length
        DatasetName.KODAK: 240,  # 10 x Testset Length
        DatasetName.SET14: 280,  # 20 x Testset Length
    }
    return mapping[dataset_name]
