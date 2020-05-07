import os

from ssdn.params import ConfigValue, DatasetType, NoiseAlgorithm, Pipeline
from typing import Dict


DEFAULT_RUN_DIR = "runs"


def base():
    return {
        ConfigValue.TRAIN_ITERATIONS: 5000,
        ConfigValue.TRAIN_MINIBATCH_SIZE: 1,
        ConfigValue.TEST_MINIBATCH_SIZE: 2,
        ConfigValue.IMAGE_CHANNELS: 3,
        ConfigValue.TRAIN_PATCH_SIZE: 64,
        ConfigValue.LEARNING_RATE: 3e-4,
        ConfigValue.LR_RAMPDOWN_FRACTION: 0.1,
        ConfigValue.LR_RAMPUP_FRACTION: 0.3,
        ConfigValue.EVAL_INTERVAL: 1000,
        ConfigValue.PRINT_INTERVAL: 1000,
        ConfigValue.SNAPSHOT_INTERVAL: 1000,
        ConfigValue.DATALOADER_WORKERS: 1,
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
        cfg (Dict): Configuration to infer for.
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


def infer_pipeline(algorithm: NoiseAlgorithm) -> Pipeline:
    if algorithm in [NoiseAlgorithm.SELFSUPERVISED_DENOISING]:
        return Pipeline.SSDN
    elif algorithm in [
        NoiseAlgorithm.SELFSUPERVISED_DENOISING_MEAN_ONLY,
        NoiseAlgorithm.NOISE_TO_NOISE,
        NoiseAlgorithm.NOISE_TO_CLEAN,
    ]:
        return Pipeline.MSE
    elif algorithm in [NoiseAlgorithm.NOISE_TO_VOID]:
        return Pipeline.MASK_MSE
    else:
        raise NotImplementedError("Algorithm does not have a default pipeline.")


def infer_blindspot(algorithm: NoiseAlgorithm):
    if algorithm in [
        NoiseAlgorithm.SELFSUPERVISED_DENOISING,
        NoiseAlgorithm.SELFSUPERVISED_DENOISING_MEAN_ONLY,
    ]:
        return True
    elif algorithm in [
        NoiseAlgorithm.NOISE_TO_NOISE,
        NoiseAlgorithm.NOISE_TO_CLEAN,
        NoiseAlgorithm.NOISE_TO_VOID,
    ]:
        return False
    else:
        raise NotImplementedError("Not known if algorithm requires blindspot.")


def infer(cfg: Dict, model_only: bool = False) -> Dict:
    if cfg.get(ConfigValue.PIPELINE, None) is None:
        cfg[ConfigValue.PIPELINE] = infer_pipeline(cfg[ConfigValue.ALGORITHM])
    if cfg.get(ConfigValue.BLINDSPOT, None) is None:
        cfg[ConfigValue.BLINDSPOT] = infer_blindspot(cfg[ConfigValue.ALGORITHM])

    if not model_only:
        infer_datasets(cfg)
    return cfg


def config_name(cfg: Dict) -> str:
    cfg = infer(cfg)
    config_lst = [cfg[ConfigValue.ALGORITHM].value]

    # Check if pipeline cannot be inferred
    inferred_pipeline = infer_pipeline(cfg[ConfigValue.ALGORITHM])
    if cfg[ConfigValue.PIPELINE] != inferred_pipeline:
        config_lst += [cfg[ConfigValue.PIPELINE].value + "_pipeline"]
    # Check if blindspot enable cannot be inferred
    inferred_blindspot = infer_blindspot(cfg[ConfigValue.ALGORITHM])
    if cfg[ConfigValue.BLINDSPOT] != inferred_blindspot:
        config_lst += [
            "blindspot" if cfg[ConfigValue.BLINDSPOT] else "blindspot_disabled"
        ]
    # Add noise information
    config_lst += [cfg[ConfigValue.NOISE_STYLE]]
    if cfg[ConfigValue.PIPELINE] in [Pipeline.SSDN]:
        config_lst += ["sigma_" + cfg[ConfigValue.NOISE_VALUE].value]

    if cfg[ConfigValue.IMAGE_CHANNELS] == 1:
        config_lst += ["mono"]
    if cfg[ConfigValue.PIPELINE] in [Pipeline.SSDN]:
        if cfg[ConfigValue.DIAGONAL_COVARIANCE]:
            config_lst += ["diag"]

    config_name = "-".join(config_lst)
    return config_name
