from __future__ import annotations

from enum import Enum, auto
from typing import List


class NoiseAlgorithm(Enum):
    SELFSUPERVISED_DENOISING = "ssdn"
    SELFSUPERVISED_DENOISING_MEAN_ONLY = "ssdn_u_only"
    NOISE_TO_NOISE = "n2n"
    NOISE_TO_CLEAN = "n2c"
    NOISE_TO_VOID = "n2v"  # Unsupported


class NoiseValue(Enum):
    UNKNOWN_CONSTANT = "const"
    UNKNOWN_VARIABLE = "var"
    KNOWN = "known"


class Pipeline(Enum):
    MSE = "mse"
    SSDN = "ssdn"
    MASK_MSE = "mask_mse"


class Blindspot(Enum):
    ENABLED = "blindspot"
    DISABLED = "normal"


class ConfigValue(Enum):
    INFER_CFG = auto()
    ALGORITHM = auto()
    BLINDSPOT = auto()
    PIPELINE = auto()
    IMAGE_CHANNELS = auto()

    NOISE_STYLE = auto()

    LEARNING_RATE = auto()
    LR_RAMPUP_FRACTION = auto()
    LR_RAMPDOWN_FRACTION = auto()

    NOISE_VALUE = auto()
    DIAGONAL_COVARIANCE = auto()

    EVAL_INTERVAL = auto()
    PRINT_INTERVAL = auto()
    SNAPSHOT_INTERVAL = auto()
    TRAIN_ITERATIONS = auto()

    DATALOADER_WORKERS = auto()
    TRAIN_DATASET_NAME = auto()
    TRAIN_DATASET_TYPE = auto()
    TRAIN_DATA_PATH = auto()
    TRAIN_PATCH_SIZE = auto()
    TRAIN_MINIBATCH_SIZE = auto()

    TEST_DATASET_NAME = auto()
    TEST_DATASET_TYPE = auto()
    TEST_DATA_PATH = auto()
    TEST_MINIBATCH_SIZE = auto()
    PIN_DATA_MEMORY = auto()


class DatasetType(Enum):
    HDF5 = auto()
    FOLDER = auto()


class StateValue(Enum):
    INITIALISED = auto()
    MODE = auto()

    ITERATION = auto()
    REFERENCE = auto()
    HISTORY = auto()


class HistoryValue(Enum):
    TRAIN = auto()
    EVAL = auto()
    TIMINGS = auto()


class PipelineOutput(Enum):
    INPUTS = auto()
    LOSS = "loss"
    IMG_DENOISED = "out"
    IMG_MU = "out_mu"
    NOISE_STD_DEV = "noise_std"
    MODEL_STD_DEV = "model_std"
