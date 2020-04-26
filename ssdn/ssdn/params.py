from enum import Enum, auto


class NoiseAlgorithm(Enum):
    SELFSUPERVISED_DENOISING = "ssdn"
    SELFSUPERVISED_DENOISING_MEAN_ONLY = "ssdn_u_only"
    NOISE_TO_NOISE = "n2n"
    NOISE_TO_CLEAN = "n2c"
    NOISE_TO_VOID = "n2v"  # Unsupported


class NoiseValue(Enum):
    CONSTANT_UNKNOWN = "constant_unknown"
    VARIABLE_UNKNOWN = "variable_unknown"
    KNOWN = "known"


class Pipeline(Enum):
    MSE = auto()
    SSDN = auto()


class ConfigValue(Enum):
    # Denoiser Modes
    ALGORITHM = auto()
    BLINDSPOT = auto()
    PIPELINE = auto()
    IMAGE_CHANNELS = auto()

    NOISE_STYLE = auto()

    LEARNING_RATE = auto()
    LR_RAMPUP_FRACTION = auto()
    LR_RAMPDOWN_FRACTION = auto()
    # SSDN Pipeline Configuration
    NOISE_VALUE_STATE = auto()
    DIAGONAL_COVARIANCE = auto()
    # Training Loop Configuration
    EVAL_INTERVAL = auto()
    PRINT_INTERVAL = auto()
    TRAIN_ITERATIONS = auto()

    DATALOADER_WORKERS = auto()
    TRAIN_DATASET_TYPE = auto()
    TRAIN_DATA_PATH = auto()
    TRAIN_PATCH_SIZE = auto()
    TRAIN_MINIBATCH_SIZE = auto()

    TEST_DATASET_TYPE = auto()
    TEST_DATA_PATH = auto()
    TEST_MINIBATCH_SIZE = auto()
    PIN_DATA_MEMORY = auto()


class DatasetType(Enum):
    HDF5 = auto()
    FOLDER = auto()


class StateValue(Enum):
    INITIALISED = auto()
    EVAL_MODE = auto()

    INPUT = auto()
    ITERATION = auto()
    REFERENCE = auto()
    HISTORY = auto()


class HistoryValue(Enum):
    PSNR = auto()
    LOSS = auto()


class DenoiserMode(Enum):
    TRAIN = auto()
    EVAL = auto()


class PipelineOutput(Enum):
    INPUTS = auto()
    IMG_DENOISED = auto()
    IMG_MU = auto()
    IMG_PME = auto()
    LOSS = auto()
    NOISE_STD_DEV = auto()
