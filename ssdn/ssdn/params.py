from enum import Enum, auto


class NoiseAlgorithm(Enum):
    SELFSUPERVISED_DENOISING = "ssdn"
    SELFSUPERVISED_DENOISING_MEAN_ONLY = "ssdn_u_only"
    NOISE_TO_NOISE = "n2n"
    NOISE_TO_CLEAN = "n2c"
    NOISE_TO_VOID = "n2v"  # Unsupported


class NoiseValue(Enum):
    UNKNOWN = "unknown"
    KNOWN_GLOBAL = "known_global"
    KNOWN_PER_IMAGE = "known_per_image"


class Pipeline(Enum):
    MSE = auto()
    SSDN = auto()

class ConfigValue(Enum):
    ALGORITHM = auto()
    BLINDSPOT = auto()
    NOISE_STYLE = auto()
    DIAGONAL_COVARIANCE = auto()
    TRAINING_PATCH_SIZE = auto()

    MINIBATCH_SIZE = auto()
    LEARNING_RATE = auto()
    LR_RAMPUP_FRACTION = auto()
    LR_RAMPDOWN_FRACTION = auto()
    PIPELINE = auto()
    NOISE_VALUE_STATE = auto()

    EVAL_INTERVAL = auto()
    PRINT_INTERVAL = auto()
    TRAINING_ITERATIONS = auto()


class StateValue(Enum):
    INPUT = auto()
    ITERATION = auto()
    REFERENCE = auto()
    LOSS_HISTORY = auto()
    TEST_HISTORY = auto()


class PipelineOutput(Enum):
    INPUTS = auto()
    IMG_DENOISED = auto()
    IMG_MU = auto()
    IMG_PME = auto()
    LOSS = auto()
    NOISE_STD_DEV = auto()
