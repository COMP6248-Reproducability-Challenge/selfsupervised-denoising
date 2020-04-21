from enum import Enum, auto


class DataFormat:
    BHWC = "BHWC"
    BCHW = "BCHW"
    CHW = "CHW"
    HWC = "HWC"


class DataDim(Enum):
    BATCH = auto()
    CHANNEL = auto()
    WIDTH = auto()
    HEIGHT = auto()


BCHW_DIMS = {
    DataDim.BATCH: 0,
    DataDim.CHANNEL: 1,
    DataDim.HEIGHT: 2,
    DataDim.WIDTH: 3,
}

BHWC_DIMS = {
    DataDim.BATCH: 0,
    DataDim.CHANNEL: 3,
    DataDim.HEIGHT: 1,
    DataDim.WIDTH: 2,
}

CHW_DIMS = {
    DataDim.CHANNEL: 0,
    DataDim.HEIGHT: 1,
    DataDim.WIDTH: 2,
}

HWC_DIMS = {
    DataDim.CHANNEL: 2,
    DataDim.HEIGHT: 0,
    DataDim.WIDTH: 1,
}


DATA_FORMAT_DIMS = {
    DataFormat.BCHW: BCHW_DIMS,
    DataFormat.BHWC: BHWC_DIMS,
    DataFormat.CHW: CHW_DIMS,
    DataFormat.HWC: HWC_DIMS,
}
