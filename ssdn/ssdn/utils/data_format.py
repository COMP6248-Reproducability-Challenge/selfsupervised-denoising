import ssdn.utils.utils as utils

from enum import Enum, auto
from collections import OrderedDict
from typing import Dict, Tuple


class DataFormat:
    BHWC = "BHWC"
    BWHC = "BWHC"
    BCHW = "BCHW"
    BCWH = "BCWH"
    CHW = "CHW"
    CWH = "CWH"
    HWC = "HWC"
    WHC = "WHC"


PIL_FORMAT = DataFormat.CWH
PIL_BATCH_FORMAT = DataFormat.BCWH
""" Formats used by Pillow/PIL.
"""


class DataDim(Enum):
    BATCH = auto()
    CHANNEL = auto()
    WIDTH = auto()
    HEIGHT = auto()


DIM_CHAR_DICT = {
    DataDim.BATCH: "B",
    DataDim.CHANNEL: "C",
    DataDim.HEIGHT: "H",
    DataDim.WIDTH: "W",
}
""" Enumeration association to char representations.
"""

CHAR_DIM_DICT = dict((v, k) for k, v in DIM_CHAR_DICT.items())
""" Character association to enumeration representations.
"""

DATA_FORMAT_INDEX_DIM = {}
""" Storage for pre-defined dimension format dictionaries that map
axis index to dimension type.
"""

DATA_FORMAT_DIM_INDEX = {}
""" Storage for pre-defined dimension format dictionaries that map
dimension type to axis index.
"""


def make_index_dim_dict(data_format: str) -> Dict:
    dim_dict = OrderedDict()
    for i, c in enumerate(data_format):
        dim_dict[i] = CHAR_DIM_DICT[c]
    return dim_dict


def make_dim_index_dict(data_format: str) -> Dict:
    dim_dict = OrderedDict()
    for i, c in enumerate(data_format):
        dim_dict[CHAR_DIM_DICT[c]] = i
    return dim_dict


def add_format(data_format: str):
    global DATA_FORMAT_INDEX_DIM
    DATA_FORMAT_INDEX_DIM[data_format] = make_index_dim_dict(data_format)
    global DATA_FORMAT_DIM_INDEX
    DATA_FORMAT_DIM_INDEX[data_format] = make_dim_index_dict(data_format)


# Create dictionary entries for all formats in DataFormat class
for data_format in utils.list_constants(DataFormat):
    add_format(data_format)


def permute_tuple(cur: str, target: str) -> Tuple[int]:
    assert sorted(cur) == sorted(target)

    # Ensure reference dictionaries exist
    if cur not in DATA_FORMAT_INDEX_DIM:
        add_format(cur)
    if target not in DATA_FORMAT_DIM_INDEX:
        add_format(target)

    dims_cur = DATA_FORMAT_DIM_INDEX[cur]
    dims_target = DATA_FORMAT_DIM_INDEX[target]
    transpose = [dims_cur[target] for target in dims_target.keys()]
    return tuple(transpose)
