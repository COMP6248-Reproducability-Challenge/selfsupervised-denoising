import torch
import os
import sys

from pathlib import Path
from ssdn.utils import MetricDict
from ssdn.params import HistoryValue, StateValue
OrderedDefaultDict = MetricDict

if __name__ == "__main__":
    """Some old training files were created using a class that appeared in the __main__
    namespace, this breaks unpickling when that class is not found in __main__
    Run this to replace the reference. All training files will be found recursively
    from the provided root directory
    """
    if len(sys.argv) < 2:
        raise ValueError("Expected root path argument")

    root = sys.argv[1]
    for path in Path(root).rglob('*.training'):
        print(path)
        state_dict = torch.load(path, map_location="cpu")
        history = state_dict["state"][StateValue.HISTORY]
        history[HistoryValue.TRAIN] = MetricDict(history[HistoryValue.TRAIN])
        history[HistoryValue.EVAL] = MetricDict(history[HistoryValue.EVAL])
        bak_path = str(path) + "_bak"
        os.rename(path, bak_path)
        torch.save(state_dict, path)
        os.remove(bak_path)
