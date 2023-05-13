from os import PathLike
from pathlib import Path
import numpy as np
import ncnn


class NcnnCLIPModel:
    def __init__(self, param_path: PathLike | str, bin_path: PathLike | str):
        assert (
            Path(param_path).exists() and Path(bin_path).exists()
        ), "param or bin file does not exist"
        self.net = ncnn.Net()
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.squeeze(0)))
            _, out0 = ex.extract("out0")
        return np.array(out0)

    def __del__(self):
        self.net.clear()


# >>> import ncnn.model_zoo as model_zoo
# >>> model_zoo.get_model_list()
