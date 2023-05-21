import os
from pathlib import Path
from typing import Literal
import numpy as np
import ncnn


class NcnnCLIPModel:
    def __init__(
        self,
        param_path: os.PathLike | str,
        bin_path: os.PathLike | str,
        dtype,
    ):
        assert (
            Path(param_path).exists() and Path(bin_path).exists()
        ), "param or bin file does not exist"
        self.dtype = dtype
        self.embed_dim = 640

        self.net = ncnn.Net()
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        self.net.opt.lightmode = True
        self.net.opt.use_packing_layout = True
        if dtype == np.float16:
            self.net.opt.use_fp16_arithmetic = True
            self.net.opt.use_fp16_storage = True
            self.net.opt.use_fp16_packed = True
        if dtype == np.int8:
            self.net.opt.use_int8_storage = True
            self.net.opt.use_int8_arithmetic = True
            self.net.opt.use_int8_packed = True
        # print(self.net.opt.__dir__())

        self.net.opt.use_image_storage = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[1:] == (3, 256, 256), "input shape must be (X, 3, 256, 256)"
        out = np.empty((x.shape[0], self.embed_dim), dtype=self.dtype)
        for i, im in enumerate(x):
            with self.net.create_extractor() as ex:
                ex.input("in0", ncnn.Mat(im))
                _, out0 = ex.extract("out0")
                out[i] = out0
        return out

    def __call__(self, x):
        return self.forward(x)

    def __del__(self):
        self.net.clear()

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# >>> import ncnn.model_zoo as model_zoo
# >>> model_zoo.get_model_list()
