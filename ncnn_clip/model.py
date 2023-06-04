from __future__ import annotations
from pathlib import Path
from typing import Literal

import numpy as np
import ncnn
from numpy.typing import DTypeLike

MODEL_ROOT = Path(__file__).parent.parent / "models"


class NcnnCLIPModel:
    def __init__(
        self,
        param_path: Path,
        bin_path: Path,
        dtype,
    ):
        self.net = ncnn.Net()  # type: ignore
        self.dtype = dtype
        self.embed_dim = 640
        self.image_size = 256

        assert (
            param_path.exists() and bin_path.exists()
        ), "param or bin file does not exist"

        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))

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

    def image_forward(self, im: np.ndarray) -> np.ndarray:
        with self.net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(im))  # type: ignore
            _, out0 = ex.extract("out0")
        return out0

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[1:] == (
            3,
            self.image_size,
            self.image_size,
        ), f"input shape must be (B, 3, {self.image_size}, {self.image_size}), given: {x.shape}"
        out = np.empty((x.shape[0], self.embed_dim), dtype=self.dtype)
        for i, im in enumerate(x):
            out[i] = self.image_forward(im)
        return out
        # return np.apply_along_axis(self.image_forward, 0, x)

    def __call__(self, x):
        return self.forward(x)

    def __del__(self):
        self.net.clear()

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @staticmethod
    def load_model(
        model: Literal["convnext", "vit"],
        dtype: DTypeLike,
        device: Literal["cpu", "gpu"],
    ) -> NcnnCLIPModel:
        dtype = np.dtype(dtype)
        model_path = MODEL_ROOT / device / model
        assert model_path.exists(), f"{model_path} does not exist"
        match dtype:
            case np.float32:
                dtype_str = "fp32"
            case np.float16:
                dtype_str = "fp16"
            case np.int8:
                dtype_str = "int8"
            case _:
                raise ValueError(
                    f"unsupported dtype: {dtype}. Supported dtypes: fp32, fp16, int8"
                )

        return NcnnCLIPModel(
            param_path=model_path / f"{dtype_str}.param",
            bin_path=model_path / f"{dtype_str}.bin",
            dtype=dtype,
        )


# >>> import ncnn.model_zoo as model_zoo
# >>> model_zoo.get_model_list()


# import psutil

# def get_process_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     return mem_info.rss
