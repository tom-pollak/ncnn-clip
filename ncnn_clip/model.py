from __future__ import annotations
from pathlib import Path

import fastai.vision.all as fv
import numpy as np
import ncnn

MODEL_ROOT = fv.Path(__file__).parents[1] / "models/ncnn/"


class NcnnCLIPModel:
    def __init__(
        self,
        param_path: Path,
        bin_path: Path,
        dtype,
    ):
        assert (
            Path(param_path).exists() and Path(bin_path).exists()
        ), "param or bin file does not exist"
        self.dtype = dtype
        self.embed_dim = 640
        self.image_size = 256

        self.net = ncnn.Net()  # type: ignore
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
        assert x.shape[1:] == (
            3,
            self.image_size,
            self.image_size,
        ), f"input shape must be (B, 3, {self.image_size}, {self.image_size})"
        out = np.empty((x.shape[0], self.embed_dim), dtype=self.dtype)
        for i, im in enumerate(x):
            with self.net.create_extractor() as ex:
                ex.input("in0", ncnn.Mat(im))  # type: ignore
                _, out0 = ex.extract("out0")
                out[i] = out0
        return out

    def __call__(self, x):
        return self.forward(x)

    def __del__(self):
        self.net.clear()

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @staticmethod
    def load_model(dtype) -> NcnnCLIPModel:
        match dtype:
            case np.float32:
                return NcnnCLIPModel(
                    MODEL_ROOT / "clip_convnext.param",
                    MODEL_ROOT / "clip_convnext.bin",
                    dtype=dtype,
                )
            case np.float16:
                return NcnnCLIPModel(
                    MODEL_ROOT / "clip_convnext_fp16.param",
                    MODEL_ROOT / "clip_convnext_fp16.bin",
                    dtype=dtype,
                )
            case np.int8:
                return NcnnCLIPModel(
                    MODEL_ROOT / "clip_convnext_int8.param",
                    MODEL_ROOT / "clip_convnext_int8.bin",
                    dtype=dtype,
                )
            case _:
                raise ValueError(f"unsupported dtype: {dtype}")


# >>> import ncnn.model_zoo as model_zoo
# >>> model_zoo.get_model_list()


# import psutil

# def get_process_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     return mem_info.rss
