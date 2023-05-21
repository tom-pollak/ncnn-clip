import os
from typing import Callable, Literal
import numpy as np
import open_clip
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from ncnn_clip.model import NcnnCLIPModel

# import psutil

# def get_process_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     return mem_info.rss


def load_image_model(
    model_type: Literal["torch", "ncnn_fp32", "ncnn_fp16", "ncnn_int8"]
) -> Callable[[torch.Tensor], torch.Tensor] | NcnnCLIPModel:
    if model_type == "torch":
        model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"
        )
        model.eval()
        return model.encode_image
    elif model_type == "ncnn_fp32":
        return NcnnCLIPModel(
            "models/clip_convnext.param",
            "models/clip_convnext.bin",
            dtype=np.float32,
        )
    elif model_type == "ncnn_fp16":
        return NcnnCLIPModel(
            "models/clip_convnext_fp16.param",
            "models/clip_convnext_fp16.bin",
            dtype=np.float16,
        )
    elif model_type == "ncnn_int8":
        return NcnnCLIPModel(
            "models/clip_convnext_int8.param",
            "models/clip_convnext_int8.bin",
            dtype=np.int8,
        )
    raise ValueError(f"Unknown model type: {model_type}")


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.Resampling.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _new_axis(t: torch.Tensor):
    return t[np.newaxis, :]


def preprocess(image: os.PathLike) -> torch.Tensor:
    return Compose(
        [
            Image.open,
            Resize(256, interpolation=BICUBIC),  # type: ignore
            CenterCrop(256),
            _convert_image_to_rgb,
            ToTensor(),
            _new_axis,
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )(image)
