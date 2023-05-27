import PIL.Image
from PIL.Image import Image

import pathlib
from pathlib import Path

import numpy as np
from torchvision.transforms import CenterCrop, Compose, Resize
import fastai.vision.all as fv

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.Resampling.BICUBIC


def _convert_to_rgb(image: Image) -> Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _new_axis(a):
    return np.expand_dims(a, axis=0)


def _normalize_transform(mean, std):
    return lambda image: (image - mean) / std


def _preprocess_image(image: Image, image_size) -> np.ndarray:
    return Compose(
        [
            Resize(image_size, interpolation=BICUBIC),  # type: ignore
            CenterCrop(size=(image_size, image_size)),
            _convert_to_rgb,
            np.array,
            _new_axis,
            _normalize_transform(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )(image)


def preprocess(items: list[Image] | list[Path], image_size=256) -> np.ndarray:
    """
    Preprocesses list of image paths or list of PIL.Image objects.
    """
    item_types = {type(item) for item in items}
    if len(item_types) != 1:
        raise ValueError("All items must be either PIL.Image.Image or pathlib.Path.")
    t = item_types.pop()
    images: list[Image]
    match t:
        case pathlib.Path | pathlib.PosixPath | pathlib.WindowsPath:
            images = [PIL.Image.open(path) for path in items]  # type: ignore
        case PIL.Image.Image | fv.PILImage:
            images = items  # type: ignore
        case _:
            raise ValueError(f"t is of an Unexpected type {repr(t)}")

    return np.concatenate(
        [_preprocess_image(image, image_size) for image in images], axis=0
    )
