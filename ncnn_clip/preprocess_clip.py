from typing import Callable
import PIL.Image
from PIL.Image import Image

import pathlib
from pathlib import Path

import numpy as np
from torchvision.transforms import CenterCrop, Compose, Resize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.Resampling.BICUBIC


def _convert_to_rgb(image: Image) -> Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _to_ndarray(img: Image) -> np.ndarray:
    norm_img = np.asarray(img).astype(np.float32) / 255
    return norm_img.transpose(2, 0, 1)


def _normalize_transform(mean, std) -> Callable[[np.ndarray], np.ndarray]:
    mean = np.array(mean)[:, None, None]
    std = np.array(std)[:, None, None]

    def norm(image):
        return (image - mean) / std

    return norm


def _new_axis(a):
    return np.expand_dims(a, axis=0)


def _preprocess_image(image: Image, image_size) -> np.ndarray:
    return Compose(
        [
            Resize(image_size, interpolation=BICUBIC),  # type: ignore
            CenterCrop(size=(image_size, image_size)),
            _convert_to_rgb,
            _to_ndarray,
            _normalize_transform(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
            _new_axis,
        ]
    )(image)


def preprocess(
    items: list[Image] | list[Path] | np.ndarray, image_size=256
) -> np.ndarray:
    """
    Preprocesses list of image paths or list of PIL.Image objects.
    """
    if isinstance(items, np.ndarray):
        assert items.ndim == 4
    item_types = {type(item) for item in items}
    if len(item_types) != 1:
        raise ValueError("All items must be either PIL.Image.Image or pathlib.Path.")
    t = item_types.pop()

    if issubclass(t, pathlib.Path):
        images = [PIL.Image.open(path) for path in items]  # type: ignore
    elif issubclass(t, np.ndarray):
        images = [PIL.Image.fromarray(image) for image in items]
    elif issubclass(t, PIL.Image.Image):
        images = items  # type: ignore
    else:
        raise ValueError(f"t is of an Unexpected type {repr(t)}")

    images: list[Image]
    return np.concatenate(
        [_preprocess_image(image, image_size) for image in images], axis=0
    ).astype(np.float32)
