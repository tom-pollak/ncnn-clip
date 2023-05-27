from pathlib import Path
import PIL.Image

import numpy as np
from ncnn_clip.preprocess_clip import preprocess

IMAGE_FOLDER = Path(__file__).parent.parent / "imagenet-sample-images"


def get_image_files(image_folder: Path) -> list[Path]:
    assert (
        image_folder.exists() and image_folder.is_dir()
    ), f"{image_folder} is not a directory"
    return list(image_folder.glob("**/*.JPEG"))


def get_image_fps(nimages: int):
    return get_image_files(IMAGE_FOLDER)[:nimages]


def get_images(nimages: int) -> list[PIL.Image.Image]:
    dummy_fps = get_image_fps(nimages)
    return list(map(PIL.Image.open, dummy_fps))


def get_random_images(nimages: int) -> list[PIL.Image.Image]:
    dummy_input_np = np.random.randn(nimages, 256, 256, 3).astype(np.float32)
    return [PIL.Image.fromarray(image, mode="RGB") for image in dummy_input_np]


def get_processed_images(nimages: int) -> np.ndarray:
    return preprocess(get_images(nimages))


def get_random_processed_images(nimages: int) -> np.ndarray:
    return preprocess(get_random_images(nimages))
