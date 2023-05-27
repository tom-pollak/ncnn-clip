import fastai.vision.all as fv

import numpy as np

from ncnn_clip.preprocess_clip import preprocess

IMAGE_FOLDER = fv.Path(__file__).parents[1] / "imagenet-sample-images"

def get_image_fps(nimages: int):
    return fv.get_image_files(IMAGE_FOLDER)[:nimages]


def get_images(nimages: int):
    return fv.get_image_files(IMAGE_FOLDER)[:nimages].map(fv.PILImage.create)


def get_random_images(nimages: int) -> list[fv.Image.Image]:
    dummy_input_np = np.random.randn(nimages, 3, 256, 256).astype(np.float32)
    return [fv.Image.fromarray(image, mode="RGB") for image in dummy_input_np]


def get_processed_images(nimages: int) -> np.ndarray:
    return preprocess(get_images(nimages))


def get_random_processed_images(nimages: int) -> np.ndarray:
    return preprocess(get_random_images(nimages))
