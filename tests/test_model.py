# %%
import PIL.Image

import torch
import numpy as np

from ncnn_clip.preprocess_clip import preprocess
from ncnn_clip import dummy_inputs
from ncnn_clip.open_clip import load_open_clip_eval_preprocess

preprocess_gt = load_open_clip_eval_preprocess("convnext")


def batch_process_gt(images) -> np.ndarray:
    return torch.cat(
        [preprocess_gt(image).unsqueeze(0) for image in images], dim=0
    ).numpy()


def test_preprocess_fuzz():
    dummy_images = dummy_inputs.get_random_images(100)
    ground_truth = batch_process_gt(dummy_images)
    processed_images = preprocess(dummy_images)

    assert np.allclose(
        processed_images,
        ground_truth,
    ), "preprocess() does not match preprocess_ground_truth()"


def test_preprocess_images():
    dummy_images = dummy_inputs.get_images(100)
    ground_truth = batch_process_gt(dummy_images)
    processed_images = preprocess(dummy_images)
    print(processed_images.shape, ground_truth.shape)

    assert np.allclose(
        processed_images,
        ground_truth,
    ), "preprocess() does not match preprocess_ground_truth()"


def test_preprocess_fp():
    dummy_fps = dummy_inputs.get_image_fps(100)
    dummy_images = list(map(PIL.Image.open, dummy_fps))

    ground_truth = batch_process_gt(dummy_images)
    processed_images = preprocess(list(dummy_fps))

    assert np.allclose(
        processed_images,
        ground_truth,
    ), "preprocess() does not match preprocess_ground_truth()"
