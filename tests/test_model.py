# %%
import numpy as np
import fastai.vision.all as fv

from ncnn_clip.preprocess_clip import preprocess
from ncnn_clip import dummy_inputs



def batch_process_gt(images) -> np.ndarray:
    return np.concatenate(
        [
            np.expand_dims(preprocess_gt(image), axis=0)  # type: ignore
            for image in images
        ],
        axis=0,
    )


def test_preprocess_fuzz():
    dummy_images = dummy_inputs.get_random_images(100)
    ground_truth = batch_process_gt(dummy_images)
    processed_images = preprocess(dummy_images)

    assert np.allclose(
        processed_images,
        ground_truth,
    ), "preprocess() does not match preprocess_ground_truth()"


def test_preprocess_images():
    dummy_images = list(dummy_inputs.get_images(100))
    ground_truth = batch_process_gt(dummy_images)
    processed_images = preprocess(dummy_images)

    assert np.allclose(
        processed_images,
        ground_truth,
    ), "preprocess() does not match preprocess_ground_truth()"


def test_preprocess_fp():
    dummy_fps = dummy_inputs.get_image_fps(100)
    dummy_images = list(dummy_fps.map(fv.PILImage.create))

    ground_truth = batch_process_gt(dummy_images)
    processed_images = preprocess(list(dummy_fps))

    assert np.allclose(
        processed_images,
        ground_truth,
    ), "preprocess() does not match preprocess_ground_truth()"
