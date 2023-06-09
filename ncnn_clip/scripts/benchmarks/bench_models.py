import time
import numpy as np

import torch

from ncnn_clip import NcnnCLIPModel, dummy_inputs
from ncnn_clip.open_clip import load_open_clip_model

NIMAGES = 100


def run_inference(model, input):
    start_time = time.perf_counter()
    out = model(input)
    end_time = time.perf_counter()
    print(f"Time per iteration: {(end_time - start_time) / NIMAGES:.3f} seconds")
    return out


def bench_models():
    print(f"Benching inference speed for {NIMAGES} images:")
    np.random.seed(0)
    dummy_input = dummy_inputs.get_random_processed_images(NIMAGES)

    model_torch = load_open_clip_model("convnext").encode_image
    _ = run_inference(model_torch, torch.from_numpy(dummy_input).to(torch.float32))

    model_fp32 = NcnnCLIPModel.load_model("convnext", np.float32, "cpu")
    _ = run_inference(model_fp32, dummy_input.astype(np.float32))

    model_fp16 = NcnnCLIPModel.load_model("convnext", np.float16, "cpu")
    _ = run_inference(model_fp16, dummy_input.astype(np.float32)) # still takes fp32 input

    # model_int8 = NcnnCLIPModel.load_model(np.int8)
    # _ = run_inference(model_int8, dummy_input.astype(np.int8))


if __name__ == "__main__":
    bench_models()
