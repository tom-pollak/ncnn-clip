import time
import numpy as np
import torch
from ncnn_clip.model import load_image_model


def run_inference(
    model,
    dtype,
    iterations: int = 100,
    use_batch: bool = False,
    use_torch: bool = False,
):
    np.random.seed(0)
    if use_torch:
        dummy_input = torch.randn(iterations, 3, 256, 256, dtype=dtype)
    else:
        dummy_input = np.random.randn(iterations, 3, 256, 256).astype(dtype)
    out = np.empty((iterations, 640), dtype=np.float16)

    start_time = time.perf_counter()
    if use_batch:
        out = model(dummy_input)
    else:
        for i, _input in enumerate(dummy_input):
            out[i] = model(_input)
    end_time = time.perf_counter()
    print(f"Time per iteration: {(end_time - start_time) / iterations:.3f} seconds\n")
    return out


if __name__ == "__main__":
    model = load_image_model("torch")
    out = run_inference(
        model, torch.float32, use_batch=True, use_torch=True
    )

    model_fp32 = load_image_model("ncnn_fp32")
    _ = run_inference(model_fp32, np.float32)

    model_fp16 = load_image_model("ncnn_fp16")
    _ = run_inference(model_fp16, np.float16)

    model_int8 = load_image_model("ncnn_int8")
    _ = run_inference(model_int8, np.int8)
