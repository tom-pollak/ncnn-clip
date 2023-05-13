import numpy as np
from memory_profiler import profile
from ncnn_clip import NcnnCLIPModel


@profile
def run_inference():
    np.random.seed(0)
    dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float16)

    model = NcnnCLIPModel(
        "models/clip_convnext_fp16.param", "models/clip_convnext_fp16.bin"
    )
    out = model(dummy_input)
    return out


if __name__ == "__main__":
    out = run_inference()
    print(out.shape)
