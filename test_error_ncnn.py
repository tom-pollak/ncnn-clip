# %%
import open_clip
import numpy as np
import torch
from ncnn_clip import NcnnCLIPModel
from memory_profiler import profile


# %%
torch.manual_seed(0)
dummy_input = torch.rand(1, 3, 256, 256, dtype=torch.float)
dummy_input_np = dummy_input.numpy()


# %%
@profile
def run_models():
    # 1000 Mib
    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"
    )
    model.eval()

    # 203 Mib
    ncnn_model = NcnnCLIPModel("models/clip_convnext.param", "models/clip_convnext.bin")

    # 184 Mib
    ncnn_model_fp16 = NcnnCLIPModel(
        "models/clip_convnext_fp16.param", "models/clip_convnext_fp16.bin"
    )

    # %%
    torch_out = model.encode_image(dummy_input).detach().numpy()
    ncnn_out = ncnn_model(dummy_input_np)
    ncnn_fp16_out = ncnn_model_fp16(dummy_input_np)
    return torch_out, ncnn_out, ncnn_fp16_out

torch_out, ncnn_out, ncnn_fp16_out = run_models()
# %%
abs_torch_out = np.abs(torch_out) + 1e-8

abs_error_fp32 = np.abs(torch_out - ncnn_out)
abs_error_fp16 = np.abs(torch_out - ncnn_fp16_out)

rel_error_fp32 = (abs_error_fp32 / abs_torch_out).mean()
rel_error_fp16 = (abs_error_fp16 / abs_torch_out).mean()

print(f"fp32: {rel_error_fp32}, fp16: {rel_error_fp16}")

# import ncnn.model_zoo as model_zoo
# >>> model_zoo.get_model_list()

# %%
