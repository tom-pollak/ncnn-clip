# %%
import open_clip
import numpy as np
import torch
from ncnn_clip.model import NcnnCLIPModel
from PIL import Image

# %%
model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"
)
model.eval()

ncnn_model = NcnnCLIPModel("models/clip_convnext.param", "models/clip_convnext.bin")

ncnn_model_fp16 = NcnnCLIPModel(
    "models/clip_convnext_fp16.param", "models/clip_convnext_fp16.bin"
)

ncnn_model_int8 = NcnnCLIPModel(
    "models/clip_convnext_int8.param", "models/clip_convnext_int8.bin"
)

# %%
batch_size = 1
with open("imagelist.txt", "r") as image_paths:
    images = image_paths.read().splitlines()[:batch_size]
images = [Image.open(image) for image in images]
# %%

dummy_input = preprocess(images[0]).unsqueeze(0)
dummy_input_np = dummy_input.numpy().squeeze()
dummy_input_np_fp16 = dummy_input_np.astype(np.float16)
dummy_input_np_int8 = dummy_input_np.astype(np.int8)

# %%
# torch.manual_seed(0)
# dummy_input = torch.rand(1, 3, 256, 256, dtype=torch.float32)
# dummy_input_np = dummy_input.numpy().squeeze()
# dummy_input_np_fp16 = dummy_input_np.astype(np.float16)
# dummy_input_np_int8 = dummy_input_np.astype(np.int8)

# %%
torch_out = model.encode_image(dummy_input).detach().numpy()
ncnn_out = ncnn_model(dummy_input_np)
ncnn_fp16_out = ncnn_model_fp16(dummy_input_np_fp16)
ncnn_int8_out = ncnn_model_int8(dummy_input_np_int8)

# %%
abs_torch_out = np.abs(torch_out) + 1e-8

abs_error_fp32 = np.abs(torch_out - ncnn_out)
abs_error_fp16 = np.abs(torch_out - ncnn_fp16_out)
abs_error_int8 = np.abs(torch_out - ncnn_int8_out)

rel_error_fp32 = (abs_error_fp32 / abs_torch_out).mean()
rel_error_fp16 = (abs_error_fp16 / abs_torch_out).mean()
rel_error_int8 = (abs_error_int8 / abs_torch_out).mean()

print(f"fp32: {rel_error_fp32}, fp16: {rel_error_fp16}, int8: {rel_error_int8}")

# import ncnn.model_zoo as model_zoo
# >>> model_zoo.get_model_list()

# %%
