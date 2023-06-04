# %%
from pathlib import Path

from ncnn_clip import dummy_inputs, NcnnCLIPModel
from ncnn_clip.open_clip import load_open_clip_model
import torch
import numpy as np

MODEL_ARCH = "convnext"
DEVICE = "cpu"
TORCH_OUT = Path("assets/torch_image_features.pt")
FP32_OUT = Path("assets/fp32_image_features.npy")
FP16_OUT = Path("assets/fp16_image_features.npy")
INT8_OUT = Path("assets/int8_image_features.npy")

# %%
images = dummy_inputs.get_processed_images(100)

# %%
model_torch = load_open_clip_model(MODEL_ARCH)
torch_image_features: torch.Tensor = model_torch.encode_image(torch.from_numpy(images))
torch_image_features /= torch_image_features.norm(dim=-1, keepdim=True)
torch.save(torch_image_features, TORCH_OUT)

# %%
model_fp32 = NcnnCLIPModel.load_model(MODEL_ARCH, np.float32, DEVICE)
fp32_image_features: np.ndarray = model_fp32(images.astype(np.float32))
fp32_image_features /= np.linalg.norm(fp32_image_features, axis=-1, keepdims=True)
np.save(FP32_OUT, fp32_image_features)

model_fp16 = NcnnCLIPModel.load_model(MODEL_ARCH, np.float16, DEVICE)
fp16_image_features: np.ndarray = model_fp16(images.astype(np.float16))
fp16_image_features /= np.linalg.norm(fp16_image_features, axis=-1, keepdims=True)
np.save(FP16_OUT, fp16_image_features)

model_int8 = NcnnCLIPModel.load_model(MODEL_ARCH, np.int8, DEVICE)
int8_image_features: np.ndarray = model_int8(images.astype(np.int8))
int8_image_features = (
    int8_image_features.astype(np.float32)
    / np.linalg.norm(int8_image_features.astype(np.float32), axis=-1, keepdims=True)
).astype(np.int8)
np.save(INT8_OUT, int8_image_features)
