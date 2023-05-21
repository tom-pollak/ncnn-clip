# %%
from ncnn_clip import preprocess, load_image_model
import os
import torch
import numpy as np

# %%
image_folder = "imagenet-sample-images"
images = [
    preprocess(image_folder + "/" + image)  # type: ignore
    for image in os.listdir(image_folder)
    if image.endswith(".JPEG")
]
images = torch.cat(tuple(images), dim=0)
images_np: np.ndarray = images.numpy()

# %%
# model_torch = load_image_model("torch")
# torch_image_features: torch.Tensor = model_torch(images)  # type: ignore
# torch_image_features /= torch_image_features.norm(dim=-1, keepdim=True)
# torch.save(torch_image_features, "assets/torch_image_features.pt")
# del torch_image_features, model_torch, images

# model_fp32 = load_image_model("ncnn_fp32")
# fp32_image_features: np.ndarray = model_fp32(images_np.astype(np.float32))  # type: ignore
# fp32_image_features /= np.linalg.norm(fp32_image_features, axis=-1, keepdims=True)
# np.save("assets/fp32_image_features.npy", fp32_image_features)

# %%
model_fp16 = load_image_model("ncnn_fp16")
fp16_image_features: np.ndarray = model_fp16(images_np.astype(np.float16))  # type: ignore
fp16_image_features /= np.linalg.norm(fp16_image_features, axis=-1, keepdims=True)
np.save("assets/fp16_image_features.npy", fp16_image_features)
del fp16_image_features, model_fp16

model_int8 = load_image_model("ncnn_int8")
int8_image_features: np.ndarray = model_int8(images_np.astype(np.int8))  # type: ignore
int8_image_features = (
    int8_image_features.astype(np.float32)
    / np.linalg.norm(int8_image_features.astype(np.float32), axis=-1, keepdims=True)
).astype(np.int8)
np.save("assets/int8_image_features.npy", int8_image_features)
del int8_image_features, model_int8

# %%
