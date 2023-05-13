# %%
import torch
import torch.onnx
from torch import nn
import open_clip
from PIL import Image
import skimage
import os

# %%
model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"
)
tokenizer = open_clip.get_tokenizer(
    "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"
)

model.eval()
for param in model.parameters():
    param.requires_grad = False

# %%
image = Image.open(
    os.path.join(skimage.data_dir, os.listdir(skimage.data_dir)[0])
).convert("RGB")
dummy_input = preprocess(image).unsqueeze(0)


# %%
class ImageEncoderWrapper(nn.Module):
    def __init__(self, encode_image_func):
        super().__init__()
        self.encode_image_func = encode_image_func

    def forward(self, input):
        with torch.no_grad():
            return self.encode_image_func(input)


# %%
image_encoder = ImageEncoderWrapper(model.encode_image)
image_encoder.eval()

# %%
image_encoder(dummy_input);
# works

# %%
mod = torch.jit.trace(image_encoder, dummy_input)
mod.save("models/clip_convnext.pt")

# %%
