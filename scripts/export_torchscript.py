# %%
import fastai.vision.all as fv
import torch
from torch import nn

from ncnn_clip import dummy_inputs
from ncnn_clip.open_clip import load_open_clip_model, load_open_clip_tokenizer

MODEL_ARC = "convnext"
TORCHSCRIPT_SAVE_PATH = fv.Path("models/torch/clip_convnext_torchscript.pt")

print(f"Starting TorchScript export of {MODEL_ARC} to {TORCHSCRIPT_SAVE_PATH}...")

# %%
model = load_open_clip_model(MODEL_ARC)
tokenizer = load_open_clip_tokenizer(MODEL_ARC)

for param in model.parameters():
    param.requires_grad = False


# %%
class ImageEncoderWrapper(nn.Module):
    def __init__(self, encode_image_func):
        super().__init__()
        self.encode_image_func = encode_image_func

    def forward(self, input):
        with torch.no_grad():
            return self.encode_image_func(input)


image_encoder = ImageEncoderWrapper(model.encode_image)
image_encoder.eval()

# %%
batch_size = 1
dummy_input = dummy_inputs.get_processed_images(batch_size)
print("Sanity check. Dummy output:", image_encoder(dummy_input).shape)  # works

# %%
# could this be trace_module?
mod = torch.jit.trace(image_encoder, dummy_input)  # type: ignore
mod.save(TORCHSCRIPT_SAVE_PATH)  # type: ignore

print("Done!")
