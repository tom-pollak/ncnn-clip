from pathlib import Path

import torch
from torch import nn

from ncnn_clip import dummy_inputs
from ncnn_clip.open_clip import load_open_clip_model

MODEL_ARC = "convnext"
TORCHSCRIPT_SAVE_PATH = Path("models/torch/clip_convnext_torchscript.pt")


class ImageEncoderWrapper(nn.Module):
    def __init__(self, encode_image_func):
        super().__init__()
        self.encode_image_func = encode_image_func

    def forward(self, input):
        with torch.no_grad():
            return self.encode_image_func(input)


def export_torchscript(save_script=True) -> torch.ScriptModule:
    print(f"Starting TorchScript export of {MODEL_ARC} to {TORCHSCRIPT_SAVE_PATH}...")

    model = load_open_clip_model(MODEL_ARC)

    for param in model.parameters():
        param.requires_grad = False

    image_encoder = ImageEncoderWrapper(model.encode_image)
    image_encoder.eval()

    batch_size = 1
    dummy_input = dummy_inputs.get_processed_images(batch_size)
    print("Sanity check. Dummy output:", image_encoder(dummy_input).shape)  # works

    # could this be trace_module?
    mod: torch.ScriptModule = torch.jit.trace(image_encoder, dummy_input)  # type: ignore

    if save_script:
        mod.save(TORCHSCRIPT_SAVE_PATH)  # type: ignore

    print("Done!")
    return mod


if __name__ == "__main__":
    export_torchscript()
