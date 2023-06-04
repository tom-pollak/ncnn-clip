from pathlib import Path

import torch
from torch import nn

from ncnn_clip import dummy_inputs
from ncnn_clip.open_clip import load_open_clip_model

TORCHSCRIPT_SAVE_ROOT = Path("models")


class ImageEncoderWrapper(nn.Module):
    def __init__(self, encode_image_func):
        super().__init__()
        self.encode_image_func = encode_image_func

    def forward(self, input):
        with torch.no_grad():
            return self.encode_image_func(input)


def export_torchscript(model, device, save_script=True) -> torch.ScriptModule:
    print("Starting TorchScript export...")

    for param in model.parameters():
        param.requires_grad = False

    # TODO freeze batch norm?

    image_encoder = ImageEncoderWrapper(model.encode_image)
    image_encoder.eval()

    batch_size = 1
    dummy_input = torch.tensor(dummy_inputs.get_processed_images(batch_size))
    dummy_input = dummy_input.to(device)
    print("Sanity check. Dummy output:", image_encoder(dummy_input).shape)  # works

    # could this be trace_module?
    mod: torch.ScriptModule = torch.jit.trace(image_encoder, dummy_input)  # type: ignore

    if save_script:
        if device != "cpu":
            device = "gpu"

        save_path = TORCHSCRIPT_SAVE_ROOT / device / "torchscript.pt"
        assert save_path.parent.exists(), f"save path {save_path.parent} does not exist"
        mod.save(save_path)  # type: ignore
        print("Saved TorchScript to", save_path)

    return mod


if __name__ == "__main__":
    model = load_open_clip_model("convnext", device="cpu")
    export_torchscript(model, "cpu", save_script=True)
