from typing import Callable

import open_clip
import torch
import torchvision

from ncnn_clip.utils import MODULE_ROOT

MODELS = {
    "convnext": "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg",
    "vit-b-16": "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "vit-b-32": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "vit-b-32-roberta": "hf-hub:laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
}


def load_open_clip_model(model_name: str, device="cpu") -> open_clip.CLIP:
    model: open_clip.CLIP
    model, _, _ = open_clip.create_model_and_transforms(MODELS[model_name])  # type: ignore
    model = model.to(device)
    model.eval()
    return model


def load_int8_image_encoder() -> torch.jit.ScriptModule:
    model = torch.jit.load(MODULE_ROOT / "models/gpu/convnext/int8_torchscript.pt")
    return model


def load_open_clip_eval_preprocess(model_name: str) -> torchvision.transforms.Compose:
    eval_preprocess: torchvision.transforms.Compose
    _, _, eval_preprocess = open_clip.create_model_and_transforms(MODELS[model_name])  # type: ignore
    return eval_preprocess


def load_open_clip_tokenizer(model_name: str) -> Callable[[list[str]], torch.Tensor]:
    return open_clip.get_tokenizer(MODELS[model_name])
