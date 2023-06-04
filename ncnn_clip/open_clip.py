from typing import Callable

import open_clip
import torch
import torchvision

MODELS = {
    "convnext": "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg",
    "vit": "",
}


def load_open_clip_model(model_name: str, device="cpu") -> open_clip.CLIP:
    model: open_clip.CLIP
    model, _, _ = open_clip.create_model_and_transforms(MODELS[model_name])  # type: ignore
    model = model.to(device)
    model.eval()
    return model


def load_open_clip_eval_preprocess(model_name: str) -> torchvision.transforms.Compose:
    eval_preprocess: torchvision.transforms.Compose
    _, _, eval_preprocess = open_clip.create_model_and_transforms(MODELS[model_name])  # type: ignore
    return eval_preprocess


def load_open_clip_tokenizer(model_name: str) -> Callable[[list[str]], torch.Tensor]:
    return open_clip.get_tokenizer(MODELS[model_name])
