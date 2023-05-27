import open_clip

MODELS = {"convnext": "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"}


def load_open_clip_model(model_name: str):
    model: open_clip.CLIP
    model, _, _ = open_clip.create_model_and_transforms(MODELS[model_name]) # type: ignore
    model.eval()
    return model


def load_open_clip_tokenizer(model_name: str):
    return open_clip.get_tokenizer(MODELS[model_name])
