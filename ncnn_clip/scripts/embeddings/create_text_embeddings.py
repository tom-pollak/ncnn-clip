# %%
import torch
import os

from ncnn_clip.open_clip import load_open_clip_model, load_open_clip_tokenizer

# %%
image_folder = "imagenet-sample-images"
image_classes = [
    image.split("_", 1)[1].split(".")[0].replace("_", " ")
    for image in os.listdir(image_folder)
    if image.endswith(".JPEG")
]

# %%
model = load_open_clip_model("convnext")
tokenizer = load_open_clip_tokenizer("convnext")

# %%
label_prefix = "a photo of "
class_tokens = tokenizer([label_prefix + image_class for image_class in image_classes])

with torch.no_grad():
    class_features = model.encode_text(class_tokens)
    class_features /= class_features.norm(dim=-1, keepdim=True)

# %%
with open("assets/class_labels.txt", "w") as f:
    f.write("\n".join(image_classes))

torch.save(class_features, "assets/class_features.pt")

# %%
