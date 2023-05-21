# %%
import open_clip
import torch
import os

# %%
image_folder = "imagenet-sample-images"
image_classes = [
    image.split("_", 1)[1].split(".")[0].replace("_", " ")
    for image in os.listdir(image_folder)
    if image.endswith(".JPEG")
]

# %%
model, _, _ = open_clip.create_model_and_transforms(
    "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"
)
tokenizer = open_clip.get_tokenizer(
    "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"
)

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
