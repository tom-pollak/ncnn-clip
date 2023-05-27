from pathlib import Path

from ncnn_clip.dummy_inputs import get_image_files

IMAGE_FOLDER = Path(__file__).parents[2] / "imagenet-sample-images"

# needed for ncnn2table
IMAGE_LIST = Path("imagelist.txt")
images = get_image_files(IMAGE_FOLDER)
with open(IMAGE_LIST, "w") as image_paths:
    for image in images:
        image_paths.write(f"{IMAGE_FOLDER}/{image}\n")
