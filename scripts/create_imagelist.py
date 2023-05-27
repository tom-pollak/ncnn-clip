import fastai.vision.all as fv

IMAGE_FOLDER = fv.Path(__file__).parents[2] / "imagenet-sample-images"

# needed for ncnn2table
IMAGE_LIST = fv.Path("imagelist.txt")
images = fv.get_image_files(IMAGE_FOLDER)
with open(IMAGE_LIST, "w") as image_paths:
    for image in images:
        image_paths.write(f"{IMAGE_FOLDER}/{image}\n")
