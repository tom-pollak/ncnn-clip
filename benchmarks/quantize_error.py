# %%
from functools import partial
import numpy as np
import torch

from ncnn_clip import dummy_inputs
from ncnn_clip.model import NcnnCLIPModel
from ncnn_clip.open_clip import load_open_clip_model

# %%
model = load_open_clip_model("convnext")
ncnn_model = NcnnCLIPModel.load_model(np.float32)
ncnn_model_fp16 = NcnnCLIPModel.load_model(np.float16)
ncnn_model_int8 = NcnnCLIPModel.load_model(np.int8)

# %%
batch_size = 1
np.random.seed(0)
dummy_input = dummy_inputs.get_random_processed_images(batch_size)

# %%
torch_out = model.encode_image(torch.from_numpy(dummy_input)).detach().numpy()
ncnn_fp32_out = ncnn_model(dummy_input)
ncnn_fp16_out = ncnn_model_fp16(dummy_input.astype(np.float16))
ncnn_int8_out = ncnn_model_int8(dummy_input.astype(np.int8))


# %%
def rel_error(gt, pred):
    abs_error = np.abs(gt - pred)
    abs_gt = np.abs(gt) + 1e-8
    return (abs_error / abs_gt).mean()


torch_gt_rel_error = partial(gt=torch_out)

print(
    f"fp32: {torch_gt_rel_error(pred=ncnn_fp32_out):.4f}, fp16: {torch_gt_rel_error(pred=ncnn_fp16_out):.4f}, int8: {torch_gt_rel_error(pred=ncnn_int8_out):.4f}"
)

# import ncnn.model_zoo as model_zoo
# >>> model_zoo.get_model_list()

# %%
