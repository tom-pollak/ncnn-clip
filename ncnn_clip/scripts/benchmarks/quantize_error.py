from functools import partial
import numpy as np
import torch

from ncnn_clip import dummy_inputs
from ncnn_clip.model import NcnnCLIPModel
from ncnn_clip.open_clip import load_open_clip_model, load_int8_image_encoder


def quantize_error():
    model = load_open_clip_model("convnext")
    # int8_model = load_int8_image_encoder()
    ncnn_model = NcnnCLIPModel.load_model("convnext", np.float32, "cpu")
    ncnn_model_fp16 = NcnnCLIPModel.load_model("convnext", np.float16, "cpu")
    ncnn_model_int8 = NcnnCLIPModel.load_model("convnext", np.int8, "cpu")

    batch_size = 5
    np.random.seed(0)
    dummy_input = dummy_inputs.get_random_processed_images(batch_size)

    torch_gt = (
        model.encode_image(torch.from_numpy(dummy_input).to(torch.float32))
        .detach()
        .numpy()
    )
    # torch_int8_gt = int8_model(torch.from_numpy(dummy_input).to(torch.float32))

    ncnn_fp32_out = ncnn_model(dummy_input.astype(np.float32))
    ncnn_fp16_out = ncnn_model_fp16(
        dummy_input.astype(np.float32)
    )  # still takes fp32 input
    ncnn_int8_out = ncnn_model_int8(dummy_input.astype(np.float32))

    random_out = np.random.randn(*torch_gt.shape) * torch_gt.std() + torch_gt.mean()

    def rel_error(gt, pred):
        abs_error = np.abs(gt - pred)
        abs_gt = np.abs(gt) + 1e-8
        return (abs_error / abs_gt).mean()

    torch_gt_rel_error = partial(rel_error, gt=torch_gt)
    float32_rel_err = torch_gt_rel_error(pred=ncnn_fp32_out)
    float16_rel_err = torch_gt_rel_error(pred=ncnn_fp16_out)
    random_rel_err = torch_gt_rel_error(pred=random_out)

    # torch_int8_rel_error = rel_error(gt=torch_int8_gt, pred=torch_int8_gt)
    int8_rel_err = rel_error(gt=torch_gt, pred=ncnn_int8_out)  # change to torch_int8_gt

    print(f"fp32: {float32_rel_err:.4f}, fp16: {float16_rel_err:.4f}")
    print(f"ncnn int8: {int8_rel_err:.4f}")
    print(f"random: {random_rel_err:.4f}")


if __name__ == "__main__":
    quantize_error()

# import ncnn.model_zoo as model_zoo
# >>> model_zoo.get_model_list()
