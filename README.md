## Generate ConvNext for NCNN

> git clone recursive, follow install instructions in issue:
> https://github.com/Tencent/ncnn/issues/2759

`python export_torchscript.py`

- device=cpu
- inputshape=[1,3,256,256]fp16
- fp16=1

```bash
pnnx models/cpu/convnext/torchscript.pt inputshape=[1,3,256,256]f32 fp16=0 device=cpu   \
  ncnnbin=models/cpu/convnext/fp32.bin ncnnparam=models/cpu/convnext/fp32.param ncnnpy=/Users/tom/.Trash/ncnn_clip_convnext_fp32.py \
  pnnxbin=/Users/tom/.Trash/pnnx_fp32.bin pnnxparam=/Users/tom/.Trash/pnnx_fp32.param pnnxpy=/Users/tom/.Trash/pnnx_clip_convnext_fp32.py pnnxonnx=/Users/tom/.Trash/fp32.onnx && \
rm debug*
```

Still take float32 input

```bash
pnnx models/cpu/convnext/torchscript.pt inputshape=[1,3,256,256]f32 fp16=1 device=cpu   \
  ncnnbin=models/cpu/convnext/fp16.bin ncnnparam=models/cpu/convnext/fp16.param ncnnpy=/Users/tom/.Trash/ncnn_clip_convnext_fp16.py \
  pnnxbin=/Users/tom/.Trash/pnnx_fp16.bin pnnxparam=/Users/tom/.Trash/pnnx_fp16.param pnnxpy=/Users/tom/.Trash/pnnx_clip_convnext_fp16.py pnnxonnx=/Users/tom/.Trash/fp16.onnx && \
rm debug*
```

`python test_convnext_ncnn.py`

## Inputs

> preprocess takes list of images (H, W, 3) or file paths and converts to tensor of (3, 256, 256), notice how 3 is first dimension

## improvements

- https://github.com/mlfoundations/open_clip#model-distillation
- https://github.com/mlfoundations/open_clip#int8-support
- open_clip.utils.freeze_batch_norm_2d
