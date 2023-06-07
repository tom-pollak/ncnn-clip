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

```bash
pnnx models/gpu/convnext/int8_torchscript.pt inputshape=[1,3,256,256]int8 fp16=1 device=gpu   \
  ncnnbin=models/gpu/convnext/int8.bin ncnnparam=models/gpu/convnext/int8.param ncnnpy=/Users/tom/.Trash/ncnn_clip_convnext_int8.py \
  pnnxbin=/Users/tom/.Trash/pnnx_int8.bin pnnxparam=/Users/tom/.Trash/pnnx_int8.param pnnxpy=/Users/tom/.Trash/pnnx_clip_convnext_int8.py pnnxonnx=/Users/tom/.Trash/int8.onnx && \
rm debug*
```

`python test_convnext_ncnn.py`

## Inputs

> preprocess takes list of images (H, W, 3) or file paths and converts to tensor of (3, 256, 256), notice how 3 is first dimension

## improvements

- https://github.com/mlfoundations/open_clip#model-distillation
- https://github.com/mlfoundations/open_clip#int8-support
- https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result
- https://github.com/Tencent/ncnn/wiki/quantized-int8-inference
- https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/quantized-int8-inference.md
- https://github.com/Tencent/ncnn/wiki/use-ncnn-with-opencv
  - How to set mean and norm values

- open_clip.utils.freeze_batch_norm_2d
- int8_model.set_grad_checkpointing(enable=False)

## ncnn2table

- mean vals \* 255
- 1 / norm vals / 255
- mean=[0.48145466, 0.4578275, 0.40821073] norm=[0.26862954, 0.26130258, 0.27577711]

mean=[122.7709383, 116.7460125, 104.09373615000001] norm=[0.014598426619242919, 0.015007768493717055, 0.014220065717024086]
