## Generate ConvNext for NCNN

> git clone recursive, follow install instructions in issue:
> https://github.com/Tencent/ncnn/issues/2759

`python export_torchscript.py`

- device=cpu
- inputshape=[1,3,256,256]fp16
- fp16=1

```bash
pnnx models/convnext/torch/clip_torchscript.pt inputshape=[1,3,256,256]   \
  ncnnbin=models/convnext/ncnn/clip.ncnn.bin ncnnparam=models/convnext/ncnn/clip.ncnn.param ncnnpy=models/convnext/ncnn/clip.ncnn.py \
  pnnxbin=models/convnext/pnnx/clip.bin pnnxparam=models/convnext/pnnx/clip.param pnnxpy=models/convnext/pnnx/clip.py pnnxonnx=models/convnext/pnnx/clip.onnx
```

```bash
rm debug*
```

`python test_convnext_ncnn.py`

## Inputs

> preprocess takes list of images (H, W, 3) or file paths and converts to tensor of (3, 256, 256), notice how 3 is first dimension
