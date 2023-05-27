## Generate ConvNext for NCNN

> git clone recursive, follow install instructions in issue:
> https://github.com/Tencent/ncnn/issues/2759

`python export_torchscript.py`

- device=cpu
- inputshape=[1,3,256,256]fp16
- fp16=1

```bash
pnnx models/torch/clip_convnext_torchscript.pt inputshape=[1,3,256,256]   \
  ncnnbin=models/ncnn/clip_convnext.ncnn.bin ncnnparam=models/ncnn/clip_convnext.ncnn.param ncnnpy=models/ncnn/clip_convnext.ncnn.py \
  pnnxbin=models/pnnx/clip_convnext.bin pnnxparammodels/pnnx/clip_convnext.param pnnxpy=models/pnnx/clip_convnext.py pnnxonnx=models/pnnx/clip_convnext.onnx
```

```bash
rm debug*
```

`python test_convnext_ncnn.py`
