## Generate ConvNext for NCNN

> git clonse recursive, follow install instructions in issue:
> https://github.com/Tencent/ncnn/issues/2759

1. Run export_torchscript.py
2. `pnnx models/clip_convnext.pt inputshape=[1,3,256,256]`
   - This will get converted to `[3, 256, 256]`

```bash
mv clip_convnext.ncnn.bin clip_convnext.bin
mv clip_convnext.ncnn.param clip_convnext.param
rm *pnnx*
```

1. `python test_convnext_ncnn.py`
