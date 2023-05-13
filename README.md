## Generate ConvNext for NCNN

> git clonse recursive, follow install instructions in issue:
> https://github.com/Tencent/ncnn/issues/2759

1. Run export_torchscript.py
2. pnnx clip_convnext.pt inputshape=[1,3,256,256]
3.

```bash
mv clip_convnext.ncnn.bin clip_convnext.bin
mv clip_convnext.ncnn.param clip_convnext.param
rm *pnnx*
```

4. `python test_convnext_ncnn.py`
