import argparse


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--bench", action="store_true")
    argparser.add_argument("--quantize-error", action="store_true")
    argparser.add_argument("--export-torchscript", action="store_true")
    return argparser.parse_args()


def main():
    args = parse_args()
    if args.bench:
        from ncnn_clip.scripts.benchmarks.bench_models import bench_models

        bench_models()
    if args.quantize_error:
        from ncnn_clip.scripts.benchmarks.quantize_error import quantize_error

        quantize_error()
    if args.export_torchscript:
        from ncnn_clip.scripts.export_torchscript import export_torchscript

        export_torchscript()


if __name__ == "__main__":
    main()
