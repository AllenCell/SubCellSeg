"""Command line interface: ``subcellseg predict | train | finetune | make-csv | info``."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from subcellseg.model import PATCH_SIZE


def _channel(value: str):
    """A channel is an index if it looks like one, otherwise a channel name."""
    return int(value) if value.lstrip("-").isdigit() else value


def _add_predict(sub) -> None:
    p = sub.add_parser(
        "predict", help="Segment images.",
        description="Segment an image, a folder of images, or a remote image.",
    )
    p.add_argument("--input", "-i", required=True,
                   help="Image file, folder of images, or URL.")
    p.add_argument("--output", "-o", default="subcellseg_output",
                   help="Output folder (default: %(default)s).")
    p.add_argument("--checkpoint", "-c",
                   help="Weights to use. Default: the pretrained SubCellSeg weights.")
    p.add_argument("--channel", type=_channel, default=0,
                   help="Channel to segment: index or name (default: 0).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability above which a voxel is foreground (default: 0.5).")
    p.add_argument("--save-prob", action="store_true",
                   help="Also save the probability map as *_prob.tif (uint8).")
    p.add_argument("--device", default="auto",
                   help="auto, cpu, cuda, cuda:1, ... (default: auto).")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="Sliding-window overlap (default: 0.5).")
    p.add_argument("--blend", choices=["gaussian", "constant"], default="gaussian",
                   help="How overlapping windows are combined (default: gaussian).")
    p.add_argument("--patch-size", type=int, nargs=3, default=list(PATCH_SIZE),
                   metavar=("Z", "Y", "X"), help="Sliding-window size (default: 32 128 128).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Windows per forward pass; lower it if the GPU runs out of memory.")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute results that already exist.")
    p.set_defaults(run=_run_predict)


def _run_predict(args) -> None:
    from subcellseg.inference import predict

    predict(
        args.input, args.output,
        checkpoint=args.checkpoint, channel=args.channel, threshold=args.threshold,
        save_probability=args.save_prob, device=args.device, overwrite=args.overwrite,
        patch_size=args.patch_size, overlap=args.overlap, blend=args.blend,
        batch_size=args.batch_size,
    )


def _add_train(sub, name: str, finetune: bool) -> None:
    what = "Fine-tune the pretrained model on" if finetune else "Train a model from scratch on"
    p = sub.add_parser(name, help=f"{what} your data.", description=f"{what} your data.")
    p.add_argument("--train-csv", required=True,
                   help="CSV with a raw image column and a segmentation column.")
    p.add_argument("--val-csv",
                   help="Validation CSV. Default: hold out part of --train-csv.")
    p.add_argument("--output", "-o", default="subcellseg_run",
                   help="Folder for checkpoints and logs (default: %(default)s).")
    p.add_argument("--init", default="pretrained" if finetune else None,
                   help="Checkpoint to start from."
                        + (" Default: the pretrained SubCellSeg weights." if finetune else ""))
    p.add_argument("--channel", type=_channel, default=0,
                   help="Channel of the raw images: index or name (default: 0).")
    p.add_argument("--raw-col", default="raw", help="Raw image column (default: raw).")
    p.add_argument("--seg-col", default="seg", help="Segmentation column (default: seg).")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Fraction held out for validation without --val-csv (default: 0.1).")
    p.add_argument("--epochs", type=int, default=700, help="Maximum epochs (default: 700).")
    p.add_argument("--patience", type=int, default=100,
                   help="Stop after this many epochs without improvement (default: 100).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Patches per step; lower it if the GPU runs out of memory (default: 8).")
    p.add_argument("--patches-per-image", type=int, default=1,
                   help="Random patches drawn per image per epoch (default: 1).")
    p.add_argument("--patch-size", type=int, nargs=3, default=list(PATCH_SIZE),
                   metavar=("Z", "Y", "X"), help="Training patch size (default: 32 128 128).")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4).")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Data loading processes (default: 4).")
    p.add_argument("--cache-dir",
                   help="Folder for an on-disk cache of preprocessed images (faster epochs).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs (default: 1).")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--resume", help="Checkpoint of an interrupted run to continue.")
    p.set_defaults(run=_run_train)


def _run_train(args) -> None:
    from subcellseg.training import train

    train(
        args.train_csv, args.val_csv, args.output,
        init=args.init, channel=args.channel, raw_col=args.raw_col, seg_col=args.seg_col,
        val_fraction=args.val_fraction, patch_size=args.patch_size,
        batch_size=args.batch_size, patches_per_image=args.patches_per_image,
        max_epochs=args.epochs, patience=args.patience, lr=args.lr,
        num_workers=args.num_workers, cache_dir=args.cache_dir, device=args.device,
        devices=args.devices, seed=args.seed, resume=args.resume,
    )


def _add_make_csv(sub) -> None:
    p = sub.add_parser(
        "make-csv", help="Build a training CSV from two folders.",
        description="Pair raw images with segmentations that share a file name.",
    )
    p.add_argument("--raw-dir", required=True, help="Folder of raw images.")
    p.add_argument("--seg-dir", required=True, help="Folder of segmentations.")
    p.add_argument("--seg-suffix", default="",
                   help="Text appended to the raw name in segmentation file names, "
                        "e.g. _seg for image1.tif / image1_seg.tif.")
    p.add_argument("--output", "-o", default="train.csv", help="CSV to write.")
    p.set_defaults(run=_run_make_csv)


def _run_make_csv(args) -> None:
    from subcellseg import io

    segs = {io.image_stem(p): p for p in io.find_images(args.seg_dir)}
    rows, unmatched = [], []
    for raw in io.find_images(args.raw_dir):
        seg = segs.get(io.image_stem(raw) + args.seg_suffix)
        if seg is None:
            unmatched.append(raw)
        else:
            rows.append((str(Path(raw).resolve()), str(Path(seg).resolve())))
    if not rows:
        sys.exit("No raw image has a matching segmentation. Check --seg-suffix.")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["raw", "seg"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pair(s) to {args.output}")
    if unmatched:
        print(f"{len(unmatched)} raw image(s) had no segmentation, e.g. {unmatched[0]}")


def _add_info(sub) -> None:
    p = sub.add_parser("info", help="Show the installation and whether a GPU is usable.")
    p.set_defaults(run=_run_info)


def _run_info(args) -> None:
    import torch

    import subcellseg
    from subcellseg.weights import WEIGHTS_FILENAME, cache_dir

    weights = cache_dir() / WEIGHTS_FILENAME
    print(f"subcellseg  {subcellseg.__version__}")
    print(f"python      {sys.version.split()[0]} ({sys.platform})")
    print(f"torch       {torch.__version__}")
    if torch.cuda.is_available():
        print(f"gpu         {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    elif torch.version.cuda is None:
        print("gpu         not usable: this PyTorch build is CPU-only. If you have an "
              "NVIDIA GPU, reinstall PyTorch with CUDA (see the README).")
    else:
        print("gpu         none detected; running on CPU")
    print(f"weights     {weights} ({'present' if weights.exists() else 'not downloaded yet'})")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="subcellseg",
        description="SubCellSeg: 3D segmentation of fluorescently tagged cellular structures.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_predict(sub)
    _add_train(sub, "train", finetune=False)
    _add_train(sub, "finetune", finetune=True)
    _add_make_csv(sub)
    _add_info(sub)

    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        sys.exit(f"Error: {_root_cause(err)}")


def _root_cause(err: BaseException) -> BaseException:
    """The original error behind ``err``.

    Libraries wrap errors ("applying transform ...") in ways that hide the
    message that actually explains the problem, such as which file is wrong.
    """
    while err.__cause__ is not None:
        err = err.__cause__
    return err


if __name__ == "__main__":
    main()
