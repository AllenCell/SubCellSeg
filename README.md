# SubCellSeg

[![tests](https://github.com/AllenCell/SubCellSeg/actions/workflows/tests.yml/badge.svg)](https://github.com/AllenCell/SubCellSeg/actions/workflows/tests.yml)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

<img width="2163" height="2570" alt="Fig1_combined_v14" src="https://github.com/user-attachments/assets/ec5a2fd7-5c90-46c8-a704-286436a2d351" />


SubCellSeg is one 3D segmentation model for many subcellular structures. Most
3D segmentation is still done one structure at a time, with a separate model or
workflow per structure. SubCellSeg was instead trained jointly on 13
endogenously tagged structures from the WTC-11 hiPSC Single-Cell Image Dataset:
actin bundles, actomyosin bundles, adherens junctions, desmosomes, gap
junctions, tight junctions, the ER (two markers), the Golgi, endosomes,
mitochondria, nuclear pores and nuclear speckles. It is never told which
structure it is looking at. In the paper this single model reached a mean Dice
of 0.909 on the 13 structures and beat each of the 13 specialist models trained
on one structure (mean 0.773). Give it a z-stack of a tagged structure, get back
a 3D mask.

You can use it in three ways:

- **Segment your images** with the pretrained model. One command.
- **Fine-tune** it on your own images and masks when the pretrained model is
  not accurate enough for your structure or microscope.
- **Build on it** from Python, napari or Fiji.

You do not need programming experience for the first two.

<!-- RELEASE TODO: add a figure (raw image next to SubCellSeg mask) and the paper reference here. -->

## What you need

- A computer running **Windows** or **Linux**.
- An **NVIDIA graphics card** is strongly recommended. Segmenting one typical
  z-stack takes a few seconds on a GPU and tens of minutes without one.
- Images as 3D z-stacks: TIFF or OME-TIFF out of the box; CZI, ND2 and OME-Zarr
  with one extra install step.

## What to expect

The training images are live WTC-11 hiPS cells on a spinning-disk confocal at
100x (voxels 0.108 x 0.108 x 0.29 µm), one mEGFP-tagged protein per image. In
the paper the model also worked without any retraining on images from other
microscopes (Airyscan, a second spinning disk), at 20x to 100x, on
drug-treated cells where mitochondria fragmented or the Golgi dispersed, on
structures it had never seen (PCNA foci, nucleoli, microtubules), and on
sarcomeres in iPSC-derived cardiomyocytes. So try it as is first.

Where it does not work: signals defined by texture rather than by objects with
edges. EZH2, a diffuse chromatin-associated signal, gave only sparse
detections. It cannot recover detail that the optics did not resolve. And the
mask is semantic, not instance: to count objects, split the mask afterwards
with connected components or a watershed.

When to fine-tune: when your mask convention differs from what the model
learned. The clearest case in the paper is the nuclear envelope via lamin B1.
The model segments it as a thin shell, but a nuclear mask needs the filled
nucleus, so Dice out of the box was 0.31. Fine-tuning on 10% of the labelled
images (16 z-stacks) gave 0.94, better than training a new model from scratch
on all 160. Lysosomes via LAMP-1 went from 0.73 to 0.93 the same way. A few
dozen annotated stacks is usually enough.

## 1. Install

Two commands. Full step-by-step instructions, including what to do if
something goes wrong, are in **[docs/INSTALL.md](docs/INSTALL.md)**.

**Windows** (in PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux** (in a terminal):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then close the window, open a new one, and on either system run:

```
uv tool install --python 3.12 --torch-backend auto https://github.com/AllenCell/SubCellSeg/archive/refs/heads/main.zip
```

Check that it worked:

```
subcellseg info
```

## 2. Segment your images

```
subcellseg predict --input my_images --output results
```

`my_images` can be one image or a folder of images. For every image, a mask
named `<image name>_seg.tif` appears in `results`. In the mask, voxels inside
the structure are 255 and everything else is 0. Open it in Fiji or napari on
top of your original image.

If your images have several channels, say which one holds the structure:

```
subcellseg predict --input my_images --output results --channel 1
```

The first time you run this, the model (about 65 MB) is downloaded
automatically.

More detail, options and tips: **[docs/SEGMENT.md](docs/SEGMENT.md)**.

## 3. Fine-tune on your own data

If the pretrained model is not good enough for your images, teach it with
examples: a folder of images and a folder of matching masks.

```
subcellseg make-csv --raw-dir my_data/raw --seg-dir my_data/seg --output my_data/train.csv
subcellseg finetune --train-csv my_data/train.csv --output my_model
subcellseg predict --input my_images --output results --checkpoint my_model/checkpoints/best.ckpt
```

The full walkthrough is in **[docs/TRAINING.md](docs/TRAINING.md)**.

## Use from Python

```python
import tifffile
from subcellseg import SubCellSeg

model = SubCellSeg()  # loads the pretrained model
volume = tifffile.imread("cell.tif")  # a (Z, Y, X) array
mask = model.segment(volume)  # uint8, 255 inside the structure
probability = model.predict_proba(volume)  # float32, 0 to 1
```

`subcellseg.predict(...)` and `subcellseg.train(...)` do the same as the commands
above. There is a runnable example in
[examples/quickstart.ipynb](examples/quickstart.ipynb).

## napari and Fiji

SubCellSeg is also available as a napari plugin and a Fiji plugin, for people who
prefer to click rather than type.

<!-- RELEASE TODO: link the napari plugin and the Fiji plugin. -->

## Video tutorials

<!-- RELEASE TODO: link the installation, segmentation and training videos. -->

Coming soon.

## Benchmarks

Measured on 2026-11-08 with a fresh `uv tool install --torch-backend auto`
(torch 2.14.1+cu130, Python 3.12) on a Linux SLURM node: one NVIDIA A100-SXM4-80GB
(driver 580.178.04) and 2 cores of an AMD EPYC 7713. Test images are the public
`variance/<id>.zarr` files (channel `Labeled Structure`, 624 x 924 pixels per slice)
from `https://animatedcell-test-data.s3.us-west-2.amazonaws.com/`.

| Step | Time |
|---|---|
| Install with GPU support, empty uv cache (5.7 GB on disk, 529 MiB torch wheel) | 26 s |
| `pytest` (10 tests, includes a short training run) on the GPU | 80 s |

| Image | Z slices | GPU, `subcellseg predict` wall time | GPU peak memory | CPU (2 cores) wall time |
|---|---|---|---|---|
| 40 | 38 | 15.1 s first run, 9.3 s repeat | 2.0 GB | 957 s |
| 6919 | 65 | 10.7 s | 2.5 GB | not run |
| 11370 | 70 | 11.0 s | 2.5 GB | not run |

Wall time is the whole command: starting Python and PyTorch, loading the model,
reading the image over the network (1 to 5 s here) and writing the outputs.
The sliding-window inference itself takes about 1.3 s for image 40 on the A100
(63 windows at roughly 48 windows/s). The default `--batch-size 4` and
`--patch-size 32 128 128` were used.

The CPU and GPU masks for image 40 agree on 99.997 % of voxels. Against the
classic `SEG_STRUCT` segmentation stored in the same files:

| Image | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| 40 | 0.533 | 0.363 | 0.732 | 0.419 |
| 6919 | 0.877 | 0.781 | 0.902 | 0.853 |
| 11370 | 0.846 | 0.733 | 0.883 | 0.812 |

To reproduce, run `tools/benchmark/bench_predict.sh cuda results 40 6919 11370`
(it times `subcellseg predict`, records GPU memory, and runs
`tools/benchmark/compare_to_gt.py`; the comparison needs `subcellseg[zarr]`
and matplotlib). Add your own machine as a new row in the tables above.

## Maintenance

This code is being actively developed. Please see
[CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

- Suraj Mishra (`smishra3`) maintains the package, triages issues and pull
  requests and handles security reports ([SECURITY.md](SECURITY.md)). Ashwin
  Samudre (`kiryteo`) and Jessica Yu (`jessicasyu`) review pull requests.
- Expect a first response to an issue within two weeks. Wrong segmentation
  output and installation problems come first; feature requests are discussed
  in the issue before any work starts.
- Releases are tagged `vX.Y.Z` from `main` when there is a user-visible change
  (see [CHANGELOG.md](CHANGELOG.md)). Model weights have their own
  `weights-vN` tags and are never replaced in place.
- Planned: the napari and Fiji plugins, video tutorials, the items marked
  RELEASE TODO in the docs, and importing the paper's benchmark and figure
  code into this repository.
- After the paper is published, active development will slow down and this
  section will say so. Bug fixes and security updates will continue.

## How this code was developed

The model, its training recipe and all experiments in the paper were developed
by the authors from 2024 to 2026 with [cyto-dl](https://github.com/AllenCell/cyto-dl).
That development history lives in the MegaSeg repository and is being imported
here. This package re-implements inference and fine-tuning without cyto-dl so
that it installs on its own; the pretrained weights are the paper's cyto-dl
checkpoint, converted with `tools/export_weights.py`.

The package code and documentation were drafted with AI assistance (Claude
Code) and reviewed, tested and edited by the authors, following the Allen
Institute's guidelines for AI coding agents. Commits that contain AI-generated
code carry an `Assisted-by:` trailer naming the tool and model.

<!-- RELEASE TODO: verify on the test images that the masks from this package match the cyto-dl predictions used in the paper, and record the result here. -->

## Getting help

Ask questions and report problems on the
[issue tracker](https://github.com/AllenCell/SubCellSeg/issues). Run
`subcellseg info` and include its output: it shows your versions and whether
your GPU is being used.

Want to help? See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

<!-- RELEASE TODO: add the paper citation. -->

To be added with the publication.

The pretrained model was trained with
[cyto-dl](https://github.com/AllenCell/cyto-dl). You do not need cyto-dl to use
SubCellSeg: this package is standalone.

## License

SubCellSeg is free to use, change and share under the
[BSD 3-Clause License](LICENSE).
