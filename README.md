# SubCellSeg

SubCellSeg finds fluorescently tagged structures in 3D microscopy images of cells.
It is a single deep learning model that works across many different cellular
structures, so there is nothing to choose or tune: give it a z-stack, get back
a 3D mask of the structure.

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

model = SubCellSeg()                         # loads the pretrained model
volume = tifffile.imread("cell.tif")      # a (Z, Y, X) array
mask = model.segment(volume)              # uint8, 255 inside the structure
probability = model.predict_proba(volume) # float32, 0 to 1
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

## Getting help

Ask questions and report problems on the
[issue tracker](https://github.com/AllenCell/SubCellSeg/issues). Run
`subcellseg info` and include its output: it shows your versions and whether
your GPU is being used.

## Citation

<!-- RELEASE TODO: add the paper citation. -->

To be added with the publication.

## License

<!-- RELEASE TODO: add the license and a LICENSE file. -->

To be added.
