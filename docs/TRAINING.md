# Fine-tuning SubCellSeg on your own images

Fine-tuning means showing SubCellSeg examples from *your* experiment so that it
gets better at *your* images. You give it images together with the correct
masks, and it adjusts itself. It starts from everything the pretrained model
already knows, so it needs far fewer examples than training a model from
nothing.

Fine-tune when the pretrained model gives masks that are not good enough: a
structure it has not seen, a different microscope or magnification, or
unusual sample preparation. Always [try the pretrained model first](SEGMENT.md).

You need a computer with an NVIDIA graphics card. Check with `subcellseg info`.

The whole process is five steps:

1. Collect images and their masks.
2. Put them in two folders.
3. Make a list of them.
4. Fine-tune.
5. Use your new model.

## Step 1. Collect images and their masks

For each example you need two files:

- the **image**: a 3D z-stack of the channel with your structure;
- the **mask**: a 3D image of exactly the same size, where voxels inside the
  structure are bright and everything else is 0.

Any value above 0 in the mask counts as "structure", so masks that use 1, 255
or object labels all work.

**Where do masks come from?** The quickest route is usually to correct
SubCellSeg's own output rather than drawing from scratch:

1. Run `subcellseg predict` on your images.
2. Open each image with its mask in napari or Fiji and fix the mistakes.
3. Save the corrected masks.

Masks from another segmentation method that you trust work too.

**How many examples?** Start with what you have, even if it is only a
handful, look at the result, and add more examples where the model still
struggles. Examples that cover the variety in your data (different days,
different cell densities, dim and bright cells) help more than many similar
ones.

<!-- RELEASE TODO: add guidance from the paper on how accuracy depends on the number of training images. -->

The masks must be accurate. The model learns to copy them, mistakes included.

## Step 2. Put them in two folders

Make one folder for the images and one for the masks. Give each mask **the
same file name** as its image:

```
my_data/
    raw/
        cell_01.tif
        cell_02.tif
        cell_03.tif
    seg/
        cell_01.tif
        cell_02.tif
        cell_03.tif
```

If your masks have something added to the name, such as `cell_01_seg.tif`,
that is fine too; see the next step.

## Step 3. Make a list of them

SubCellSeg reads a list (a CSV file) saying which mask belongs to which image.
This command writes it for you:

```
subcellseg make-csv --raw-dir my_data/raw --seg-dir my_data/seg --output my_data/train.csv
```

If the mask names have an ending such as `_seg`, add `--seg-suffix _seg`.

It reports how many pairs it found:

```
Wrote 3 pair(s) to my_data/train.csv
```

Check that the number is what you expect. If some images had no matching
mask, it tells you which.

You can open `train.csv` in Excel. It has two columns, `raw` and `seg`, and
one row per example. You can also write this file yourself.

## Step 4. Fine-tune

```
subcellseg finetune --train-csv my_data/train.csv --output my_model
```

SubCellSeg sets a few of your examples aside for checking its progress (the
"validation" images) and learns from the rest. You will see lines like:

```
Training on 10 image(s), validating on 2.
Starting from weights in C:\Users\you\.cache\subcellseg\subcellseg_v1.pt
Training on the GPU.
Epoch 12: 100%|##########| 2/2 [00:01<00:00, 1.70it/s, val/loss=0.351, val/dice=0.693, train/loss=0.471]
```

One **epoch** is one pass through your training images. The numbers on the
right tell you how it is going:

- `val/dice` is the agreement between SubCellSeg's mask and your mask on the
  validation images, from 0 (no overlap) to 1 (identical). **You want this to
  go up.**
- `val/loss` and `train/loss` are the error. You want these to go down.

The numbers jump around from one epoch to the next, especially with few
images. Look at the trend, not at single values.

Training stops by itself when the validation images have stopped improving
for 50 epochs (100 when training from scratch), or after 1000 epochs. You can also stop it at any time with
`Ctrl+C`; the best model so far is already saved.

When it finishes it tells you where the model is:

```
Finished after 240 epoch(s).
Best validation loss: 0.1234
Best model: my_model\checkpoints\best.ckpt
```

### What is in the output folder

```
my_model/
    checkpoints/
        best.ckpt       <- the model to use
        last.ckpt       <- the most recent state, for continuing a run
    metrics.csv         <- the numbers from every epoch, for plotting
```

## Step 5. Use your new model

Give `subcellseg predict` your model with `--checkpoint`:

```
subcellseg predict --input my_images --output results --checkpoint my_model/checkpoints/best.ckpt
```

**Judge the result on images the model has not seen**, that is, images that
were not in your training folder. A model always looks good on the examples it
learned from.

If the result is not good enough yet, correct some of the new masks, add them
to your folders, and repeat from Step 3.

## Making training faster

Reading large images from disk is usually what takes the time, not the GPU.
Add a cache folder and SubCellSeg prepares each image once and reuses it:

```
subcellseg finetune --train-csv my_data/train.csv --output my_model --cache-dir my_data/cache
```

The cache can be large, several times the size of your image files. Delete
the folder when you are done.

## Options

Run `subcellseg finetune --help` to see all of them.

| Option | Meaning | Default |
|---|---|---|
| `--train-csv` | The list from Step 3 | required |
| `--output`, `-o` | Folder for the new model | `subcellseg_run` |
| `--channel` | Channel with the structure, if images have several (number from 0, or name) | `0` |
| `--epochs` | Stop after this many epochs at the latest | `1000` |
| `--patience` | Stop when there was no improvement for this many epochs | `50` (fine-tune), `100` (from scratch) |
| `--batch-size` | Lower it if the GPU runs out of memory | `32` |
| `--lr` | Learning rate | `5e-5` (fine-tune), `1e-4` (from scratch) |
| `--val-csv` | Your own list of validation images, instead of an automatic choice | automatic |
| `--cache-dir` | Folder for prepared images, for speed | off |
| `--resume` | Continue an interrupted run from `last.ckpt` | off |

### How much GPU memory?

| `--batch-size` | GPU memory needed |
|---|---|
| 2 | about 1.5 GB |
| 4 | about 3 GB |
| 8 | about 5 GB |
| 16 | about 10 GB |
| 32 (default) | about 20 GB |

The default is the batch size the models in the paper were trained with. A
smaller batch size also gives good results and is the first thing to lower
when the GPU runs out of memory.

## Problems

**Training is extremely slow on Windows.** The batch size is probably too big
for your graphics card. Windows does not stop with an error when GPU memory
runs out; it quietly switches to much slower memory instead. Stop with
`Ctrl+C` and run again with a smaller `--batch-size`, for example 8 or 4.

**"CUDA out of memory".** Use a smaller `--batch-size`.

**"Image and segmentation differ in shape".** A mask does not have the same
size as its image. The message names the files.

**"... has no column 'raw'".** Your CSV uses different column names. Either
rename the columns to `raw` and `seg`, or tell SubCellSeg the names with
`--raw-col` and `--seg-col`.

**"At least 2 image pairs are needed".** With a single example there is
nothing left to check progress on. Add more examples.

**`val/dice` stays near 0.** Check a mask in Fiji or napari: it must overlay
its image correctly, and the structure must be the bright part. Also check
`--channel`.

**The run was interrupted.** Continue it:

```
subcellseg finetune --train-csv my_data/train.csv --output my_model --resume my_model/checkpoints/last.ckpt
```

## Training from scratch

`subcellseg train` takes the same options as `subcellseg finetune` but starts from
an untrained model instead of the pretrained one. This needs much more data
and time, and is only worth it if your images are very different from
fluorescence microscopy of cells.

```
subcellseg train --train-csv my_data/train.csv --output my_model
```

## For the curious: what happens during training

- Each image is normalised to zero mean and unit variance.
- Every epoch, one random 3D patch of 32 x 128 x 128 voxels (Z, Y, X) is cut
  from each training image, with mild random brightness and contrast changes.
- The network is a 3D residual U-Net (MONAI `DynUNet`), trained with the
  Generalized Dice + Focal loss, the Adam optimiser (learning rate 0.00005
  when fine-tuning, 0.0001 from scratch; weight decay 0.0001) and a learning
  rate that decays by 0.5% per epoch.
- Validation uses four fixed patches per validation image, the same ones
  every epoch, so that the numbers are comparable between epochs.

These are the settings used for the models in the paper, with one difference:
the paper's runs cut a fresh random validation patch every epoch.

<!-- RELEASE TODO (done 2026-11-10): checked against the paper's Methods (Hyperparameters, Fine-tuning protocol) and the cyto-dl configs. -->
