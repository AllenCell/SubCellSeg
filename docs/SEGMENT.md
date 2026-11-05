# Segmenting your images

This page assumes SubCellSeg is [installed](INSTALL.md).

## Try it on an example first

This command segments a public example image from the Allen Institute for Cell
Science, straight from the internet. It needs the `subcellseg[zarr]` option from
[Extra image formats](INSTALL.md#extra-image-formats).

```
subcellseg predict --input https://animatedcell-test-data.s3.us-west-2.amazonaws.com/variance/10932.zarr --channel "Labeled Structure" --output example_results
```

<!-- RELEASE TODO: confirm this is the example image to advertise, or replace it with a hosted TIFF so the example needs no extra option. -->

After a few seconds on a GPU you get `example_results/10932_seg.tif`.

## Segment your own images

Put the images you want to segment in one folder, for example `my_images`.
Then run:

```
subcellseg predict --input my_images --output results
```

You can type the folder's full location instead, for example
`--input "D:\Experiments\2024-05-01"`. Use quotes if the name contains spaces.
A handy trick: type `subcellseg predict --input ` and then drag the folder from
the file explorer into the terminal window.

`--input` can also be a single image file.

### What you get

For each image `name.tif` you get `results/name_seg.tif`:

- the same size as your image, so it overlays exactly;
- **255** where SubCellSeg sees the structure, **0** everywhere else;
- the same pixel size as your image, if the image file recorded one.

For a time-lapse you get one mask per timepoint: `name_T000_seg.tif`,
`name_T001_seg.tif`, and so on.

### Look at the result

- **Fiji:** open your image and the mask, then `Image > Color > Merge Channels`.
- **napari:** drag both files into the window; set the mask to a colour and
  lower its opacity.

## Images with several channels

SubCellSeg segments one channel: the one with your fluorescently tagged
structure. By default it takes the first channel. To choose another, count
channels starting from 0 (the first channel is 0, the second is 1):

```
subcellseg predict --input my_images --output results --channel 1
```

If your files store channel names, you can use the name instead:

```
subcellseg predict --input my_images --output results --channel "EGFP"
```

## Getting good results

- **Give SubCellSeg 3D z-stacks.** It is a 3D model. A single 2D image runs, but
  the result will be poor.
- **Use whole fields of view rather than small crops.** SubCellSeg adjusts to the
  brightness of each image as a whole, so cutting an image into small tiles
  and segmenting each tile separately can give clearly worse results than
  segmenting the whole image.
- **Images similar to the training data work best.** SubCellSeg was trained on
  3D spinning-disk confocal images taken with a 100x objective (voxels of
  about 0.108 x 0.108 x 0.29 micrometres). The further your images are from
  that, the more likely you will benefit from
  [fine-tuning](TRAINING.md).
- **If the mask is too generous or too strict**, change the threshold. Every
  voxel gets a score between 0 and 1 for how likely it belongs to the
  structure; by default, voxels above 0.5 are kept. Lower values keep more:

  ```
  subcellseg predict --input my_images --output results --threshold 0.3 --overwrite
  ```

  To see the scores themselves, add `--save-prob`. You then also get
  `name_prob.tif`, an 8-bit image where 0 means score 0 and 255 means score 1.

## Good to know

- **Running the command again skips images that are already done.** If a long
  run is interrupted, just run the same command again and it continues where
  it stopped. Add `--overwrite` to redo everything, for example after changing
  the threshold.
- **Use your own fine-tuned model** with `--checkpoint`:

  ```
  subcellseg predict --input my_images --output results --checkpoint my_model/checkpoints/best.ckpt
  ```

- **Only the top level of the folder is searched.** Images in sub-folders are
  not picked up.
- **Files with several scenes** (some CZI files): only the first scene is
  segmented.

## All options

Run `subcellseg predict --help` to see them. The ones you are most likely to use:

| Option | Meaning | Default |
|---|---|---|
| `--input`, `-i` | Image, folder of images, or web address | required |
| `--output`, `-o` | Folder for the results | `subcellseg_output` |
| `--channel` | Channel with the structure (number from 0, or name) | `0` |
| `--threshold` | Score above which a voxel counts as structure | `0.5` |
| `--save-prob` | Also save the score image | off |
| `--checkpoint`, `-c` | Use your own model | pretrained model |
| `--overwrite` | Redo images that are already done | off |
| `--device` | `auto`, `cpu` or `cuda` | `auto` |
| `--batch-size` | Lower it if the GPU runs out of memory | `4` |

## Problems

**"CUDA out of memory".** Add `--batch-size 1`. If it still fails, add
`--device cpu` (slow, but it works for any image size your computer's memory
can hold).

**It is very slow.** Run `subcellseg info`. If the `gpu` line does not name your
graphics card, see [If something is not right](INSTALL.md#if-something-is-not-right).

**"Could not open ...".** The file format needs an extra reader; see
[Extra image formats](INSTALL.md#extra-image-formats).

**"Channel ... is out of range".** The image has fewer channels than you
asked for. Remember that counting starts at 0.

**The mask is empty or clearly wrong.** Check that you picked the right
channel. Then try a lower `--threshold` with `--save-prob` and look at the
score image. If the structure is visible there but faint, lower the threshold
further; if it is not, the model needs [fine-tuning](TRAINING.md) for your
images.

## How this relates to the paper

`subcellseg predict` blends overlapping windows with 50% overlap and Gaussian
weighting, which gives smoother results at window borders. The benchmarks in
the paper used 30% overlap and uniform weighting. To reproduce those settings:

```
subcellseg predict --input my_images --output results --overlap 0.3 --blend constant
```

<!-- RELEASE TODO: confirm these are the settings behind the paper's numbers. -->
