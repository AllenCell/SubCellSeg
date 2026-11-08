"""Compare a SubCellSeg mask with the SEG_STRUCT ground-truth channel of a variance zarr.

usage: python compare_to_gt.py <zarr url> <mask tif> [prob tif] [--png out.png]
"""
import argparse, logging
import numpy as np, tifffile
logging.getLogger("bioio.bio_image").setLevel(logging.ERROR)
from bioio import BioImage

ap = argparse.ArgumentParser()
ap.add_argument("zarr"); ap.add_argument("mask"); ap.add_argument("prob", nargs="?")
ap.add_argument("--png")
a = ap.parse_args()

im = BioImage(a.zarr)
c = list(im.channel_names)
gt = im.get_image_dask_data("ZYX", C=c.index("SEG_STRUCT"), T=0).compute() > 0
raw = im.get_image_dask_data("ZYX", C=c.index("Labeled Structure"), T=0).compute()
pred = tifffile.imread(a.mask) > 0
assert pred.shape == gt.shape, (pred.shape, gt.shape)

inter = np.logical_and(pred, gt).sum(); 
dice = 2 * inter / (pred.sum() + gt.sum())
iou = inter / np.logical_or(pred, gt).sum()
print(f"shape {pred.shape}  pred fg {pred.mean()*100:.2f}%  gt fg {gt.mean()*100:.2f}%")
print(f"dice {dice:.4f}  iou {iou:.4f}  precision {inter/pred.sum():.4f}  recall {inter/gt.sum():.4f}")

if a.png:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    z = gt.sum(axis=(1, 2)).argmax()
    fig, ax = plt.subplots(1, 4 if a.prob else 3, figsize=(20, 5))
    lo, hi = np.percentile(raw[z], (1, 99.8))
    ax[0].imshow(raw[z], cmap="gray", vmin=lo, vmax=hi); ax[0].set_title(f"raw z={z}")
    ax[1].imshow(gt[z], cmap="gray"); ax[1].set_title("GT SEG_STRUCT")
    ax[2].imshow(pred[z], cmap="gray"); ax[2].set_title(f"SubCellSeg mask (dice {dice:.3f})")
    if a.prob:
        ax[3].imshow(tifffile.imread(a.prob)[z], cmap="magma"); ax[3].set_title("probability")
    for x in ax: x.axis("off")
    fig.tight_layout(); fig.savefig(a.png, dpi=80); print("wrote", a.png)
