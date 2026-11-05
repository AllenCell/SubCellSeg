"""Training data: CSV manifests, transforms and data loaders."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from monai.transforms import Transform

from megaseg import io

PathLike = Union[str, Path]
Item = Dict[str, str]  # {"raw": path, "seg": path}


def _resolve(value: str, base: Path) -> str:
    value = value.strip()
    if io.is_url(value):
        return value
    path = Path(value)
    return str(path if path.is_absolute() else base / path)


def read_manifest(csv_path: PathLike, raw_col: str = "raw", seg_col: str = "seg") -> List[Item]:
    """Read a CSV listing image / segmentation pairs.

    The CSV needs one column with the path of each raw image and one with the
    path of its segmentation. Relative paths are taken relative to the CSV.
    """
    csv_path = Path(csv_path)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no rows.")
    for col in (raw_col, seg_col):
        if col not in rows[0]:
            raise ValueError(
                f"{csv_path} has no column {col!r}. Columns found: {list(rows[0])}"
            )

    base = csv_path.resolve().parent
    items = [
        {"raw": _resolve(row[raw_col], base), "seg": _resolve(row[seg_col], base)}
        for row in rows
    ]
    missing = [
        p for item in items for p in item.values()
        if not io.is_url(p) and not Path(p).exists()
    ]
    if missing:
        shown = "\n  ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} file(s) listed in {csv_path} do not exist, e.g.:\n  {shown}"
        )
    return items


def split_items(
    items: List[Item], val_fraction: float = 0.1, seed: int = 12345
) -> Tuple[List[Item], List[Item]]:
    """Randomly hold out a validation set. At least one image goes to each side."""
    if len(items) < 2:
        raise ValueError(
            "At least 2 image pairs are needed to hold out a validation set."
        )
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    n_val = min(len(items) - 1, max(1, round(len(items) * val_fraction)))
    return shuffled[n_val:], shuffled[:n_val]


class LoadPaird(Transform):
    """Load a raw image and its segmentation as (1, Z, Y, X) float tensors."""

    def __init__(self, channel: Union[int, str] = 0):
        self.channel = channel

    def __call__(self, item: Item) -> Dict[str, torch.Tensor]:
        raw = io.read_volume(io.open_image(item["raw"]), self.channel)
        seg = io.read_volume(io.open_image(item["seg"]), 0)
        if raw.shape != seg.shape:
            raise ValueError(
                f"Image and segmentation differ in shape: {item['raw']} is "
                f"{raw.shape}, {item['seg']} is {seg.shape}."
            )
        return {
            "raw": torch.from_numpy(raw.astype(np.float32))[None],
            "seg": torch.from_numpy(seg.astype(np.float32))[None],
        }


def build_transforms(
    patch_size: Sequence[int],
    channel: Union[int, str] = 0,
    augment: bool = True,
    patches_per_image: int = 1,
):
    """Transforms that turn a manifest row into random training patches.

    Raw intensities are z-scored per image. Any segmentation voxel above zero
    counts as foreground, so 0/1, 0/255 and label images all work.
    """
    from monai.transforms import (
        Compose,
        NormalizeIntensityd,
        RandAdjustContrastd,
        RandHistogramShiftd,
        RandSpatialCropd,
        RandSpatialCropSamplesd,
        RandStdShiftIntensityd,
        SpatialPadd,
        ThresholdIntensityd,
        ToTensord,
    )

    keys = ["raw", "seg"]
    steps = [
        LoadPaird(channel),
        NormalizeIntensityd(keys="raw", channel_wise=True),
        ThresholdIntensityd(keys="seg", threshold=0.1, above=False, cval=1.0),
        # Images smaller than a patch are padded so every crop has the same size.
        SpatialPadd(keys=keys, spatial_size=patch_size),
    ]
    if patches_per_image > 1:
        steps.append(
            RandSpatialCropSamplesd(
                keys=keys, roi_size=patch_size, num_samples=patches_per_image,
                random_size=False,
            )
        )
    else:
        steps.append(RandSpatialCropd(keys=keys, roi_size=patch_size, random_size=False))
    if augment:
        steps += [
            RandHistogramShiftd(keys="raw", prob=0.1, num_control_points=(90, 500)),
            RandStdShiftIntensityd(keys="raw", prob=0.1, factors=0.1),
            RandAdjustContrastd(keys="raw", prob=0.1, gamma=(0.9, 1.5)),
        ]
    steps.append(ToTensord(keys=keys, track_meta=False))
    return Compose(steps)


def fixed_patches(
    items: List[Item],
    patch_size: Sequence[int],
    channel: Union[int, str] = 0,
    patches_per_image: int = 4,
    seed: int = 0,
) -> List[Dict[str, torch.Tensor]]:
    """Cut the same patches from each image every time this is called.

    Used for validation: scoring every epoch on identical patches makes the
    validation loss comparable from one epoch to the next.
    """
    transform = build_transforms(
        patch_size, channel, augment=False, patches_per_image=patches_per_image
    )
    transform.set_random_state(seed=seed)
    patches: List[Dict[str, torch.Tensor]] = []
    for item in items:
        out = transform(item)
        patches.extend(out if isinstance(out, list) else [out])
    return patches


def make_loader(
    items: List,
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    cache_dir: Optional[PathLike] = None,
):
    """Data loader over manifest rows, or over patches that are already loaded.

    With ``cache_dir``, loaded and normalised images are stored on disk after
    their first use, which makes later epochs much faster.
    """
    from monai.data import DataLoader, Dataset, PersistentDataset
    from monai.data.utils import pickle_hashing

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        dataset = PersistentDataset(
            items, transform, cache_dir=cache_dir, hash_transform=pickle_hashing
        )
    else:
        dataset = Dataset(items, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        # Start workers as fresh processes on every platform. Linux would fork
        # them by default, and a forked worker inherits the image reader's
        # thread pool without its threads, so its first read never returns.
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
