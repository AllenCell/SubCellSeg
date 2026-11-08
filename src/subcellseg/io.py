"""Reading and writing 3D images."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]

# Longest first, so "x.ome.tiff" is matched as ".ome.tiff" and not ".tiff".
IMAGE_SUFFIXES = (
    ".ome.tiff",
    ".ome.tif",
    ".ome.zarr",
    ".tiff",
    ".tif",
    ".czi",
    ".zarr",
    ".nd2",
)

# Formats that need an optional reader: suffix -> name of the pip extra.
_EXTRAS = {".czi": "czi", ".ome.zarr": "zarr", ".zarr": "zarr", ".nd2": "nd2"}


def is_url(path: PathLike) -> bool:
    return str(path).startswith(("http://", "https://", "s3://", "gs://"))


def image_suffix(path: PathLike) -> str | None:
    """The supported image suffix of ``path``, or None if it is not an image."""
    name = str(path).rstrip("/\\").lower()
    return next((s for s in IMAGE_SUFFIXES if name.endswith(s)), None)


def image_stem(path: PathLike) -> str:
    """File name without directory or image suffix: 'a/b.ome.tiff' -> 'b'."""
    name = str(path).rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
    suffix = image_suffix(name)
    return name[: -len(suffix)] if suffix else Path(name).stem


def find_images(input_path: PathLike) -> list[PathLike]:
    """Resolve an input to a list of images.

    ``input_path`` may be a single image, a remote URL, or a directory. A
    directory is searched at its top level only. A ``.zarr`` store is one image
    even though it is a directory on disk.
    """
    if is_url(input_path):
        return [str(input_path)]
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.is_file() or image_suffix(path) in (".zarr", ".ome.zarr"):
        return [path]
    images = sorted(p for p in path.iterdir() if image_suffix(p))
    if not images:
        raise FileNotFoundError(
            f"No images found in {path}. Supported: {', '.join(IMAGE_SUFFIXES)}"
        )
    return images


def open_image(path: PathLike):
    """Open an image with bioio. Returns a ``BioImage``."""
    from bioio import BioImage

    # bioio tries each installed reader in turn and logs a warning for every one
    # that does not fit, e.g. the OME reader on a plain TIFF. That is expected.
    logging.getLogger("bioio.bio_image").setLevel(logging.ERROR)
    try:
        return BioImage(str(path) if is_url(path) else path)
    except Exception as err:
        extra = _EXTRAS.get(image_suffix(path) or "")
        if extra:
            raise RuntimeError(
                f"Could not open {path}. Reading this format needs the optional "
                f'"subcellseg[{extra}]" reader, or the file is damaged. See "Extra '
                'image formats" in docs/INSTALL.md for how to install it.'
            ) from err
        raise


def resolve_channel(image, channel: int | str) -> int:
    """Turn a channel index or channel name into a valid channel index."""
    names = [str(n) for n in image.channel_names]
    if isinstance(channel, str):
        if channel not in names:
            raise ValueError(f"Channel {channel!r} not found. Available: {names}")
        return names.index(channel)
    if not 0 <= channel < image.dims.C:
        raise ValueError(
            f"Channel {channel} is out of range: the image has {image.dims.C} channel(s): {names}"
        )
    return channel


def read_volume(image, channel: int | str = 0, timepoint: int = 0) -> np.ndarray:
    """Read one channel and timepoint of an open image as a (Z, Y, X) array."""
    c = resolve_channel(image, channel)
    return image.get_image_dask_data("ZYX", C=c, T=timepoint).compute()


def pixel_sizes(image) -> Sequence[float | None] | None:
    """Physical (Z, Y, X) voxel size in microns, where the file records it."""
    try:
        size = image.physical_pixel_sizes
        return (size.Z, size.Y, size.X)
    except Exception:
        return None


def write_volume(
    path: PathLike,
    data: np.ndarray,
    voxel_size: Sequence[float | None] | None = None,
) -> None:
    """Write a (Z, Y, X) array as a compressed OME-TIFF."""
    import tifffile

    metadata = {"axes": "ZYX"}
    for axis, size in zip("ZYX", voxel_size or ()):
        if size:
            metadata[f"PhysicalSize{axis}"] = float(size)
            metadata[f"PhysicalSize{axis}Unit"] = "µm"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name first so an interrupted run never leaves behind
    # a truncated file that a later run would mistake for a finished result.
    partial = path.with_name(path.name + ".part")
    try:
        tifffile.imwrite(
            partial,
            data,
            ome=True,
            compression="zlib",
            photometric="minisblack",
            metadata=metadata,
        )
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
