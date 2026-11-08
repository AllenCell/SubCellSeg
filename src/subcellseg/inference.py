"""Inference: segment arrays in memory or image files on disk."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np
import torch

from subcellseg import io
from subcellseg.model import PATCH_SIZE, load_network
from subcellseg.weights import resolve_checkpoint

PathLike = Union[str, Path]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "gpu":
        device = "cuda"
    if device.startswith("cuda") and not torch.cuda.is_available():
        hint = (
            " This PyTorch build has no CUDA support; reinstall PyTorch with CUDA (see the README)."
            if torch.version.cuda is None
            else ""
        )
        raise RuntimeError(f"A GPU was requested but CUDA is not available.{hint}")
    return torch.device(device)


class SubCellSeg:
    """A loaded SubCellSeg model.

    >>> model = SubCellSeg()                      # pretrained weights
    >>> mask = model.segment(volume)           # volume: (Z, Y, X) numpy array
    >>> prob = model.predict_proba(volume)     # float32 in [0, 1]

    Parameters
    ----------
    checkpoint
        Weights to load. Defaults to the pretrained SubCellSeg weights, which are
        downloaded on first use. A checkpoint written by ``subcellseg train`` also
        works.
    device
        "auto" (GPU if available), "cpu", "cuda", "cuda:1", ...
    """

    def __init__(self, checkpoint: PathLike | None = None, device: str = "auto"):
        self.device = _resolve_device(device)
        self.network = load_network(resolve_checkpoint(checkpoint))
        self.network.to(self.device).eval()

    def _stitch_device(self, n_voxels: int) -> torch.device:
        """Where to assemble the full-size output.

        Stitching on the GPU is faster, but the input, the output and the
        blending weights all have to fit. Large volumes are assembled on the
        CPU, with only the individual patches going to the GPU.
        """
        if self.device.type != "cuda":
            return self.device
        free, _ = torch.cuda.mem_get_info(self.device)
        return self.device if n_voxels * 4 * 4 < 0.5 * free else torch.device("cpu")

    @torch.inference_mode()
    def predict_proba(
        self,
        volume: np.ndarray,
        *,
        patch_size: Sequence[int] = PATCH_SIZE,
        overlap: float = 0.5,
        blend: str = "gaussian",
        batch_size: int = 4,
        progress: bool = False,
    ) -> np.ndarray:
        """Foreground probability for every voxel of a (Z, Y, X) volume.

        Parameters
        ----------
        volume
            Single-channel 3D image, any numeric dtype. Intensities are
            normalised internally, so no preprocessing is needed.
        patch_size
            Sliding-window size (Z, Y, X). Volumes smaller than this are padded.
        overlap
            Fraction by which neighbouring windows overlap.
        blend
            How overlapping windows are combined: "gaussian" or "constant".
        batch_size
            Number of windows run through the network at once.
        """
        from monai.inferers import sliding_window_inference

        if volume.ndim != 3:
            raise ValueError(
                f"Expected a 3D (Z, Y, X) volume, got shape {volume.shape}. "
                "Select a single channel and timepoint first."
            )

        x = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))
        # Z-score over the whole volume, as during training.
        std = x.std(unbiased=False)
        x = (x - x.mean()) / (std if std > 0 else 1.0)
        x = x[None, None].to(self._stitch_device(x.numel()))

        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            logits = sliding_window_inference(
                inputs=x,
                roi_size=tuple(patch_size),
                sw_batch_size=batch_size,
                predictor=self.network,
                overlap=overlap,
                mode=blend,
                sw_device=self.device,
                device=x.device,
                progress=progress,
            )
        return torch.sigmoid(logits[0, 0].float()).cpu().numpy()

    def segment(self, volume: np.ndarray, threshold: float = 0.5, **kwargs) -> np.ndarray:
        """Binary mask of a (Z, Y, X) volume: uint8, 255 inside the structure.

        Extra keyword arguments are passed to :meth:`predict_proba`.
        """
        return to_mask(self.predict_proba(volume, **kwargs), threshold)


def to_mask(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probability > threshold).astype(np.uint8) * 255


def predict(
    input_path: PathLike,
    output_dir: PathLike = "subcellseg_output",
    *,
    checkpoint: PathLike | None = None,
    channel: int | str = 0,
    threshold: float = 0.5,
    save_probability: bool = False,
    device: str = "auto",
    overwrite: bool = False,
    progress: bool = True,
    **inference_kwargs,
) -> list[Path]:
    """Segment an image file, a directory of images, or a remote image.

    For each input ``name.ext`` this writes ``name_seg.tif`` (binary mask,
    uint8, 0 / 255) to ``output_dir``. Time-lapse inputs produce one file per
    timepoint, ``name_T000_seg.tif``, ``name_T001_seg.tif``, ...

    Parameters
    ----------
    input_path
        Image file, directory of images, or URL of a remote image.
    output_dir
        Where results are written. Created if needed.
    checkpoint
        Weights to use. Defaults to the pretrained SubCellSeg weights.
    channel
        Channel to segment, as an index or a channel name.
    threshold
        Probability above which a voxel is foreground.
    save_probability
        Also write the probability map as ``name_prob.tif`` (uint8, 0-255).
    device
        "auto", "cpu", "cuda", "cuda:1", ...
    overwrite
        Recompute results that already exist. By default they are skipped, so
        an interrupted batch can simply be run again.
    inference_kwargs
        Passed to :meth:`SubCellSeg.predict_proba` (patch_size, overlap, blend,
        batch_size).

    Returns
    -------
    Paths of the mask files, including ones that already existed.
    """
    images = io.find_images(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SubCellSeg(checkpoint=checkpoint, device=device)
    print(f"SubCellSeg: {len(images)} image(s), device={model.device}", file=sys.stderr)
    if model.device.type == "cpu":
        print("  Running on CPU. This works but is slow for large volumes.", file=sys.stderr)

    written: list[Path] = []
    for path in images:
        image = io.open_image(path)
        stem = io.image_stem(path)
        n_time = image.dims.T
        voxel_size = io.pixel_sizes(image)
        for t in range(n_time):
            name = f"{stem}_T{t:03d}" if n_time > 1 else stem
            mask_path = output_dir / f"{name}_seg.tif"
            prob_path = output_dir / f"{name}_prob.tif"
            written.append(mask_path)
            done = mask_path.exists() and (prob_path.exists() or not save_probability)
            if done and not overwrite:
                print(f"  {name}: already done, skipping", file=sys.stderr)
                continue

            volume = io.read_volume(image, channel=channel, timepoint=t)
            print(f"  {name}: shape {tuple(volume.shape)}", file=sys.stderr)
            if volume.shape[0] == 1:
                print(
                    "    Warning: this image has a single Z plane. SubCellSeg is a "
                    "3D model and expects a z-stack.",
                    file=sys.stderr,
                )
            probability = model.predict_proba(volume, progress=progress, **inference_kwargs)
            io.write_volume(mask_path, to_mask(probability, threshold), voxel_size)
            if save_probability:
                io.write_volume(prob_path, np.round(probability * 255).astype(np.uint8), voxel_size)

    print(f"Done. Results in {output_dir.resolve()}", file=sys.stderr)
    return written
