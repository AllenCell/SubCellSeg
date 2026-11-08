"""Download and cache the pretrained SubCellSeg weights."""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

WEIGHTS_FILENAME = "subcellseg_v1.pt"

# Where the released weights are hosted: an asset of the "weights-v1" release
# of the repository. Override with SUBCELLSEG_WEIGHTS_URL.
WEIGHTS_URL = (
    "https://github.com/AllenCell/SubCellSeg/releases/download/weights-v1/" + WEIGHTS_FILENAME
)
WEIGHTS_SHA256 = "858689a3c2fcdf84422c5b375011ad2d822bd8fc556e2ebe48b49603fef89300"


def cache_dir() -> Path:
    """Directory for downloaded weights. Override with SUBCELLSEG_CACHE_DIR."""
    override = os.environ.get("SUBCELLSEG_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / "subcellseg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report(blocks: int, block_size: int, total: int) -> None:
    done = blocks * block_size / 2**20
    if total > 0:
        pct = min(100, int(done * 2**20 * 100 / total))
        sys.stderr.write(f"\r  {done:.0f} / {total / 2**20:.0f} MB ({pct}%)")
    else:
        sys.stderr.write(f"\r  {done:.0f} MB")
    sys.stderr.flush()


def fetch_pretrained() -> Path:
    """Return the path to the pretrained weights, downloading them on first use."""
    path = cache_dir() / WEIGHTS_FILENAME
    if path.exists():
        return path

    url = os.environ.get("SUBCELLSEG_WEIGHTS_URL") or WEIGHTS_URL
    if not url:
        raise RuntimeError(
            "No download location is configured for the pretrained weights.\n"
            f"Place the weights file at {path}, or pass a checkpoint explicitly "
            "(--checkpoint on the command line, checkpoint= in Python)."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    print(f"Downloading SubCellSeg weights to {path}", file=sys.stderr)
    try:
        try:
            urllib.request.urlretrieve(url, partial, reporthook=_report)
        except OSError as err:  # no network, blocked download, missing file
            raise RuntimeError(
                f"Could not download the pretrained weights from {url} ({err}).\n"
                f"Download that file in a web browser and save it as {path}, "
                "or pass a weights file with --checkpoint."
            ) from None
        sys.stderr.write("\n")
        if WEIGHTS_SHA256 and not os.environ.get("SUBCELLSEG_WEIGHTS_URL"):
            found = sha256(partial)
            if found != WEIGHTS_SHA256:
                raise RuntimeError(
                    f"Downloaded weights are corrupt (sha256 {found}). Try again, "
                    "or download the file manually and pass it with --checkpoint."
                )
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return path


def resolve_checkpoint(checkpoint: str | Path | None = None) -> Path:
    """Return ``checkpoint`` if given, else the pretrained weights."""
    if checkpoint is None:
        return fetch_pretrained()
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path
