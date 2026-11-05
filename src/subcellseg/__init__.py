"""SubCellSeg: 3D segmentation of fluorescently tagged cellular structures."""

from subcellseg.inference import SubCellSeg, predict

__version__ = "0.1.0"

__all__ = ["SubCellSeg", "predict", "train", "__version__"]


def train(*args, **kwargs):
    """Train or fine-tune a model. See :func:`subcellseg.training.train`."""
    # Imported lazily so that inference does not pay for importing Lightning.
    from subcellseg.training import train as _train

    return _train(*args, **kwargs)
