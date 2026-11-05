"""MegaSeg: 3D segmentation of fluorescently tagged cellular structures."""

from megaseg.inference import MegaSeg, predict

__version__ = "0.1.0"

__all__ = ["MegaSeg", "predict", "train", "__version__"]


def train(*args, **kwargs):
    """Train or fine-tune a model. See :func:`megaseg.training.train`."""
    # Imported lazily so that inference does not pay for importing Lightning.
    from megaseg.training import train as _train

    return _train(*args, **kwargs)
