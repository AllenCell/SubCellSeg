"""The MegaSeg network and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

# MegaSeg is a residual 3D DynUNet with a single input and a single output channel.
ARCHITECTURE = {
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 1,
    "strides": [1, 2, 2, 2, 2],
    "kernel_size": [3, 3, 3, 3, 3],
    "upsample_kernel_size": [2, 2, 2, 2],
    "dropout": 0.0,
    "res_block": True,
}

# Patch size the model was trained with, as (Z, Y, X).
PATCH_SIZE = (32, 128, 128)

_PREFIX = "backbone."


def build_network() -> torch.nn.Module:
    """Return an untrained MegaSeg network. Output is logits, not probabilities."""
    from monai.networks.nets import DynUNet

    return DynUNet(**ARCHITECTURE)


def read_state_dict(checkpoint: Union[str, Path]) -> dict:
    """Read network weights from any supported checkpoint file.

    Accepts the released weights file, a checkpoint written by ``megaseg train``,
    and checkpoints from the original cyto-dl training runs. All three keep the
    network under the ``backbone.`` prefix; a bare network state dict also works.
    """
    # weights_only=True: checkpoints are loaded as data, never as pickled code.
    obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(f"{checkpoint} is not a MegaSeg checkpoint.")
    state = obj.get("state_dict", obj)
    if any(key.startswith(_PREFIX) for key in state):
        state = {k[len(_PREFIX):]: v for k, v in state.items() if k.startswith(_PREFIX)}
    return state


def load_network(checkpoint: Union[str, Path]) -> torch.nn.Module:
    """Build the network and load weights from ``checkpoint``."""
    network = build_network()
    try:
        network.load_state_dict(read_state_dict(checkpoint), strict=True)
    except RuntimeError as err:
        raise ValueError(
            f"{checkpoint} does not match the MegaSeg architecture. "
            "Only MegaSeg DynUNet checkpoints can be loaded."
        ) from err
    return network
