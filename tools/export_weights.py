"""Turn a training checkpoint into a release weights file.

Maintainer tool. A training checkpoint also holds optimiser state and training
bookkeeping; the release file keeps only the network weights, so it is smaller
and loads with ``torch.load(..., weights_only=True)``.

    python tools/export_weights.py path/to/epoch_650.ckpt megaseg_v1.pt

Prints the sha256 to paste into ``src/megaseg/weights.py``.
"""

import argparse

import torch

from megaseg.model import ARCHITECTURE, build_network, read_state_dict
from megaseg.weights import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="cyto-dl or megaseg training checkpoint")
    parser.add_argument("output", help="weights file to write")
    args = parser.parse_args()

    state = read_state_dict(args.checkpoint)
    build_network().load_state_dict(state, strict=True)  # fail here if it does not fit

    torch.save(
        {
            "state_dict": {f"backbone.{k}": v for k, v in state.items()},
            "architecture": ARCHITECTURE,
        },
        args.output,
    )
    print(f"Wrote {args.output}")
    print(f"sha256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
