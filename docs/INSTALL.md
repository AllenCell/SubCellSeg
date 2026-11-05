# Installing SubCellSeg

This takes about five minutes and you only do it once. No programming
experience is needed: you will copy three commands into a terminal.

## Step 1. Open a terminal

A terminal is a window where you type commands.

- **Windows:** press the Start button, type `PowerShell`, press Enter.
- **Linux:** open the application called `Terminal`.

To run a command, paste it into the terminal and press Enter.

## Step 2. Install uv

[uv](https://docs.astral.sh/uv/) is a small program that installs Python
software. It takes care of Python itself too, so you do not need to install
Python separately.

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Now **close the terminal and open a new one**. This is needed for the terminal
to notice the new program.

## Step 3. Install SubCellSeg

```
uv tool install --python 3.12 --torch-backend auto https://github.com/AllenCell/SubCellSeg/archive/refs/heads/main.zip
```

This downloads SubCellSeg and everything it needs (about 3 GB with GPU support),
so it can take a few minutes. `--torch-backend auto` makes uv look at your
graphics card and install the matching version of PyTorch.

## Step 4. Check that it works

```
subcellseg info
```

You should see something like this:

```
subcellseg  0.1.0
python      3.12.13 (win32)
torch       2.14.1+cu130
gpu         NVIDIA GeForce RTX 4090 (CUDA 13.0)
weights     C:\Users\you\.cache\subcellseg\subcellseg_v1.pt (not downloaded yet)
```

The line to look at is `gpu`. If it names your graphics card, you are done.
Continue with [segmenting your images](SEGMENT.md).

## If something is not right

**`subcellseg` is not recognized / command not found.**
Close the terminal and open a new one. If that does not help, run
`uv tool update-shell`, then open a new terminal again.

**`uv` is not recognized / command not found.**
You did not open a new terminal after Step 2. Close it and open a new one.

**The `gpu` line says "none detected" or "CPU-only", but the computer has an
NVIDIA graphics card.**
First update the graphics driver from
[nvidia.com/drivers](https://www.nvidia.com/drivers), restart the computer, and
then install SubCellSeg again with the command from Step 3 with `--reinstall`
added:

```
uv tool install --reinstall --python 3.12 --torch-backend auto https://github.com/AllenCell/SubCellSeg/archive/refs/heads/main.zip
```

**The computer has no NVIDIA graphics card.**
SubCellSeg still works, on the processor. It is much slower: expect tens of
minutes per image rather than seconds. Training without a GPU is not
practical.

**The download is blocked.**
Some institutes block downloads from the terminal. Ask your IT department to
allow `astral.sh`, `github.com`, `githubusercontent.com`, `pypi.org` and
`download.pytorch.org`.

**The model cannot be downloaded.**
SubCellSeg downloads its model (about 65 MB) the first time you segment an
image. If that fails, download the file `subcellseg_v1.pt` in a web browser
from the [releases page](https://github.com/AllenCell/SubCellSeg/releases/tag/weights-v1)
and tell SubCellSeg where it is:

```
subcellseg predict --input my_images --output results --checkpoint Downloads/subcellseg_v1.pt
```

## Extra image formats

TIFF and OME-TIFF work out of the box. For other formats, install SubCellSeg with
the matching option. For example, for Zeiss CZI files:

```
uv tool install --reinstall --python 3.12 --torch-backend auto "subcellseg[czi] @ https://github.com/AllenCell/SubCellSeg/archive/refs/heads/main.zip"
```

| Format | Option |
|---|---|
| Zeiss `.czi` | `subcellseg[czi]` |
| Nikon `.nd2` | `subcellseg[nd2]` |
| OME-Zarr `.zarr` | `subcellseg[zarr]` |
| All of the above | `subcellseg[all]` |

## Updating and removing

To update to the newest version, run the install command from Step 3 again
with `--reinstall` added.

To remove SubCellSeg:

```
uv tool uninstall subcellseg
```

The downloaded model is kept in a folder called `.cache/subcellseg` in your home
folder. Delete that folder to remove it too.

## For developers

To work on the code, install it in editable mode in its own environment:

```
git clone https://github.com/AllenCell/SubCellSeg
cd SubCellSeg
uv venv --python 3.12
uv pip install --python .venv --torch-backend auto -e ".[dev]"
```

Keep the `--python .venv` part. Without it, if a conda environment is active
in your terminal (Anaconda activates `base` by default), uv installs into that
conda environment instead of the new one.

Activate the environment with `.venv\Scripts\activate` (Windows) or
`source .venv/bin/activate` (Linux), then run the tests with `pytest`.

With pip or conda instead of uv, install PyTorch first by following
[pytorch.org/get-started](https://pytorch.org/get-started/locally/), so that
you get the build that matches your GPU, and then run `pip install -e .`.
On Windows this order matters: plain `pip install` gives a CPU-only PyTorch.

SubCellSeg needs Python 3.10 or newer.
