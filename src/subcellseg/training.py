"""Training and fine-tuning."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

# Importing Lightning logs a warning about an optional GPU compiler on Windows.
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from monai.losses import GeneralizedDiceFocalLoss  # noqa: E402

from subcellseg.data import build_transforms, fixed_patches, make_loader, read_manifest, split_items  # noqa: E402
from subcellseg.model import PATCH_SIZE, build_network, read_state_dict  # noqa: E402
from subcellseg.weights import resolve_checkpoint  # noqa: E402

PathLike = str | Path


class SubCellSegModule(pl.LightningModule):
    """SubCellSeg network with its training loss and optimiser."""

    def __init__(self, lr: float = 1e-4, weight_decay: float = 1e-4, lr_gamma: float = 0.995):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = build_network()
        self.loss = GeneralizedDiceFocalLoss(sigmoid=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        loss = self.loss(self(batch["raw"]), batch["seg"])
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["raw"]),
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits, target = self(batch["raw"]), batch["seg"]
        loss = self.loss(logits, target)

        predicted, actual = logits > 0, target > 0.5
        total = predicted.sum() + actual.sum()
        dice = 2 * (predicted & actual).sum() / total if total > 0 else torch.ones(())

        n = len(batch["raw"])
        self.log("val/loss", loss, prog_bar=True, batch_size=n, sync_dist=True)
        self.log("val/dice", dice, prog_bar=True, batch_size=n, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.backbone.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_gamma)
        return [optimizer], [scheduler]


def _quiet_lightning() -> None:
    """Hide Lightning messages that mean nothing to someone training a model."""
    for name in ("pytorch_lightning", "lightning_fabric"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for message in (
        ".*does not have many workers.*",
        ".*LeafSpec.*",
        # With mixed precision the first optimiser steps may be skipped while
        # the gradient scale settles; PyTorch then reports this harmless order.
        ".*lr_scheduler.step.*before.*optimizer.step.*",
    ):
        warnings.filterwarnings("ignore", message=message)


def train(
    train_csv: PathLike,
    val_csv: PathLike | None = None,
    output_dir: PathLike = "subcellseg_run",
    *,
    init: PathLike | None = None,
    channel: int | str = 0,
    raw_col: str = "raw",
    seg_col: str = "seg",
    val_fraction: float = 0.2,
    patch_size: Sequence[int] = PATCH_SIZE,
    batch_size: int = 32,
    patches_per_image: int = 1,
    max_epochs: int = 1000,
    patience: int | None = None,
    lr: float | None = None,
    weight_decay: float = 1e-4,
    lr_gamma: float = 0.995,
    num_workers: int = 4,
    cache_dir: PathLike | None = None,
    device: str = "auto",
    devices: int = 1,
    seed: int = 12345,
    resume: PathLike | None = None,
) -> Path:
    """Train SubCellSeg from scratch, or fine-tune it on a new dataset.

    Parameters
    ----------
    train_csv
        CSV with one row per training image: a column with the raw image path
        and a column with the segmentation path.
    val_csv
        CSV of validation images. If omitted, ``val_fraction`` of the rows of
        ``train_csv`` are held out instead.
    output_dir
        Where checkpoints and the training log are written.
    init
        Starting weights. ``None`` trains from scratch. ``"pretrained"`` starts
        from the released SubCellSeg weights (fine-tuning). A path starts from
        that checkpoint.
    channel
        Channel of the raw images to train on, as an index or a channel name.
    patch_size
        Size (Z, Y, X) of the random patches the model is trained on.
    batch_size
        Patches per optimisation step.
    patches_per_image
        Random patches drawn from each image per epoch.
    max_epochs, patience
        Train for at most ``max_epochs``; stop early once the validation loss
        has not improved for ``patience`` epochs. ``patience`` defaults to 50
        when fine-tuning and 100 when training from scratch.
    lr
        Learning rate. Defaults to 5e-5 when fine-tuning and 1e-4 when
        training from scratch.

    The defaults are the settings used for the models in the paper.
    cache_dir
        Directory for an on-disk cache of preprocessed images. Speeds up
        training considerably when images are large.
    device
        "auto" (GPU if available), "cpu" or "gpu".
    devices
        Number of GPUs.
    resume
        Checkpoint of an interrupted run to continue.

    Returns
    -------
    Path of the best checkpoint, usable as ``subcellseg predict --checkpoint``.
    """
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    _quiet_lightning()
    pl.seed_everything(seed, workers=True)
    output_dir = Path(output_dir)
    finetuning = init is not None
    if lr is None:
        lr = 5e-5 if finetuning else 1e-4
    if patience is None:
        patience = 50 if finetuning else 100

    train_items = read_manifest(train_csv, raw_col, seg_col)
    if val_csv is not None:
        val_items = read_manifest(val_csv, raw_col, seg_col)
    else:
        train_items, val_items = split_items(train_items, val_fraction, seed)
    print(f"Training on {len(train_items)} image(s), validating on {len(val_items)}.")

    train_loader = make_loader(
        train_items,
        build_transforms(patch_size, channel, augment=True, patches_per_image=patches_per_image),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        cache_dir=cache_dir,
    )
    # Validation patches are cut once and reused, so every epoch is scored on
    # exactly the same data.
    val_loader = make_loader(
        fixed_patches(val_items, patch_size, channel, seed=seed),
        transform=None,
        batch_size=batch_size,
        shuffle=False,
    )

    module = SubCellSegModule(lr=lr, weight_decay=weight_decay, lr_gamma=lr_gamma)
    if init is not None and resume is None:
        source = resolve_checkpoint(None if str(init) == "pretrained" else init)
        module.backbone.load_state_dict(read_state_dict(source), strict=True)
        print(f"Starting from weights in {source}")
    elif resume is None:
        print("Training from scratch.")
    print(f"Learning rate {lr:g}; stopping after {patience} epochs without improvement.")

    use_gpu = device in ("gpu", "cuda") or (device == "auto" and torch.cuda.is_available())
    if use_gpu:
        torch.set_float32_matmul_precision("high")
    print(f"Training on the {'GPU' if use_gpu else 'CPU (slow; a GPU is strongly recommended)'}.")
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        monitor="val/loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices,
        precision="16-mixed" if use_gpu else "32-true",
        max_epochs=max_epochs,
        callbacks=[checkpoint, EarlyStopping(monitor="val/loss", mode="min", patience=patience)],
        logger=CSVLogger(save_dir=str(output_dir), name="", version=""),
        default_root_dir=str(output_dir),
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=str(resume) if resume else None)

    best = Path(checkpoint.best_model_path)
    score = checkpoint.best_model_score
    print(f"\nFinished after {trainer.current_epoch} epoch(s).")
    if score is not None:
        print(f"Best validation loss: {float(score):.4f}")
    print(f"Best model: {best}")
    print(f'Use it with:  subcellseg predict --input IMAGES --checkpoint "{best}"')
    return best
