"""Training and fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pytorch_lightning as pl
import torch
from monai.losses import GeneralizedDiceFocalLoss

from megaseg.data import build_transforms, make_loader, read_manifest, split_items
from megaseg.model import PATCH_SIZE, build_network, read_state_dict
from megaseg.weights import resolve_checkpoint

PathLike = Union[str, Path]


class MegaSegModule(pl.LightningModule):
    """MegaSeg network with its training loss and optimiser."""

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
            "train/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
            batch_size=len(batch["raw"]), sync_dist=True,
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


def train(
    train_csv: PathLike,
    val_csv: Optional[PathLike] = None,
    output_dir: PathLike = "megaseg_run",
    *,
    init: Optional[PathLike] = None,
    channel: Union[int, str] = 0,
    raw_col: str = "raw",
    seg_col: str = "seg",
    val_fraction: float = 0.1,
    patch_size: Sequence[int] = PATCH_SIZE,
    batch_size: int = 8,
    patches_per_image: int = 1,
    max_epochs: int = 700,
    patience: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    lr_gamma: float = 0.995,
    num_workers: int = 4,
    cache_dir: Optional[PathLike] = None,
    device: str = "auto",
    devices: int = 1,
    seed: int = 12345,
    resume: Optional[PathLike] = None,
) -> Path:
    """Train MegaSeg from scratch, or fine-tune it on a new dataset.

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
        from the released MegaSeg weights (fine-tuning). A path starts from
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
        has not improved for ``patience`` epochs.
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
    Path of the best checkpoint, usable as ``megaseg predict --checkpoint``.
    """
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    pl.seed_everything(seed, workers=True)
    output_dir = Path(output_dir)

    train_items = read_manifest(train_csv, raw_col, seg_col)
    if val_csv is not None:
        val_items = read_manifest(val_csv, raw_col, seg_col)
    else:
        train_items, val_items = split_items(train_items, val_fraction, seed)
    print(f"Training on {len(train_items)} image(s), validating on {len(val_items)}.")

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    train_loader = make_loader(
        train_items,
        build_transforms(patch_size, channel, augment=True, patches_per_image=patches_per_image),
        shuffle=True,
        cache_dir=Path(cache_dir) / "train" if cache_dir else None,
        **loader_kwargs,
    )
    val_loader = make_loader(
        val_items,
        build_transforms(patch_size, channel, augment=False),
        shuffle=False,
        cache_dir=Path(cache_dir) / "val" if cache_dir else None,
        **loader_kwargs,
    )

    module = MegaSegModule(lr=lr, weight_decay=weight_decay, lr_gamma=lr_gamma)
    if init is not None and resume is None:
        source = resolve_checkpoint(None if str(init) == "pretrained" else init)
        module.backbone.load_state_dict(read_state_dict(source), strict=True)
        print(f"Starting from weights in {source}")
    elif resume is None:
        print("Training from scratch.")

    use_gpu = device in ("gpu", "cuda") or (device == "auto" and torch.cuda.is_available())
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
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=str(resume) if resume else None)

    best = Path(checkpoint.best_model_path)
    print(f"Best checkpoint: {best}")
    print(f"Use it with:  megaseg predict --input IMAGES --checkpoint \"{best}\"")
    return best
