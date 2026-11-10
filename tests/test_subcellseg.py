"""Smoke tests. They use randomly initialised weights and run on the CPU, so
they need neither a GPU nor the pretrained weights."""

import csv

import numpy as np
import pytest
import tifffile
import torch

from subcellseg import SubCellSeg, io, predict
from subcellseg.cli import main
from subcellseg.model import build_network, load_network, read_state_dict

SMALL_PATCH = (16, 32, 32)


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory):
    """A weights file in the release format, with random weights."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("weights") / "random.pt"
    state = {f"backbone.{k}": v for k, v in build_network().state_dict().items()}
    torch.save({"state_dict": state}, path)
    return path


@pytest.fixture(scope="session")
def model(checkpoint):
    return SubCellSeg(checkpoint=checkpoint, device="cpu")


def _volume(shape=(18, 40, 44), seed=0):
    return np.random.default_rng(seed).integers(0, 4000, shape, dtype=np.uint16)


def test_image_names_and_discovery(tmp_path):
    assert io.image_stem("a/b/cell.ome.tiff") == "cell"
    assert io.image_stem(r"C:\data\cell.tif") == "cell"
    assert io.image_stem("https://host/bucket/123.zarr/") == "123"

    for name in ("b.ome.tif", "a.tif", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "c.zarr").mkdir()
    found = [p.name for p in io.find_images(tmp_path)]
    assert found == ["a.tif", "b.ome.tif", "c.zarr"]  # each image exactly once

    assert io.find_images(tmp_path / "c.zarr") == [tmp_path / "c.zarr"]
    with pytest.raises(FileNotFoundError):
        io.find_images(tmp_path / "missing")


def test_checkpoint_formats(tmp_path, checkpoint):
    bare = build_network().state_dict()
    torch.save(bare, tmp_path / "bare.pt")
    assert read_state_dict(tmp_path / "bare.pt").keys() == bare.keys()
    assert read_state_dict(checkpoint).keys() == bare.keys()
    load_network(checkpoint)

    torch.save({"state_dict": {"backbone.nope": torch.zeros(1)}}, tmp_path / "wrong.pt")
    with pytest.raises(ValueError):
        load_network(tmp_path / "wrong.pt")


def test_predict_proba(model):
    volume = _volume()
    prob = model.predict_proba(volume, patch_size=SMALL_PATCH)
    assert prob.shape == volume.shape
    assert prob.dtype == np.float32
    assert 0.0 <= prob.min() and prob.max() <= 1.0

    mask = model.segment(volume, patch_size=SMALL_PATCH)
    assert mask.dtype == np.uint8
    assert np.array_equal(mask > 0, prob >= 0.5)

    with pytest.raises(ValueError):
        model.predict_proba(volume[0])


def test_defaults_match_paper():
    """The package defaults are the inference and training settings of the paper."""
    import inspect

    from subcellseg.training import train

    infer = inspect.signature(SubCellSeg.predict_proba).parameters
    assert infer["patch_size"].default == (32, 128, 128)
    assert infer["overlap"].default == 0.3
    assert infer["blend"].default == "constant"
    assert inspect.signature(SubCellSeg.segment).parameters["threshold"].default == 0.5

    fit = inspect.signature(train).parameters
    assert fit["batch_size"].default == 32
    assert fit["max_epochs"].default == 1000
    assert fit["val_fraction"].default == 0.2
    # Learning rate and patience depend on whether we fine-tune; resolved inside train().
    assert fit["lr"].default is None and fit["patience"].default is None


def test_predict_ignores_intensity_scale(model):
    volume = _volume().astype(np.float32)
    a = model.predict_proba(volume, patch_size=SMALL_PATCH)
    b = model.predict_proba(volume * 7 + 300, patch_size=SMALL_PATCH)
    assert np.allclose(a, b, atol=1e-3)


def test_predict_files(tmp_path, checkpoint):
    volume = _volume()
    tifffile.imwrite(tmp_path / "cell.tif", volume)
    out = tmp_path / "out"

    written = predict(
        tmp_path / "cell.tif",
        out,
        checkpoint=checkpoint,
        device="cpu",
        save_probability=True,
        progress=False,
        patch_size=SMALL_PATCH,
    )
    assert written == [out / "cell_seg.tif"]
    mask = tifffile.imread(out / "cell_seg.tif")
    assert mask.shape == volume.shape and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert tifffile.imread(out / "cell_prob.tif").shape == volume.shape
    assert not list(out.glob("*.part"))

    # A second run skips finished work; --overwrite redoes it.
    before = (out / "cell_seg.tif").stat().st_mtime_ns
    predict(
        tmp_path / "cell.tif",
        out,
        checkpoint=checkpoint,
        device="cpu",
        progress=False,
        patch_size=SMALL_PATCH,
    )
    assert (out / "cell_seg.tif").stat().st_mtime_ns == before


def test_channel_and_timelapse(tmp_path, checkpoint, model):
    data = np.stack([_volume(seed=s) for s in range(4)]).reshape(2, 2, 18, 40, 44)
    tifffile.imwrite(tmp_path / "movie.ome.tif", data, metadata={"axes": "TCZYX"})
    out = tmp_path / "out"

    written = predict(
        tmp_path / "movie.ome.tif",
        out,
        checkpoint=checkpoint,
        device="cpu",
        channel=1,
        progress=False,
        patch_size=SMALL_PATCH,
    )
    assert [p.name for p in written] == ["movie_T000_seg.tif", "movie_T001_seg.tif"]
    for t, path in enumerate(written):
        expected = model.segment(data[t, 1], patch_size=SMALL_PATCH)
        assert np.array_equal(tifffile.imread(path), expected)

    with pytest.raises(ValueError, match="out of range"):
        predict(
            tmp_path / "movie.ome.tif",
            out,
            checkpoint=checkpoint,
            device="cpu",
            channel=5,
            overwrite=True,
            progress=False,
        )


def _write_dataset(folder, n=4):
    """Bright blobs on a noisy background, with their masks."""
    (folder / "raw").mkdir(parents=True)
    (folder / "seg").mkdir()
    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[:20, :48, :48]
    for i in range(n):
        centre = rng.integers(8, 14), rng.integers(12, 36), rng.integers(12, 36)
        blob = (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2 < 49
        raw = rng.normal(100, 10, blob.shape) + 200 * blob
        tifffile.imwrite(folder / "raw" / f"im{i}.tif", raw.astype(np.uint16))
        tifffile.imwrite(folder / "seg" / f"im{i}_seg.tif", blob.astype(np.uint8) * 255)


def test_make_csv_train_and_predict(tmp_path, checkpoint):
    _write_dataset(tmp_path)
    manifest = tmp_path / "train.csv"
    main(
        [
            "make-csv",
            "--raw-dir",
            str(tmp_path / "raw"),
            "--seg-dir",
            str(tmp_path / "seg"),
            "--seg-suffix",
            "_seg",
            "-o",
            str(manifest),
        ]
    )
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and set(rows[0]) == {"raw", "seg"}

    run = tmp_path / "run"
    main(
        [
            "finetune",
            "--train-csv",
            str(manifest),
            "-o",
            str(run),
            "--init",
            str(checkpoint),
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--patch-size",
            "16",
            "32",
            "32",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    best = run / "checkpoints" / "best.ckpt"
    assert best.exists() and (run / "checkpoints" / "last.ckpt").exists()
    assert (run / "metrics.csv").exists()
    assert any((tmp_path / "cache").iterdir())

    # The trained checkpoint feeds straight back into prediction.
    main(
        [
            "predict",
            "-i",
            str(tmp_path / "raw"),
            "-o",
            str(tmp_path / "pred"),
            "-c",
            str(best),
            "--device",
            "cpu",
            "--patch-size",
            "16",
            "32",
            "32",
        ]
    )
    assert len(list((tmp_path / "pred").glob("*_seg.tif"))) == 4


def test_validation_patches_are_repeatable(tmp_path):
    from subcellseg.data import fixed_patches

    _write_dataset(tmp_path, n=2)
    items = [
        {
            "raw": str(tmp_path / "raw" / f"im{i}.tif"),
            "seg": str(tmp_path / "seg" / f"im{i}_seg.tif"),
        }
        for i in range(2)
    ]
    first = fixed_patches(items, SMALL_PATCH, patches_per_image=3, seed=1)
    again = fixed_patches(items, SMALL_PATCH, patches_per_image=3, seed=1)

    assert len(first) == 6
    assert all(p["raw"].shape == (1, *SMALL_PATCH) for p in first)
    assert all(torch.equal(a["raw"], b["raw"]) for a, b in zip(first, again, strict=True))
    assert set(torch.unique(first[0]["seg"]).tolist()) <= {0.0, 1.0}  # 0/255 masks become 0/1


def test_training_changes_weights_and_scratch_runs(tmp_path, checkpoint):
    from subcellseg import train

    _write_dataset(tmp_path, n=3)
    with open(tmp_path / "data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "mask"])
        for i in range(3):  # relative paths, custom column names
            writer.writerow([f"raw/im{i}.tif", f"seg/im{i}_seg.tif"])

    # Worker processes on purpose: validation set-up reads images in the main
    # process first, which used to deadlock forked workers on Linux.
    best = train(
        tmp_path / "data.csv",
        output_dir=tmp_path / "run",
        raw_col="image",
        seg_col="mask",
        max_epochs=1,
        batch_size=2,
        patch_size=SMALL_PATCH,
        num_workers=2,
        device="cpu",
    )
    trained, initial = read_state_dict(best), read_state_dict(checkpoint)
    assert trained.keys() == initial.keys()
    assert any(not torch.equal(trained[k], initial[k]) for k in trained)


def test_weights_download_is_verified_and_cached(tmp_path, monkeypatch):
    from subcellseg import weights

    source = tmp_path / "hosted.pt"
    source.write_bytes(b"pretend weights")
    cache = tmp_path / "cache"
    monkeypatch.setenv("SUBCELLSEG_CACHE_DIR", str(cache))
    monkeypatch.delenv("SUBCELLSEG_WEIGHTS_URL", raising=False)
    monkeypatch.setattr(weights, "WEIGHTS_URL", source.as_uri())

    # A file that does not match the published checksum is rejected and not kept.
    monkeypatch.setattr(weights, "WEIGHTS_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="corrupt"):
        weights.fetch_pretrained()
    assert not list(cache.iterdir())

    monkeypatch.setattr(weights, "WEIGHTS_SHA256", weights.sha256(source))
    path = weights.fetch_pretrained()
    assert path == cache / weights.WEIGHTS_FILENAME
    assert path.read_bytes() == b"pretend weights"

    # Second call uses the cached copy; an unreachable location is never contacted.
    monkeypatch.setattr(weights, "WEIGHTS_URL", (tmp_path / "gone.pt").as_uri())
    assert weights.fetch_pretrained() == path

    # With nothing cached, an unreachable location gives a readable error.
    path.unlink()
    with pytest.raises(RuntimeError, match="Could not download"):
        weights.fetch_pretrained()
