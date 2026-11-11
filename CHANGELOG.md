# Changelog

All notable changes to SubCellSeg are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed
- `predict` defaults now match the paper: 30% window overlap and uniform
  (`constant`) blending, instead of 50% and Gaussian. Pass
  `--overlap 0.5 --blend gaussian` for the previous behaviour. A voxel is
  foreground when its probability is at least the threshold (was: above), so
  `_seg.tif` is exactly `_prob.tif >= 128`.
- `train` and `finetune` defaults now match the paper: batch size 32 (was 8),
  at most 1000 epochs (was 700), 20% of the images held out for validation
  (was 10%). Fine-tuning uses a learning rate of 5e-5 and stops after 50
  epochs without improvement; training from scratch keeps 1e-4 and 100.

### Fixed
- The progress line for the first-time weights download no longer floods
  logs: in a terminal it redraws once per percent, and when output goes to a
  file, CI or a notebook it writes one line per 10% (was about 8,000 redraws,
  ~150 KB). ([#5](https://github.com/AllenCell/SubCellSeg/issues/5))

### Added
- Benchmarks section in the README and `tools/benchmark/` scripts.
- Contributing guide, code of conduct, security policy, citation file,
  issue and pull request templates.
- ruff formatting and linting, pre-commit hooks, lint job in CI.

## 0.1.0 - 2026-11-05

First public release: pretrained model, `subcellseg predict`, `train`,
`finetune` and `make-csv` commands, Python API, quickstart notebook.
