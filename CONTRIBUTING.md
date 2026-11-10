# Contributing to SubCellSeg

Thank you for helping. Bug reports, documentation fixes and new features are
all welcome. This page explains how we work so that your contribution lands
quickly.

## Reporting a problem

Open an [issue](https://github.com/AllenCell/SubCellSeg/issues/new/choose) and
pick the matching template. For a segmentation that looks wrong, the template
asks for the output of `subcellseg info`, the structure that is tagged, and an
example image or screenshot. Without those we usually cannot reproduce it.

Usage questions go to [Discussions](https://github.com/AllenCell/SubCellSeg/discussions).

## Setting up for development

```bash
git clone https://github.com/AllenCell/SubCellSeg
cd SubCellSeg
uv venv --python 3.12
uv pip install --python .venv --torch-backend auto -e ".[dev,zarr]"
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pre-commit install
pytest
```

`pre-commit install` sets up the hooks that format code with ruff and strip
notebook outputs before each commit. Run `pre-commit run --all-files` to apply
them to everything at once.

## Branches, commits and pull requests

- `main` is protected. All changes, including documentation, go through a pull
  request and must pass the `tests` workflow (Linux and Windows, Python 3.10
  and 3.12) and the lint job.
- Branch from `main` and name the branch by purpose: `fix/windows-spawn`,
  `docs/colab-badge`, `feat/nd2-reader`. Contributors from outside the
  organisation work from a fork.
- Commit messages: a short imperative summary line (under 72 characters) and,
  when the reason is not obvious from the diff, a body that explains why.
- Pull requests are squash-merged, so one PR becomes one commit on `main`.
  Keep each PR to one topic. The PR template asks what changed, why, and how
  it was tested.
- One approving review is required. Maintainers may merge small documentation
  fixes themselves once CI is green.
- A change that can alter segmentation output (model code, pre- or
  post-processing, default parameters) must include before-and-after numbers
  from `tools/benchmark/bench_predict.sh` on the three variance images listed
  in the README, so reviewers can see the effect.
- Add a line to `CHANGELOG.md` under *Unreleased* for anything a user would
  notice.

## What not to commit

Model weights, checkpoints, output folders and notebook outputs. The
`.gitignore` and the `check-added-large-files` hook (500 kB) catch most of
these. Weights are distributed as GitHub release assets, never in git.

## Releases

Releases are tagged `vX.Y.Z` from `main`. The release PR bumps the version in
`pyproject.toml`, moves the *Unreleased* section of the changelog under the new
version, and updates the install URL in the README. Model weights have their
own tag series (`weights-v1`, `weights-v2`, ...). A weights asset is never
replaced in place: new weights get a new tag and a package release that pins
the new SHA256.

## AI-assisted contributions

You may use AI tools, under the Allen Institute's rules for coding agents: you
are responsible for every line you submit and must have read and understood
it. Say that a tool was used, in the pull request and as a commit trailer that
names the tool and model, for example `Assisted-by: claude-code (claude-fable-5-1)`.
Pull requests with unreviewed AI-generated code or descriptions will be closed.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind.
