# Contributing to BdPy

## Issues

Bug reports and feature requests: https://github.com/KamitaniLab/bdpy/issues

## Development setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```shell
git clone https://github.com/KamitaniLab/bdpy.git
cd bdpy
uv sync
```

## Pull requests

Open PRs against the `dev` branch. Make sure CI passes before requesting review.

## Tests, lint, type check

```shell
uv run pytest
uv run ruff check
uv run mypy bdpy
```

## Docstring style

NumPy style. The API reference on the documentation site is auto-generated from
docstrings, so keeping them accurate and well-formatted matters.

## Building the documentation locally

```shell
uv sync --group docs
uv run mkdocs serve   # preview at http://127.0.0.1:8000
uv run mkdocs build   # static build written to site/
```

The site is published to GitHub Pages automatically on push to `main`.
