# Scripts

This directory contains developer utilities, not a second application or package.

## Active utility

`test_graph.py` runs an interactive LangGraph smoke flow with mocked services and no
provider API keys:

```bash
PYTHONPATH=. uv run python scripts/test_graph.py
```

It complements automated tests; it is not a release or model-quality benchmark.

## Legacy duplicate scaffolding

The following paths are historical copies and are not authoritative:

- `scripts/docs/CONTRIBUTING.md` — pointer to `docs/CONTRIBUTING.md`.
- `scripts/ml/` — placeholder duplicate; use root `ml/` for future experiments.

The old `scripts/pyproject.toml` was removed. The root `pyproject.toml` and `uv.lock`
are the only Python dependency sources.

New production code, datasets, training code, or documentation must not be added to
these duplicate paths. Removing the redundant tracked scaffolding can be handled in
a dedicated cleanup change after verifying that no external workflow references it.

See the root `README.md` for setup and `DEVELOPMENT_ROADMAP.md` for planned work.
