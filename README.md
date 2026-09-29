# sdm1

> Structured-data model abstraction: NVIDIA SDM (TabICLv2, TabFM, Kumo*) and other tabular/relational foundation models — one Python API, one result shape, normalized telemetry.

**Status: scaffold.** The project methodology structure (plans, recaps, contract docs) is in place; library implementation is scoped in [`docs/plans/2026-09-29-sdm1-foundation.md`](docs/plans/2026-09-29-sdm1-foundation.md) and has not started.

- Developer reference — [TECHNICAL-DOCUMENTATION.md](TECHNICAL-DOCUMENTATION.md)
- Behavior contract — [FUNCTIONAL-SPECIFICATIONS.md](FUNCTIONAL-SPECIFICATIONS.md)
- Architecture — [docs/architecture/overview.md](docs/architecture/overview.md)
- Upstream inspiration — [NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models) (GPU-native foundation models, tensor subclasses, and processors for structured data)

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

The core is stdlib-only (`dependencies = []`); the torch/NVIDIA-SDM backend stack arrives later behind the `[models]` extra in an isolated venv.
