---
status: completed
created: 2026-09-29
updated: 2026-09-29
slug: sdm1-foundation
---

# sdm1 v0.1.0 foundation — types, registry, telemetry, NVIDIA SDM backend

## Context

NVIDIA's [structured-data-models](https://github.com/NVIDIA/structured-data-models) (SDM) ships GPU-native foundation models for tabular/relational data (TabICLv2, TabFM, KumoTabular, KumoRelational) plus tensor subclasses and processors. It's directly useful, but consumers bind to NVIDIA's API surface directly — no normalized result shape across model families, no telemetry stream, no availability/fallback layer.

sdm1 applies the sys1 pattern (proven in `~/Projects/sys1`: one Python API, one wire shape, normalized results, JSONL telemetry, pluggable provider registry, stdlib-only core) to this domain. Scope decided with the user on 2026-09-29: **sys1-style abstraction**, name **sdm1**, **stdlib-only core** with backends behind extras.

First backend pinned 2026-09-29: **[`nvidia/Kumo-Tabular`](https://huggingface.co/nvidia/Kumo-Tabular)** — NVIDIA's pretrained tabular foundation model for classification/regression, used through `sdm.models.KumoTabular`'s in-context API. Install: `pip install structured-data-models` (pulls torch; Python ≥3.11, CUDA). Weights are OpenMDW 1.1 (the SDM *code* is Apache-2.0 — both notes carried).

## Approach

Mirror sys1's module layout so the house pattern transfers directly:

- `src/sdm1/types.py` — `TableTask`, `Prediction`, `TableResult` dataclasses. Wire shape is designed once here and then becomes a public contract.
- `src/sdm1/registry.py` — pluggable backend registry (`sdm1.register(...)`), availability probing without importing torch.
- `src/sdm1/providers/` — one module per backend family. First: `nvidia_sdm.py` targeting **KumoTabular** (`nvidia/Kumo-Tabular` weights), shelling out to a dedicated `[models]` venv (`pip install structured-data-models`, Python ≥3.11, CUDA) via a small worker script that does the data prep (`sdm.TableTensor.from_pandas` + `sdm.infer_stypes` with overrides) and the in-context call (`model(x_context=…, y_context=…, x_query=…, num_estimators=8)`) — the core never imports torch (same subprocess pattern as sys1's `local` provider).
- `src/sdm1/parsers.py` — per-backend normalization to the shared `TableResult` shape, tolerant of the multiple output shapes SDM models return (in-context forward vs fit/predict).
- `src/sdm1/telemetry.py` — JSONL log with redaction (never dataset contents, never key values), latency/retry keys, confidence histogram; records backend/model IDs and `num_estimators`.
- `src/sdm1/interpret.py` — verdict layer over per-model-class confidence semantics (in-context vs fit/predict are not thresholded alike — sys1's Jev-vs-Decide lesson).
- `src/sdm1/cli.py` — `doctor` (availability snapshot), `models` (registry listing); task execution lands with the first runnable backend.

## Acceptance criteria

- [x] `src/sdm1/types.py` defines `TableTask`, `Prediction`, `TableResult`; the `TableResult` key set is documented in `TECHNICAL-DOCUMENTATION.md` §4 before any provider code lands
- [x] `src/sdm1/registry.py` supports `register()`/lookup and availability probing that never imports torch or `sdm`
- [x] `nvidia-sdm` provider runs a classification task end-to-end against `nvidia/Kumo-Tabular` through the `[models]` venv via subprocess, or reports unavailable with an actionable message when the venv/torch/CUDA is missing
- [x] parsers normalize the in-context output (ensemble probabilities over `num_estimators` passes) to the shared `TableResult` shape
- [x] telemetry JSONL record keys are documented; records never contain dataset cell values or API key values (redaction test proves it)
- [x] confidence interpretation documents and enforces per-model-class semantics (in-context vs fit/predict floors are separate)
- [x] `sdm1 doctor` prints a backend availability snapshot with zero runtime dependencies
- [x] full test suite passes with only `pytest` installed (core stdlib-only contract holds)
- [x] `TECHNICAL-DOCUMENTATION.md` §4–§7 and `FUNCTIONAL-SPECIFICATIONS.md` §2–§3, §7 updated to match what shipped

## Files to be touched

- `src/sdm1/types.py`, `registry.py`, `parsers.py`, `telemetry.py`, `interpret.py`, `cli.py` (new)
- `src/sdm1/providers/nvidia_sdm.py` + worker script for the models venv (new)
- `pyproject.toml` (`[models]` extra pins, console script entry point)
- `tests/` — unit tests per module + redaction/subprocess fallback tests
- `docs/architecture/overview.md`, `TECHNICAL-DOCUMENTATION.md`, `FUNCTIONAL-SPECIFICATIONS.md`, `docs/STATE-SNAPSHOT.md`

## Out of scope

- Relational (multi-table) tasks — single-table first; `KumoRelational` is a later plan
- fit/predict mode and regression tasks — the pinned path is in-context classification per the model card; extend once the first backend ships
- Wrapping non-SDM backends (other tabular FM families) — later plans, the registry makes them cheap
- An HTTP service wrapper (`sys1`-style `service/`) — only if a consumer needs it
- Training or fine-tuning anything — sdm1 routes and normalizes, never trains
- Publishing to PyPI

## Verification

- `.venv/bin/python -m pytest -q` — full suite green
- `.venv/bin/ruff check .` — clean
- `.venv/bin/python -m sdm1.cli doctor` — prints snapshot without torch installed
- `grep -c 'cell\|row_text' src/sdm1/telemetry.py` style review + redaction unit test for the no-dataset-contents rule
- Manual: with the `[models]` venv present on a CUDA machine, one end-to-end task on a small DataFrame — the Kumo-Tabular card example is the smoke shape (context rows → query rows → class probabilities, `num_estimators=8`). In tests: a tiny synthetic table with the worker mocked — no GPU, no external downloads in the default test path

## Linked artifacts

- `docs/architecture/overview.md` — module map + data flow
- `TECHNICAL-DOCUMENTATION.md` — §4 Data Contracts, §5 Public API, §6 Provider Backends, §7 CLI, §11 Observability
- `FUNCTIONAL-SPECIFICATIONS.md` — §2 Core Concepts, §3 Core Features, §7 Edge Cases
- `docs/STATE-SNAPSHOT.md` — test counts + registry state after completion

## Outcome (2026-09-29) — per-criterion evidence

1. **Types + documented key set** — `src/sdm1/types.py` defines `TableTask`/`Prediction`/`TableResult`; the 19-key contract is tabled in TECHNICAL-DOCUMENTATION §4 and frozen by `tests/test_types.py::test_as_dict_matches_contract_key_set`.
2. **Registry without torch imports** — `registry.py` register/get/all_ids; the availability probe runs `subprocess` (`nvidia_sdm.probe_stack`); no `import torch`/`import sdm` anywhere under `src/sdm1` outside `worker_sdm.py` (which never runs in-process).
3. **nvidia-sdm via subprocess, or actionable unavailability** — `run()` spawns the models-venv python with the worker; the unconfigured path raises `ProviderUnavailableError` naming `SDM1_MODELS_PYTHON` (tested). *Deferred*: the real end-to-end Kumo-Tabular run requires a CUDA machine + models venv — not available on this machine; worker protocol is covered by stub-worker tests (`test_providers_nvidia_sdm.py`).
4. **Parsers normalize the in-context output** — ensemble probabilities → `TableResult` predictions, tolerant of key aliases/list forms/derived confidence (`test_parsers.py`, 12 cases).
5. **Telemetry documented + redaction proven** — JSONL keys in TECH-DOC §4; `test_telemetry.py` proves distinctive dataset cell values never reach the log and redaction drops `rows`/`cells`/`values`-shaped keys; records carry model IDs + `num_estimators`.
6. **Per-model-class confidence semantics** — `interpret.py` docstring + `_DEFAULT_FLOORS` keyed per backend id; precedence tests in `test_interpret.py`; regression rows never gate.
7. **`sdm1 doctor` with zero runtime deps** — verified: prints the availability snapshot with no torch installed; `--json` form tested.
8. **Suite green with pytest only** — 67 tests passing, `.venv` holds only the dev extras; ruff clean.
9. **Contract docs updated** — TECH-DOC §3–§7, §11; FUNCTIONAL-SPECIFICATIONS §1–§3, §5, §7; `docs/architecture/overview.md` (module map + data flow); `docs/STATE-SNAPSHOT.md` replaced; CLAUDE.md commands + Today's state refreshed.

Follow-ups spawned: register the first three tasks (`overdue_flag`, `no_show`, `outcome_likelihood`) — see `docs/USE-CASES.md`; real-GPU end-to-end run + models-venv setup runbook on a CUDA machine.
