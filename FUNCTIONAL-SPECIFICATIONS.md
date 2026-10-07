# sdm1 — Functional Specifications

> **For:** Product/UX reference and developer onboarding (user-flow contract)
> **Repo:** https://github.com/adelvillar1/sdm1
> **Production:** n/a (library; pre-release)

This is the user-flow contract — the document that describes what the system does from a user's perspective. Here the "user" is a developer consuming the library. It captures intended behavior, edge cases, and the rules that govern features. It's the answer to "what should this library DO?"

This file stays in sync with the implementation as part of finishing a feature (see CLAUDE.md "Housekeeping protocol"). When a feature ships or changes, update both [`docs/features/<name>.md`](docs/features/) (operational reference) and the matching section here (user-flow contract). The recap workflow prompts for both.

---

## Table of Contents

1. [Installation & Setup](#1-installation--setup)
2. [Core Concepts](#2-core-concepts)
3. [Core Features](#3-core-features)
4. [Usage Flows](#4-usage-flows)
5. [Task Registry & Custom Tasks](#5-task-registry--custom-tasks)
6. [Telemetry & Calibration](#6-telemetry--calibration)
7. [Edge Cases & Error States](#7-edge-cases--error-states)
8. [Versioning & Compatibility](#8-versioning--compatibility)

---

## 1. Installation & Setup

- `pip install sdm1` — the core is stdlib-only and installs anywhere (Python ≥3.10).
- **Hosted backend (no local stack)**: `export TABPFN_API_KEY=<your Prior Labs key>` — the `tabpfn-hosted` backend is immediately usable (stdlib HTTP). Optional: `SDM1_TABPFN_BASE`, `SDM1_TABPFN_TIMEOUT`.
- Local backend stack (CUDA machine, Python ≥3.11): `python3.11 -m venv <dir> && <dir>/bin/pip install 'sdm1[models]'` — pulls `structured-data-models` (torch, NVIDIA `sdm`) plus the Kumo-Tabular weights on first run.
- Point sdm1 at the venv: `SDM1_MODELS_PYTHON=<dir>/bin/python` (or `SDM1_MODELS_VENV=<dir>`). Optional: `SDM1_LOG_DIR`, `SDM1_TIMEOUT`.
- Verify: `sdm1 doctor` reports every backend's availability with the reason (hosted probe or subprocess probe — never importing torch); `sdm1 models` lists backends with attribution metadata.

## 2. Core Concepts

- **TableTask** — a named prediction job: `target` column, `task_type` (classification/regression/forecast/anomaly), optional `stype_overrides` (column → semantic type), `num_estimators` (in-context ensemble size), and a `confidence_floor` expressed in the serving backend's own semantics. Anomaly tasks may carry `metadata["actual_column"]` (the observed-value column; default `<target>_actual`).
- **Provider / backend** — a registered model family (`tabpfn-hosted`, `nvidia-sdm`) reached through `sdm1.get(id)`; availability is probed without importing any model stack (hosted: a cheap authed GET; local: a subprocess probe).
- **Context/query split** — rows with a non-null `target` are labeled context (in-context examples); rows with a null/missing `target` are scored. `Prediction.row_index` indexes the query rows.
- **Prediction** — `{row_index, label, probabilities, confidence, extras}`; classification carries class probabilities and a backend-derived confidence; regression/forecast carry values (forecast adds the quantile band in `extras`); anomaly carries normal/anomaly plus coverage confidence and the observed actual in `extras`.
- **TableResult** — the normalized result contract (TECHNICAL-DOCUMENTATION §4): same shape regardless of backend, with verdict, low-confidence rows, and telemetry.
- **Verdict** — `pass` (all rows clear the floor), `escalate` (some rows below floor or unlabeled), `error` (backend failure — fail-open, never raises out).

## 3. Core Features

- **Score a table** (`sdm1.classify` / `classify_csv` / `sdm1 classify`): give labeled + unlabeled rows; get per-row predictions, confidences, and a verdict. Entry points: the Python API, or the CLI against a CSV.
- **Backend routing**: pick a backend with `model=` (`tabpfn-hosted` for the no-stack hosted path, `nvidia-sdm` for a local CUDA venv); the registry is pluggable (`sdm1.register`) so future backends (TabICLv2, TabFM, KumoRelational) slot in without API changes.
- **Task types**: classification (class probabilities), regression (point values), forecast (median + 0.1–0.9 quantile band per row), anomaly (expected-distribution scoring — the observed value rides in `actual_column`, the result carries normal/anomaly + coverage confidence so callers can gate on "the band must contain the observation").
- **Confidence interpretation** (`sdm1.interpret`): per-backend floors (hosted single-head vs in-context ensemble are different scales); result names exactly which query rows fell below the floor (`low_confidence_rows`).
- **Calibration logging**: every run appends one JSONL record keyed by `input_sha256` — the dataset for floor calibration later.
- **Availability snapshot** (`sdm1 doctor`): what's configured, what's probeable, why something is down — without touching torch.

## 4. Usage Flows

- **Single-table classification**: fill `target` on known rows, leave it null on new rows, call `classify` — predictions come back per query row with probabilities.
- **Backend unavailable**: `classify` returns `verdict="error"` with an actionable reason (e.g. "set SDM1_MODELS_PYTHON…") instead of raising; fix the env and rerun.
- **CSV scoring from the shell**: `sdm1 classify -i data.csv -t outcome` — human summary or `--json` for pipes; exit code 0/1/2 mirrors pass/escalate/error.
- **Floor tuning**: request-level `--floor`/`confidence_floor` overrides the task floor; calibrate over time from the JSONL histogram.

## 5. Task Registry & Custom Tasks

- v0.1.0 takes inline task specs (`target` + kwargs, or a `TableTask` instance).
- Planned (per `docs/USE-CASES.md`): pre-registered tasks (`overdue_flag`, `no_show`, `outcome_likelihood`, …) with TOML/JSON loading, following the sys1 task-registry pattern.

## 6. Telemetry & Calibration

- Every run appends one JSONL record to `$SDM1_LOG_DIR/YYYY/MM/DD/events.jsonl`: the full `TableResult` wire dict plus `ts`. Records never contain dataset cell values or key values (defensive `DEFAULT_REDACT` drops rows/cells/values-shaped keys; the redaction test proves distinctive cell values never reach the log).
- Hosted records add the vendor metering/execution fields (`billing_model_version`, `execution_mode`, `cache_outcome`, `n_estimators`) — the cost-accrual keys consumers need.
- Each record carries `confidence_histogram` (10 buckets) and `confidence_floor` so dashboards can plot calibration drift per backend.
- Consumers join records back via `input_sha256`; human verdicts appended against that key are the calibration set for fitting per-backend floors.

## 7. Edge Cases & Error States

- **No labeled rows** (every row has a null target) → `ValueError` "no context rows — in-context learning needs labeled example rows".
- **No rows to score** (every row labeled) → `ValueError` "no query rows".
- **Hosted backend unconfigured** (no `TABPFN_API_KEY`) → `verdict="error"` naming the env var; invalid/rejected key (HTTP 401/403) → same error class with the status code.
- **Hosted quota exhausted** (HTTP 429) → `verdict="error"` carrying the reset time from the response headers; retry after the window or switch `model=`.
- **Hosted wire failure** (unreachable, signed-upload failure, other HTTP errors) → `verdict="error"` with `telemetry.error_kind`; small-task latency is minutes-scale when the fit queue is cold — batch callers should expect it.
- **Hosted unparseable predict payload** → `verdict="error"`, `telemetry.error_kind="parse"`, `structured_ok=False`.
- **Local stack missing** (no `SDM1_MODELS_PYTHON`, no torch, no CUDA, no `sdm`) → `verdict="error"` with an actionable message naming the missing piece; never a bare traceback.
- **Worker failure** (nonzero exit, timeout after `SDM1_TIMEOUT`, unparseable output) → error verdict; `telemetry.error_kind` distinguishes `worker_exit` / `timeout` / `parse` / `spawn`.
- **Zero predictions from a healthy backend** → error verdict ("backend returned zero predictions").
- **Unknown class vocabulary** → local worker synthesizes placeholder class names and records a warning; hosted classification reads classes from `metadata.classes`.
- **Regression/forecast rows** carry no confidence and are never floor-gated; anomaly rows carry coverage confidence and gate like classification; a classification row missing a label escalates even at high confidence.
- **Logging is fail-open**: a broken log dir never breaks a run (telemetry returns `None`).

## 8. Versioning & Compatibility

- Result-shape semver: adding keys (top-level or per-prediction `extras`) is minor; renaming/removing any §4 key is major and is tripped by `tests/test_types.py::test_as_dict_matches_contract_key_set`.
- The `task_type` enum grows additively (`classification`, `regression`, then `forecast`, `anomaly`) — consumers must tolerate unknown task types gracefully.
- Telemetry keys are additive; consumers read with `.get()` defaults.
- Provider IDs are stable registry keys (`tabpfn-hosted`, `nvidia-sdm`); a removed ID raises `KeyError` naming the registered set.
