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
- Backend stack (CUDA machine, Python ≥3.11): `python3.11 -m venv <dir> && <dir>/bin/pip install 'sdm1[models]'` — pulls `structured-data-models` (torch, NVIDIA `sdm`) plus the Kumo-Tabular weights on first run.
- Point sdm1 at the venv: `SDM1_MODELS_PYTHON=<dir>/bin/python` (or `SDM1_MODELS_VENV=<dir>`). Optional: `SDM1_LOG_DIR`, `SDM1_TIMEOUT`.
- Verify: `sdm1 doctor` reports availability without importing torch; `sdm1 models` lists backends with attribution metadata.

## 2. Core Concepts

- **TableTask** — a named prediction job: `target` column, `task_type` (classification/regression), optional `stype_overrides` (column → semantic type), `num_estimators` (in-context ensemble size), and a `confidence_floor` expressed in the serving backend's own semantics.
- **Provider / backend** — a registered model family (`nvidia-sdm` first) reached through `sdm1.get(id)`; availability is probed by subprocess, never by importing the stack.
- **Context/query split** — rows with a non-null `target` are labeled context (in-context examples); rows with a null/missing `target` are scored. `Prediction.row_index` indexes the query rows.
- **Prediction** — `{row_index, label, probabilities, confidence}`; classification carries class probabilities and an ensemble-derived confidence, regression carries a value only.
- **TableResult** — the normalized result contract (TECHNICAL-DOCUMENTATION §4): same shape regardless of backend, with verdict, low-confidence rows, and telemetry.
- **Verdict** — `pass` (all rows clear the floor), `escalate` (some rows below floor or unlabeled), `error` (backend failure — fail-open, never raises out).

## 3. Core Features

- **Score a table** (`sdm1.classify` / `classify_csv` / `sdm1 classify`): give labeled + unlabeled rows; get per-row predictions, confidences, and a verdict. Entry points: the Python API, or the CLI against a CSV.
- **Backend routing**: `model="nvidia-sdm"` by default; registry is pluggable (`sdm1.register`) so future backends (TabICLv2, TabFM, KumoRelational) slot in without API changes.
- **Confidence interpretation** (`sdm1.interpret`): per-backend floors; result names exactly which query rows fell below the floor (`low_confidence_rows`).
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

<add-when-implemented>
- What every run records (JSONL) and what it must never record (dataset contents, key values)
- Confidence histogram per model
- How consumers build dashboards on the telemetry stream
</add-when-implemented>

## 7. Edge Cases & Error States

- **No labeled rows** (every row has a null target) → `ValueError` "no context rows — in-context learning needs labeled example rows".
- **No rows to score** (every row labeled) → `ValueError` "no query rows".
- **Backend stack missing** (no `SDM1_MODELS_PYTHON`, no torch, no CUDA, no `sdm`) → `verdict="error"` with an actionable message naming the missing piece; never a bare traceback.
- **Worker failure** (nonzero exit, timeout after `SDM1_TIMEOUT`, unparseable output) → error verdict; `telemetry.error_kind` distinguishes `worker_exit` / `timeout` / `parse` / `spawn`.
- **Zero predictions from a healthy worker** → error verdict ("backend returned zero predictions").
- **Unknown class vocabulary** → worker synthesizes placeholder class names and records a warning in the result envelope.
- **Regression rows** carry no confidence and are never floor-gated; a classification row missing a label escalates even at high confidence.
- **Logging is fail-open**: a broken log dir never breaks a run (telemetry returns `None`).

## 8. Versioning & Compatibility

<add-when-implemented>
- Result-shape semver: adding keys is minor, renaming/removing is major
- Telemetry key stability guarantees for dashboard consumers
- Deprecation policy for provider IDs and task names
</add-when-implemented>
