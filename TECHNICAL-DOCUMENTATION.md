# sdm1 — Technical Documentation

> **For:** Developer onboarding and reference
> **Repo:** https://github.com/adelvillar1/sdm1
> **Production:** n/a (library; pre-release)

This is the developer-onboarding contract — the document a new contributor reads to understand how the system is built. It's intentionally summary-style and links into `docs/` for deep dives. The two layers stay in sync as part of finishing a feature (see CLAUDE.md "Housekeeping protocol").

When a feature ships, update both the relevant `docs/` file (operational reference) **and** the matching section here (summary contract). The recap workflow prompts for both.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Architecture](#3-architecture)
4. [Data Contracts](#4-data-contracts)
5. [Public API Reference](#5-public-api-reference)
6. [Provider Backends](#6-provider-backends)
7. [CLI](#7-cli)
8. [Release & Environments](#8-release--environments)
9. [Development Workflow](#9-development-workflow)
10. [Scripts Reference](#10-scripts-reference)
11. [Observability](#11-observability)

---

## 1. Project Overview

sdm1 is a structured-data model abstraction: one Python API over tabular/relational foundation models — NVIDIA SDM's family (TabICLv2, TabFM, KumoTabular, KumoRelational) first — with a normalized result shape and JSONL telemetry. It applies the sys1 provider-abstraction pattern (one wire shape, normalized results, calibration telemetry, pluggable registry) to the structured-data-model domain.

Differentiators from using a backend directly: routing across backend families through one registry, one normalized result shape for every consumer, JSONL calibration telemetry, and backend swaps without caller changes. Supported task types: `classification`, `regression`, `forecast` (point + quantile band), `anomaly` (expected-distribution scoring: band per query row, coverage confidence against the observed value).

## 2. Tech Stack

| Layer | Technology |
|-------|------------|
| Core | Python ≥3.10, stdlib only (`dependencies = []` is a contract) |
| Backends (hosted) | Prior Labs TabPFN REST API — stdlib `urllib`, Bearer `TABPFN_API_KEY`, no local stack |
| Backends (local) | `[models]` extra: torch 2.7+, NVIDIA `sdm`, Python ≥3.11, CUDA — in a dedicated venv reached via `subprocess` |
| Build | hatchling |
| Test / Lint | pytest, pytest-cov, ruff |

For the full stack rationale, see [`docs/architecture/overview.md`](docs/architecture/overview.md).

## 3. Architecture

Module map (all stdlib-only; the worker below is the only module allowed to import torch/sdm — and it never runs inside this package's interpreter):

| Module | Role |
|--------|------|
| `types.py` | `TableTask`, `Prediction`, `TableResult` (the public wire contract), `Provider`, exceptions |
| `registry.py` | pluggable backend registry (`register`/`get`/`all_ids`) |
| `parsers.py` | tolerant worker-output → `Prediction` normalization (`structured_ok` telemetry key mirrors sys1); hosted predict-response parsing (`parse_predict_response`) |
| `telemetry.py` | JSONL log (`<log_dir>/YYYY/MM/DD/events.jsonl`), redaction, confidence histogram |
| `interpret.py` | per-backend confidence floors, verdict + `low_confidence_rows` |
| `providers/nvidia_sdm.py` | nvidia-sdm backend: venv location, availability probe (subprocess), worker protocol |
| `providers/worker_sdm.py` | runs INSIDE the `[models]` venv: pandas/sdm/torch, KumoTabular ICL call |
| `providers/tabpfn_hosted.py` | tabpfn-hosted backend: Prior Labs REST flow (prepare/upload/fit/predict), auth + quota mapping, per-task-type wire mapping |
| `api.py` | `classify()` composition: row split → run → interpret → JSONL |
| `cli.py` | `sdm1 classify/doctor/models/version` |

Data flow: rows → context/query split (target presence) → provider `run()` → (nvidia-sdm: subprocess worker | tabpfn-hosted: HTTPS + signed uploads) → parse → normalize → interpret → `TableResult` → JSONL.

Detailed architecture: [`docs/architecture/overview.md`](docs/architecture/overview.md)

## 4. Data Contracts

### TableResult keys (public contract — renaming/removing is breaking)

| Key | Type | Meaning |
|-----|------|---------|
| `table` | str | caller's table identifier |
| `task` | str | task id (`<target>:<task_type>` or the `TableTask` id) |
| `model` | str | backend id requested |
| `models_used` | list[str] | backends that produced predictions (empty on error) |
| `task_type` | `classification` \| `regression` \| `forecast` \| `anomaly` | prediction mode (enum grew additively; see §6 wire mapping) |
| `target` | str | the predicted column |
| `n_context_rows` / `n_query_rows` | int | labeled context rows / scored query rows |
| `num_estimators` | int | in-context ensemble size |
| `predictions` | list | `{row_index, label, probabilities, confidence, extras}` per query row — `extras` is additive (forecast: `{"quantiles": {"0.1".."0.9": value}}`; anomaly: quantiles + `actual`) |
| `telemetry` | dict | `worker_returncode`, `worker_latency_ms`, `structured_ok`, `confidence_floor`, `confidence_histogram`; hosted runs add `billing_model_version`, `execution_mode`, `cache_outcome`, `n_estimators`, `tabpfn_task`, `output_type`, `wire_latency_ms`; `error`, `error_kind` on failure |
| `verdict` | `pass` \| `escalate` \| `block` \| `error` | interpretation outcome |
| `escalated` | bool | any row failed the floor |
| `low_confidence_rows` | list[int] | query-row indices below the floor or missing a label |
| `latency_ms` | int | end-to-end wall time |
| `input_sha256` | str | sha256 of the canonicalized query rows — the feedback join key |
| `error` / `reason` | str \| null | failure details (never dataset contents) |
| `summary` | list[str] | human-readable interpretation lines |

### Telemetry JSONL

`$SDM1_LOG_DIR` (default `~/.local/share/sdm1/logs`) `/YYYY/MM/DD/events.jsonl`, one `{"op": "classify", ...result, "ts"}` record per run. Hard rules: never dataset cell values, never key values; writes never raise; `DEFAULT_REDACT` drops `rows`/`cells`/`values`-shaped keys defensively.

### Confidence semantics (per model class — never share floors across classes)

- **In-context backends** (`nvidia-sdm` / KumoTabular): confidence = ensemble mean max-class-probability over `num_estimators` passes — a distribution-shape statistic, not a literal "% sure".
- **Hosted fit/predict backends** (`tabpfn-hosted`): the service's own class probability for the winning class (single head from the caller's view). Closer to a literal "% sure" — never share a floor with nvidia-sdm.
- **Anomaly rows** (any backend): confidence = distribution coverage — the fraction of the predictive distribution inside the tightest symmetric quantile pair containing the observed value (0.8 max on the 9-level hosted band; 0.0 outside every band). The floor reads "at least this much of the distribution must agree with the observation".
- Floor precedence: request-level `confidence_floor` > `task.confidence_floor` > `_DEFAULT_FLOORS[model_id]` (both backends start at 0.6 — calibrate from telemetry). Regression and forecast rows have no confidence and are never floor-gated; a missing label escalates regardless.

## 5. Public API Reference

```python
sdm1.classify(table, *, target=None, task_type="classification", task=None,
              model="nvidia-sdm", stype_overrides=None, num_estimators=8,
              confidence_floor=None, table_name="table", log=True) -> TableResult
sdm1.classify_csv(path, **same_kwargs) -> TableResult
sdm1.health_report() -> dict[str, dict]      # doctor data
sdm1.register(provider) / sdm1.get(id) / sdm1.all_ids()
```

- `table` is rows as dicts (pandas: `df.to_dict("records")`); `classify_csv` reads/coerces a CSV via the stdlib `csv` module.
- **Context/query split**: rows where `target` is present and non-null are the labeled context; rows where it is null/missing get scored. `Prediction.row_index` indexes the query subset.
- `task=` accepts a full `TableTask` (overrides the target/task_type kwargs); a request-level `confidence_floor` overrides the task's floor.
- Raises `ValueError` for malformed input (no target, no context rows, no query rows). Backend failures never raise out — they return `verdict="error"` (fail-open).
- Exceptions exported: `Sdm1Error`, `ProviderUnavailableError`, `WorkerError`.

## 6. Provider Backends

### `tabpfn-hosted` — Prior Labs TabPFN over their REST API

- **Configuration**: `TABPFN_API_KEY` (required, Bearer auth — the value is never logged). `SDM1_TABPFN_BASE` (default `https://api.priorlabs.ai`). `SDM1_TABPFN_TIMEOUT` (default 600s per request — small tasks measured ~160s end-to-end including fit queue on first run; warm runs ~7s).
- **Wire flow** (live-verified 2026-10-07): `POST /tabpfn/prepare_train_set_upload` → PUT x/y CSVs to signed URLs with required headers → `POST /tabpfn/fit {train_set_upload_id, task}` → `POST /tabpfn/prepare_test_set_upload` → PUT x_test → `POST /tabpfn/predict {test_set_upload_id, fitted_train_set_id, task_config}`.
- **Task-type wire mapping**: classification → fit `classification`, predict `output_type: "probas"` (classes from `metadata.classes`); regression → fit `regression`, predict `median`; forecast → fit `regression`, predict `quantiles` (9 levels 0.1–0.9, returned levels × rows); anomaly → same wire as forecast, with the observed value read from `task.metadata["actual_column"]` (default `<target>_actual`, never sent as a feature) and normalized to normal/anomaly + coverage confidence (§4).
- **Telemetry**: `billing_model_version`, `execution_mode`, `cache_outcome`, `n_estimators` from the response metadata; `tabpfn_task`, `output_type`, per-upload and total wire latency; `structured_ok` from the parser.
- **Failure modes**: unset/invalid key → `ProviderUnavailableError` naming `TABPFN_API_KEY` (HTTP 401/403); quota exhaustion → `ProviderUnavailableError` carrying the reset time (HTTP 429); network/signed-upload/other HTTP errors → `WorkerError`; unparseable predict payload → `WorkerError` with `error_kind="parse"`. All surface as `verdict="error"` (fail-open).
- **Privacy**: dataset cells ride only in the CSV bodies uploaded to the vendor's signed URLs; telemetry carries shapes, hashes, and model outputs — never cell values, never the key value.
- **Metering** (account facts, 2026-10-07): token pools (data + prediction tokens) with daily/monthly caps per tier; token cost on v2.x is `max(5000, rows × cols × n_estimators)`, sublinear on v3; balance via the Prior Labs dashboard or `tabpfn-client.get_api_usage()`.

### `nvidia-sdm` — in-context KumoTabular via the `sdm` package

- **Configuration**: `SDM1_MODELS_PYTHON` (python inside the models venv) or `SDM1_MODELS_VENV` (root; `bin/python` derived). `SDM1_TIMEOUT` (default 900s — first run may download weights). `SDM1_WORKER_SCRIPT` (debug/test override).
- **Protocol**: core spawns `<models_python> worker_sdm.py`, writes one JSON request on stdin (`model`, `target`, `task_type`, `feature_columns`, `stype_overrides`, `num_estimators`, `rows_context`, `rows_query`); the worker prints exactly one JSON envelope on stdout (model-stack chatter diverted to stderr), returning `{ok, model_name, predictions, warnings}` or `{ok: false, error}`.
- **Worker**: builds `sdm.TableTensor.from_pandas` + `sdm.infer_stypes` (target override: categorical for classification / numerical for regression), calls `KumoTabular(task, device="cuda")` in-context, extracts class names best-effort from the categorical stype (falls back to synthesized names with a `warnings` entry).
- **Availability**: `doctor` probes `<models_python> -c "import torch, sdm ..."` in a subprocess (cached 60s) — torch is never imported in-process.
- **Failure modes** (all → `verdict="error"` with actionable reason): unconfigured env vars, spawn failure, timeout, nonzero worker exit, unparseable output.

Later registry candidates: TabICLv2, TabFM, KumoRelational (relational), fit/predict mode.

Attribution rule: NVIDIA SDM code is Apache-2.0 with per-directory third-party notices; Kumo-Tabular weights are OpenMDW 1.1 — model IDs stay attributed to their source. Prior Labs TabPFN is a hosted commercial API (account tier governs model versions and quota); no weights are redistributed here.

## 7. CLI

```
sdm1 classify --input FILE.csv --target COL [--task-type classification|regression|forecast|anomaly]
             [--model nvidia-sdm|tabpfn-hosted] [--num-estimators N] [--floor F] [--table-name NAME]
             [--no-log] [--json]
sdm1 doctor [--json]      # per-provider availability snapshot (hosted probe or subprocess probe — never imports torch)
sdm1 models [--json]      # registry listing with metadata/attribution
sdm1 version
```

Exit codes: 0 pass · 1 escalate · 2 error.

## 8. Release & Environments

| Branch | Environment | Auto-deploy | Notes |
|--------|-------------|-------------|-------|
| `main` | library release | no | Published library; `pip install sdm1` |
| `feature/*` | local dev | no | |

No remote environments — sdm1 is a library. Backend/model paths live in `CLAUDE.local.md` (gitignored).

## 9. Development Workflow

The plan-build-recap-document cycle:

1. **Plan** — draft a plan at `docs/plans/YYYY-MM-DD-<slug>.md` with acceptance criteria. See [`docs/plans/README.md`](docs/plans/README.md).
2. **Build** — implement on a `feature/*` branch, validate locally first.
3. **Recap** — write a session recap at `docs/recaps/SESSION-RECAP-YYYY-MM-DD.md` with criteria status.
4. **Document** — update this file and `FUNCTIONAL-SPECIFICATIONS.md` for the affected area. The recap workflow prompts for this.

The cycle compresses for trivial work — typos and one-line fixes don't need a plan or doc updates.

## 10. Scripts Reference

<add-when-implemented>
- Most-used scripts and common flags
</add-when-implemented>

## 11. Observability

- **JSONL**: `<log_dir>/YYYY/MM/DD/events.jsonl` (`SDM1_LOG_DIR` env or `~/.local/share/sdm1/logs`), one record per run including errors — the calibration set.
- **Redaction**: records never contain dataset cell values or API key values; `DEFAULT_REDACT` (`rows`, `rows_context`, `rows_query`, `cells`, `values`, `text`, `content`) is applied defensively on every write; `log_record` swallows write failures.
- **Histograms**: `telemetry.confidence_histogram` buckets prediction confidences into 10 bins for dashboards.
- **Join key**: `input_sha256` (canonicalized query rows) lets feedback/calibration rows join back to predictions.
