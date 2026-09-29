# Architecture overview

- **Stack**: Python ≥3.10, stdlib-only core (`dependencies = []` is a contract). torch + NVIDIA `sdm` (Python ≥3.11, CUDA) live behind the `[models]` extra in a dedicated venv reached via `subprocess` — the core never imports them.
- **Shape**: mirrors sys1's provider-abstraction pattern — one normalized result shape (`TableResult`), a pluggable backend registry, JSONL telemetry, and per-model-class confidence interpretation.

## Module map

| Module | Role |
|--------|------|
| `types.py` | `TableTask`, `Prediction`, `TableResult` (wire contract), `Provider`, exceptions |
| `registry.py` | backend registry (`register`/`get`/`all_ids`) |
| `parsers.py` | tolerant worker-output → `Prediction` normalization |
| `telemetry.py` | JSONL log, redaction, confidence histogram |
| `interpret.py` | per-backend confidence floors, verdict, `low_confidence_rows` |
| `providers/nvidia_sdm.py` | backend: venv location, availability probe, worker protocol |
| `providers/worker_sdm.py` | runs inside the `[models]` venv (pandas/sdm/torch, KumoTabular ICL) |
| `api.py` | `classify()` composition |
| `cli.py` | `sdm1 classify/doctor/models/version` |

## Data flow

```
rows (list[dict])
  → context/query split (target presence)          api.py
  → provider.run(ctx, query, task, telemetry)      registry → providers/nvidia_sdm.py
      → subprocess: <models_python> worker_sdm.py  [models] venv — pandas/sdm/torch
          → TableTensor.from_pandas + infer_stypes → KumoTabular(x_context, y_context, x_query)
          → JSON envelope on stdout
  → parse_worker_output → [Prediction]             parsers.py
  → interpret → verdict, low_confidence_rows       interpret.py
  → TableResult (public contract)                  types.py
  → JSONL record (op=classify, input_sha256, …)    telemetry.py
```

## Confidence semantics

Per model class, never shared: in-context backends (`nvidia-sdm`) emit an ensemble mean max-class-probability over `num_estimators` passes; future fit/predict backends emit single-head max-probability. Floors are keyed per backend id with the precedence request-level > task > default (`_DEFAULT_FLOORS`, nvidia-sdm 0.6). See `interpret.py` and TECHNICAL-DOCUMENTATION §4.

Upstream reference: [NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models) ([docs](https://nvidia.github.io/structured-data-models/)) — GPU-native foundation models, tensor subclasses (`TableTensor`), semantic types (`stypes`), and composable processors for tabular/relational data. First backend: **[`nvidia/Kumo-Tabular`](https://huggingface.co/nvidia/Kumo-Tabular)** — `sdm.models.KumoTabular` with the in-context API. Later candidates via the registry: TabICLv2, TabFM, KumoRelational, fit/predict mode.
