# Session recap — 2026-09-29

**Plan link**: `docs/plans/2026-09-29-sdm1-foundation.md` (completed)

## What shipped

- **v0.1.0 foundation, full stop.** `types.py` (TableTask/Prediction/TableResult wire contract), `registry.py`, `parsers.py` (tolerant worker-output normalization), `telemetry.py` (JSONL + redaction + histogram), `interpret.py` (per-backend-class confidence floors), `api.py` (`classify`/`classify_csv`/`health_report`, fail-open error semantics), `cli.py` (`classify/doctor/models/version`), and the `nvidia-sdm` provider (`providers/nvidia_sdm.py` + `worker_sdm.py` subprocess worker targeting `nvidia/Kumo-Tabular` via `sdm.models.KumoTabular`).
- Earlier in the session (pre-plan): full init-project-structure scaffold, `docs/USE-CASES.md` use-case exploration (22-project survey), Kumo-Tabular model-card research (in-context API, OpenMDW 1.1 weights vs Apache-2.0 code — both in the hard rules).

## Criteria status

All 9 acceptance criteria checked — per-criterion evidence in the plan's "Outcome" section. 67 tests passing, ruff clean, doctor/models/version verified on this machine.

## Deferred (tracked)

- **Real end-to-end Kumo-Tabular run** — needs a CUDA machine with a `[models]` venv (`python3.11 -m venv <dir> && <dir>/bin/pip install 'sdm1[models]'`, then `SDM1_MODELS_PYTHON=<dir>/bin/python`). Worker protocol fully covered by stub tests locally.
- First commit — nothing committed yet; the user owns it.

## Doc updates made

TECHNICAL-DOCUMENTATION §3–§7 + §11, FUNCTIONAL-SPECIFICATIONS §1–§3 + §5 + §7, `docs/architecture/overview.md`, `docs/STATE-SNAPSHOT.md`, CLAUDE.md (commands + Today's state). No `CLAUDE.local.md` changes this session (placeholders still to be filled by the user: NVIDIA NGC key, HF token, models venv path).

## Next-session preview

Register the first three pre-registered tasks — `overdue_flag` (commissiontracker), `no_show` (pampa-wineclub), `outcome_likelihood` (jobhound) — per `docs/USE-CASES.md`; that plan doubles as the calibration eval harness. Optionally: models-venv setup runbook for the CUDA box.
