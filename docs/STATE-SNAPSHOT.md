# State snapshot

Refresh by **replacing** this file (never appending dated sections) — see the housekeeping protocol in CLAUDE.md.

| Stat | Value | As of |
|------|-------|-------|
| Tests passing | 67 | 2026-09-29 |
| Ruff | clean | 2026-09-29 |
| Backends registered | 1 (`nvidia-sdm` — unconfigured until a models venv exists; `sdm1 doctor` reports it) | 2026-09-29 |
| Pre-registered tasks | 0 (inline task specs; task registry planned per `docs/USE-CASES.md`) | 2026-09-29 |
| Version | 0.1.0 (uncommitted — first commit belongs to the user) | 2026-09-29 |

## 2026-10-07 — tabpfn-hosted provider + forecast/anomaly task types (feature/tabpfn-hosted)

- `providers/tabpfn_hosted.py`: Prior Labs REST flow (prepare_train_set_upload → signed-URL PUTs → fit → prepare_test_set_upload → predict), stdlib urllib, Bearer TABPFN_API_KEY; 401/403/429 → actionable ProviderUnavailableError (429 carries reset); billing/execution/cache telemetry keys. Live-verified classification + quantile wire 2026-10-07.
- `types.py`: task_type enum grew forecast + anomaly (additive); Prediction gained `extras` (quantile bands, anomaly actual). 19-key top-level contract untouched.
- `parsers.py`: `parse_predict_response` for probas/median/quantiles/full shapes; anomaly coverage = tightest symmetric band containing the actual (0.8 max). Live wire fact: classes live in `metadata.classes`.
- `interpret.py`: `tabpfn-hosted` floor class (0.6 start); anomaly coverage semantics documented.
- 80 tests green, ruff clean. Not yet on main — feature branch pending merge approval.

