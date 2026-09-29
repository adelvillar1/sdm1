# Use cases — where sdm1 fits

Exploration doc, modeled on the sys1 task-catalog exploration (dev-decisions' judgment calls → sys1's eight registered tasks). Same method: start from real, repeatedly-made decisions in workloads we actually run, keep the ones that fit the model class, register them as tasks.

## What makes a use case fit sdm1

Fit criteria (the sdm1 analog of sys1's "is this a judgment call a model can gate?"):

1. **Tabular or relational input.** The decision reduces to: given rows with columns, predict a value or class per row (or for the table).
2. **Labels exist or accrue cheaply.** In-context learning conditions on labeled context rows — the Kumo-Tabular card works from a few hundred. No labels anywhere → non-fit.
3. **No training infra justified.** The whole point is skipping per-dataset feature engineering and model training. If a project already has an ML pipeline, sdm1 adds little.
4. **Latency tolerance.** The backend is a CUDA subprocess — seconds, not milliseconds. Fine for reports, dashboards, and offline scoring; wrong for inline request-path gating (that's sys1's territory).
5. **Single table first.** Classification now, regression later, all on one table. Multi-table relational is a later plan (KumoRelational).
6. **Data sensitivity manageable.** Rows go to a local GPU worker; telemetry records hashes and shapes only (hard rule in CLAUDE.md). Stay conscious with PII-heavy tables.

## Model-class fit profile (Kumo-Tabular, in-context)

| Strength | Caveat |
|----------|--------|
| Zero/few-shot prediction from a few hundred labeled rows | ICL needs context rows — a cold start with zero labels is a non-fit |
| No per-dataset training or feature engineering | Requires CUDA + a models venv (Python ≥3.11) |
| Classification now, regression later | Single-table now; relational later |
| Strong small/medium-table accuracy-efficiency frontier | Huge datasets: tuned GBDT pipelines may still win; sdm1 targets the "no ML infra" zone |
| `num_estimators` ensembles give distribution-shaped output | Seconds-scale latency — batch/offline scoring, not inline gating |

## Grounded candidates (from our own projects)

Survey of all 22 projects under `~/Projects` (2026-09-29). Eight have real tabular prediction-shaped problems; the rest have no tabular data or only operational ledgers.

| # | Project | Prediction problem | Task type | Why it fits | Priority |
|---|---------|--------------------|-----------|-------------|----------|
| 1 | **commissiontracker** | Flag commissions expected-but-unpaid past N days; cash-flow forecast per agency/supplier; cancellation/clawback risk | classification + regression (dates, amounts) | Dense money/date tables (`Booking`, `CommissionCheck`, `AgentPayment`); labels already exist (anticipated vs received) | **High** — small tables, immediate value |
| 2 | **pampa-wineclub** | RSVP no-show prediction for capacity planning; member churn/renewal; per-event wine demand | classification + regression | `members`/`orders`/`events` tables with seeded history; RSVP outcomes are natural labels | **High** — clean event history |
| 3 | **jobhound** | Predict response/interview likelihood from (overallFit, roleFamily, features) against applied/discarded outcomes; measure re-rank quality over time | classification | `Job` rows already carry LLM fit scores plus lifecycle labels that accrue automatically | **High** — self-labeling |
| 4 | **elo-scenario-lab** | Cross-check Monte Carlo game probabilities with an FM on (ratings, form, rest, odds) features; upset flagging; odds-divergence detection | classification (binary) | Already a forecaster with a calibration loop — the FM becomes a second opinion to measure against | Medium |
| 5 | **cruisingintelligence** | Itinerary price-vs-value anomaly detection; seasonal demand forecast per corridor; lead scoring | regression + classification | 120-model schema and existing value-density scoring; volume may exceed ICL comfort; relational FM is the later unlock | Medium |
| 6 | **pmocockpit** | Predict program at-risk/escalation from `ProgramEvent` trajectories; stalled-program flagging; completion-date forecast | classification + regression | Event log + `Risk`/`Task` tables; labels are eventual program status | Medium |
| 7 | **design-canvas** | Churn prediction from `UsageCounter`/`UsageEvent` + `Subscription`; view forecast for published sites; credit-exhaustion/upsell propensity | classification + regression | Usage series exist; per-user rows are small — needs aggregation into user-level features | Medium |
| 8 | **TTESS** | Predict EOY/summative ratings from accumulated walkthrough/BOY ratings; flag evidence-insufficient dimensions | classification | Rubric ratings are structured and poolable across teachers; row counts per teacher are small (ICL-friendly) | Low-Medium — high sensitivity (employment context) |

## Cross-project patterns → candidate sdm1 tasks

These repeat across projects; each is a candidate pre-registered task (the sdm1 analog of sys1's `change`, `pr_gate`, `issue_triage`…):

- **`overdue_flag`** — expected-but-not-received money/items past a threshold (commissiontracker)
- **`no_show` / `churn`** — will this person show up / stay? (pampa-wineclub, design-canvas)
- **`outcome_likelihood`** — will this row convert / respond / win? (jobhound, elo-scenario-lab)
- **`forecast`** — numeric value next period (cash flow, demand, wait times, views)
- **`anomaly`** — does this row deviate from its peers? (cruisingintelligence price-vs-value)
- **`at_risk`** — trajectory-based escalation flag (pmocockpit)

## Anti-use-cases (when NOT to reach for sdm1)

- **Inline request-path decisions** — seconds-scale CUDA subprocess; inline judgment calls belong to sys1's providers.
- **Zero labeled context rows** and no path to any — ICL has nothing to condition on.
- **Very large datasets** where a tuned GBDT pipeline is already justified.
- **Text-heavy problems** — route to LLM/Jev-class providers via sys1 instead.
- **Strict never-leaves-process-memory data** — rows do go to the local GPU worker (telemetry keeps only hashes), which is fine for our projects but a compliance consideration for external data.

## Proposed follow-up

Register the first batch of tasks once v0.1.0 lands: `overdue_flag` (commissiontracker), `no_show` (pampa-wineclub), `outcome_likelihood` (jobhound) — each already has labels, is single-table, and is small enough for ICL. That plan doubles as the real-world eval harness for sdm1's calibration telemetry.

## Sources

- [nvidia/Kumo-Tabular model card](https://huggingface.co/nvidia/Kumo-Tabular)
- [NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models) · [docs](https://nvidia.github.io/structured-data-models/)
- [Stanford STAR Project](https://star-project.stanford.edu) — relational/tabular FM research (KumoRFM-class models predicting accurately from a few hundred examples or fewer)
