# sdm1

> Structured-data model abstraction: NVIDIA SDM (TabICLv2, TabFM, Kumo*) and other tabular/relational foundation models — one Python API, one result shape, normalized telemetry.
> Repo: https://github.com/adelvillar1/sdm1

**Model-cache paths, API keys, and local venv paths are in `CLAUDE.local.md` (gitignored, auto-loaded alongside this file).** Read it before any backend, model, or credential operation.

---

## Hard rules (non-negotiable)

- **Stdlib-only at runtime.** `dependencies = []` in `pyproject.toml` is a contract. torch, NVIDIA `sdm`, cuDF, and pandas live behind the `[models]` extra in a dedicated venv that backend providers shell out to via `subprocess` — `src/sdm1` never imports them.
- **The normalized result shape is a public contract.** Once v0.1.0 defines it, the result mapping (`table`, `task`, `model`, `models_used`, `predictions`, `telemetry`, `verdict`, `latency_ms`) is consumed by downstream dashboards and tooling. Adding keys is fine; renaming or removing is a breaking change.
- **Never log raw dataset contents.** Telemetry records shapes, stypes, row/column counts, latencies, and input hashes — never cell values or row text. Dashboards and error strings stay on that side of the line.
- **Never print, log, or return API key values.** Provider config views expose whether a key env var is *set*, never its value.
- **Fail open, never raise out of logging.** A broken backend must not break the caller's path; `telemetry.log_record` swallows write errors. Preserve both invariants.
- **Confidence semantics are per-model, not global.** In-context models (TabICLv2-class) and fit/predict models (Kumo-class) do not share a confidence scale — never threshold them with the same number.
- **Attribute vendor models and weights.** NVIDIA SDM code is Apache-2.0 with per-directory third-party notices; model weights (e.g. `nvidia/Kumo-Tabular`) are OpenMDW 1.1. Keep model IDs attributed to their source and carry both license notes when wrapping.
- Default operation target is `main`. Feature branches are normal for work; merging to `main` requires explicit user approval *in the current turn*.

---

## Branch → environment topology

| Branch | Environment | Auto-deploy | Notes |
|--------|-------------|-------------|-------|
| `main` | library release | no | Published library; `pip install sdm1` |
| `feature/*` | local dev | no | |

(No remote environments — sdm1 is a library. Any HTTP service wrapper, if added later, runs wherever the consumer deploys it.)

---

## Where to find things

**Architecture**
- Tech stack & module map → `docs/architecture/overview.md`

**Features**
<!-- Add per-feature deep dives here as you build them -->

**Reference**
- Use cases (what sdm1 is for, grounded in our projects) → `docs/USE-CASES.md`
- Business context → `docs/BUSINESS-CONTEXT.md`
- Troubleshooting → `docs/TROUBLESHOOTING.md`
- State snapshot (test counts, provider registry) → `docs/STATE-SNAPSHOT.md`
- Recent sessions → `docs/recaps/`

---

## Contracts (the artifacts we treat as commitments)

Three layers of documentation act as contracts and must stay in sync as the codebase evolves. Updating them is part of finishing work, not a separate optional task.

| Contract | Location | Cadence | Purpose |
|----------|----------|---------|---------|
| **Plans** | `docs/plans/YYYY-MM-DD-<slug>.md` | Per feature | Pre-work contract: what we're building, acceptance criteria, linked docs to update |
| **Recaps** | `docs/recaps/SESSION-RECAP-YYYY-MM-DD.md` | Per session | Post-work journal: what shipped, criteria status, doc-update follow-ups |
| **Technical reference** | `TECHNICAL-DOCUMENTATION.md` (top level) | Per feature | Developer-onboarding contract: module map, result shape, provider/task registries |
| **Functional reference** | `FUNCTIONAL-SPECIFICATIONS.md` (top level) | Per feature | User-flow contract: usage flows, task semantics, edge cases |

**The cycle is**: draft a plan → implement → write a recap → update the technical and functional contracts. Each phase produces an artifact the next phase consumes.

For trivial changes (typos, single-line fixes, dependency bumps), the cycle compresses: skip the plan, write a brief recap, no contract-doc updates needed.

---

## Common commands

```bash
.venv/bin/python -m pytest -q           # test suite (67 tests)
.venv/bin/ruff check .                  # lint (config in pyproject.toml)
.venv/bin/sdm1 doctor                   # backend availability snapshot (subprocess probe)
.venv/bin/sdm1 models                   # registry listing with attribution metadata
.venv/bin/sdm1 classify -i table.csv -t target --json
```

---

## Today's state

- v0.1.0 foundation implemented 2026-09-29 (plan `2026-09-29-sdm1-foundation`): types/registry/parsers/telemetry/interpret/api/cli + `nvidia-sdm` provider with subprocess worker. 67 tests passing, ruff clean. **Nothing committed yet — the first commit is the user's.**
- Deferred verification: the real end-to-end Kumo-Tabular run needs a CUDA machine with a `[models]` venv (`SDM1_MODELS_PYTHON`); every other criterion is verified locally (the unavailable/fallback path is fully tested).
- Next plan candidates (per `docs/USE-CASES.md`): register `overdue_flag`, `no_show`, `outcome_likelihood` as the first pre-registered tasks.

---

## Housekeeping protocol — keeps the docs tree from rotting

**You are responsible for keeping the docs tree current.** The whole point of slimming this file was to push reference material into topical files; that only works if those files stay fresh. After every session that changes something material, do the following before the user wraps up (or when writing the session recap):

1. **Identify which doc(s) the change touched.** Use the "Where to find things" map above. Changes to a feature → the matching `docs/features/*.md`. Key-env or venv-path change → `CLAUDE.local.md`. Result-shape or module-map change → `docs/architecture/overview.md`.
2. **Update the relevant doc inline.** Don't leave the new fact only in a session recap. Recaps are journal entries; topical docs are the source of truth.
3. **Stat tables go in `docs/STATE-SNAPSHOT.md`, not in topical docs.** When refreshing the snapshot, **replace** it — do not append a new dated section.
4. **CLAUDE.md (this file) only gets updated when:** a hard rule changes; a new top-level docs area is added; the topology changes; or "Today's state" needs a refresh. **Never** add narrative changelogs or feature deep-dives here.
5. **`CLAUDE.local.md` updates require care.** It's gitignored, so changes leave no trace in `git log`. When you rotate a credential or URL, **mention it in the session recap** — that line is the only trace.
5a. **Contract docs (`TECHNICAL-DOCUMENTATION.md`, `FUNCTIONAL-SPECIFICATIONS.md`) must stay in sync with code.** Never declare a feature done while these are stale — track the gap as known debt in the recap if you must defer.
6. **Drift check — run before any session ends with non-trivial doc changes:**
   - `wc -l CLAUDE.md` → should stay ≤ ~300 lines. If growing, move content out.
   - `git status` → must NOT show `CLAUDE.local.md`.
   - For every doc path mentioned in CLAUDE.md, confirm the file exists.
7. **When the user reports "Claude pointed at the wrong model/backend" or "asked me for a key again":** that's a `CLAUDE.local.md` bug. Patch the local file. **Do not just answer the question — fix the source.**
8. **Quarterly hygiene pass:** re-snapshot `docs/STATE-SNAPSHOT.md`; archive recaps older than 90 days into `docs/recaps/archive/`; remove `Updated YYYY-MM-DD` markers from topical docs.

---

## Session protocol

**At the start** of any non-trivial session, run `/warmup`. It loads CLAUDE.md, the latest session recap, all active plans in `docs/plans/`, and the relevant feature/architecture docs before answering anything.

**At the end** of any session that touched anything, run `/wrapup`. It verifies a recap exists, checks "Today's state" freshness, asks for disposition on uncommitted changes, transitions plan status, escalates deferred-update debts, runs drift checks, and drafts a next-session preview.

For trivial sessions (typo fix, one-line change), skip warmup and skip the recap, but **still run `/wrapup`**.

Otherwise:

1. Read this file (auto-loaded).
2. Read `CLAUDE.local.md` if you need any key env var, venv path, or URL (auto-loaded).
3. For task-specific context, read the relevant `docs/` file from the "Where to find things" map.
4. When changes are made, follow the housekeeping protocol above.

**The six-stage cycle**: `warmup → plan → build → recap → wrapup → next session`. Skip the plan for trivial work; skip `/warmup` and `/recap` for trivial sessions; **never skip `/wrapup`** if the session touched anything.
