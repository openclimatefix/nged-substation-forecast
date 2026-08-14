# Engineering health

> **Status: 🚧 Planned.** Tooling, reproducibility, and rigour improvements from the 2026-07
> codebase review that don't change forecast behaviour. Each section is an independent,
> roughly one-PR piece of work; sections are deleted as they ship. The v0.2 epic
> [#138](https://github.com/openclimatefix/nged-substation-forecast/issues/138) has shipped in
> full, including the NWP ingestion checks
> [#161](https://github.com/openclimatefix/nged-substation-forecast/issues/161) — as-built in
> [Known ECMWF ENS data-quality issues](../architecture/ecmwf-ens-known-issues.md#an-incomplete-run-tolerated-and-reported)
> — and the Hydra removal
> [#228](https://github.com/openclimatefix/nged-substation-forecast/issues/228), as-built in
> [Model configuration](../ml_experimentation/model-configuration.md). The one section still on
> this page, the scientific-rigor tests
> [#229](https://github.com/openclimatefix/nged-substation-forecast/issues/229), now belongs to
> the v0.3 epic
> [#6](https://github.com/openclimatefix/nged-substation-forecast/issues/6).
> Task ordering lives in the GitHub Project board.

## Scientific-rigor tests and cleanup

Issue: [#229](https://github.com/openclimatefix/nged-substation-forecast/issues/229)

*Runs after the [live service and monitoring](live-service.md) land.* The feature-level
no-lookahead tests, cross-mode equivalence test, idempotency tests, and the full-stack
cross-process MLflow test all exist. Three "not cheating" guardrail tests from the original
testing strategy remain unwritten, plus general cleanup.

### Implementation details — rigor tests (deleted when they ship)

**Part 1 — scientific-rigor tests:**

- **CV-windowing no-lookahead** (complements the feature-level tests, which cover lag leakage
  but not window construction): assert no *training* row has `valid_time >= val_start` for its
  fold — i.e. the training window built by `training_window(fold)` and applied in
  `trained_cv_model` never bleeds into validation.
- **Leaderboard fairness**: two different experiments over the same fold are scored on the
  **identical** `(time_series_id, fold)` population — a regression guard on the
  experiment-independence of `eligible_time_series`.
- **Determinism**: training a fold twice with a fixed `random_seed` yields identical
  predictions. This underpins idempotent retries and a stable leaderboard.
- **Degradation smoke-tests**: ablate whole input groups — NWP absent, telemetry absent, a single
  weather variable nulled — and assert that a forecast is still produced for every time series, that
  every value stays inside physical bounds, and that nothing explodes. These consume the scenario
  vocabulary defined by the failure-scenario suite in
  [Metrics & Leaderboard](metrics-and-leaderboard.md#scoring-under-failure-scenarios), so they must
  land alongside it rather than against an ad-hoc vocabulary of their own. They are cheap and
  CI-fast — pure functions over an `AllFeatures` frame, no MLflow — and they check *survival*, not
  skill; skill under degradation is the leaderboard's job. The principle they enforce is
  [Inherent Stability](../design-philosophy/inherent-stability.md).

**Part 2 — cleanup:**

- Remove any remaining dead code/imports from the phased build-out.
- **Split `defs/cv_assets.py`** (898 lines — the complexity hotspot flagged in the 2026-07
  codebase review) into `cv_assets.py` / `production_assets.py` / `metric_assets.py`. The
  [`live_forecasts` work](live-service.md#the-live_forecasts-asset) already starts
  `production_assets.py`; move the `metrics` asset and its helpers into `metric_assets.py`
  here. Pure logic stays in `ml_core._cv_helpers`.

**Part 3 — docs freshness pass.** The permanent-docs migration from the old `dagster_plan.md`
is already done (July 2026): `docs/architecture/ml-orchestration.md` and
`docs/ml_experimentation/evaluating-new-data-sources.md` capture its important ideas. What
remains:

- Check `docs/` against the code after the live service and monitoring land — in particular
  extend `docs/ml_experimentation/dagster-workflow.md` with the live-forecast and monitoring
  flows, and update the "Known limitation" and MLflow-logging notes in
  `docs/architecture/ml-orchestration.md` if the implementations diverged from the plans.
- Run the ship-time triage (per the `github-issue-pr-workflow` skill) on any roadmap content
  the live-service work implemented — e.g. flip the relevant 🚧 statuses in
  `docs/roadmap/metrics-and-leaderboard.md` once monitoring lands.

**Verification.** Full `uv run pytest` green from the repo root, **including the full-stack
cross-process integration test**; `uv run pymarkdown scan` (per CLAUDE.md) green on the
touched docs; `grep -ri 'dagster_plan' docs/ src/ packages/` returns nothing.
