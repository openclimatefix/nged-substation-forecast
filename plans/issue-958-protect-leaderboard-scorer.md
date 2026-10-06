# Plan: protect the leaderboard scorer for autonomous research (#958)

**Problem.** Anything that scores a forecast today can also edit the scoring code, drop the rows it
is bad at, or score on actuals it should not have seen. An autonomous research session would hit all
three. The issue also bundles two kinds of work: code changes in this repository, and sysadmin steps
(Unix user, ACLs, a `sudo` rule) that only the maintainer can run.

**Solution.** The `metrics` asset becomes the only source of a leaderboard number and protects
itself. It refuses a forecast that omits rows of an eligible series, and refuses a window reaching
past `FINAL_TEST_START` unless `NGED_FINAL_TEST=1` is set. A new `scripts/score_study.py` scores a
study's predictions file by running the asset from `main`. A shared study power reader stops at the
same date. An `import-linter` contract keeps the scorer free of `dagster`, `mlflow`, and `studies`.
The sysadmin steps move into a maintainer-run runbook and a separate issue.

## Verdict, size and departures

**Verdict: worth implementing, with three departures from the issue body.** Premises checked on
`main`: `metrics` scores whatever rows it is given (`_score_forecast_group` joins to actuals and
skips series with no overlap); `_resolve_eval_window` takes dates from the fold config; no
`FINAL_TEST_START` exists; `packages/studies` has no shared power reader (`pv_dataset.py`,
`wind_product_frames.py`, and several `studies/*` scripts call `pl.scan_delta` directly); no
`import-linter` anywhere in `pyproject.toml`, CI, or `uv.lock`. PR #1028 has merged, so
`packages/studies` is editable. #960 is open and has not decided the final-test design.

**Size: complex.** One line per trigger:

- **What gets stored:** fires. A new `final_test_start` field on `CvConfig`, a new `study/`
  `experiment_name` prefix in `power_forecasts` and `forecast_metrics`, and a new predictions-file
  route into `power_forecasts`.
- **Production serving path:** does not fire. Only R&D assets and scripts change; `live_forecast_assets.py`
  and the serving path are untouched.
- **Degradation rule:** does not fire. These are R&D assets, which fail fast; no production check is
  added or edited.
- **More than one defensible design:** fires. The row-set definition, which scopes the date guard
  covers, and the in-window power question below each admit several designs.
- **Callers not nameable without searching:** fires for the `pv_dataset` / `wind_product_frames`
  reader migration and the `studies/*` direct readers.

That buys the plan, both plan reviews, and both diff reviews.

**Departures from the issue body** (items 4 to 6 added after the simplicity review):

1. **Do not add a second import-linter contract for "no package outside `packages/studies` may
   import `studies`".** `packages/studies/tests/test_study_boundaries.py` already enforces it. Only
   the scorer-isolation contract is new.
2. **Skip the optional `.claude/settings.json` deny rule and the CODEOWNERS file.** The issue itself
   says a deny rule is bypassed by Bash, and #1035's submit command restores protected paths from
   the session's starting commit. Adding `features/_lags.py` to a deny list is still unagreed
   (comment of 2026-10-05). Both stay out; each is a one-file follow-up if wanted.
3. **Split the sysadmin steps into their own issue** (see "Splitting").
4. **Add a check on the scored rows' `valid_time`.** The issue's guard tests the configured window, which in leaderboard scope is always `val_end` and so can never fire.
5. **Migrate only the two `packages/studies` readers**, and list the other direct readers in the PR body instead of editing published study scripts.
6. **Move the runbook page and the layer-1 doc alignment into the follow-up issue**, because both depend on Q1.

## Decisions for the maintainer

**Q1: in-window power (the issue's last comment). Option (a) or (b)?** My recommendation is (b).

- **(a) Staging harness.** A maintainer-user harness stages, per initialisation time, only the power
  observed before it, and runs the worker's code as the research user against the staged copy.
  Buys: the research user never holds a future actual. Costs: a harness that copies or views the
  power table per init time (about 17,500 hourly steps over the fold), and the worker's code must
  write nothing the session can read back. By the last initialisation time the staged copies cover
  almost the whole window, so the secrecy is thin where it matters. This is the largest piece of new
  infrastructure in the whole auto-research design.
- **(b) Accept in-window visibility, truncate at `FINAL_TEST_START`.** The research user reads all
  power before `FINAL_TEST_START` and none after. Layer 1's wording becomes "cannot read power after
  `FINAL_TEST_START`" instead of "cannot read validation actuals". What stops a worker using
  in-window future power is the leakage test in the submit command (#1035), which perturbs power
  after each cut-off and rejects any node whose earlier forecasts change. That test already
  enforces the property (a) tries to enforce physically, and it also catches a worker that hard-codes
  validation actuals. The remaining cost is that a worker can fit to validation actuals it can see,
  which is the selection-bias risk #960 and the Ladder guard address, not this issue.
- **Consequence of (b) for this issue:** the research user needs a copy of the power table
  truncated at `FINAL_TEST_START`, because Delta files cannot be hidden row by row. Writing that
  copy is an `scripts/` job the maintainer runs (it belongs to the sysadmin issue). Layer 1
  in the issue, the auto-research page, and #1035 then say the same thing.

**Q2: which windows does the `FINAL_TEST_START` guard cover?** Recommendation: every group except
`fold_id == "live"`. Without the exception, scoring live output past the cutoff would need
`NGED_FINAL_TEST=1`, which breaks the ad-hoc production-monitoring route; live rows are forecasts
of the future, not a held-out set. The guard applies to both `leaderboard` and `ad_hoc` scopes,
because `ad_hoc` is the route for deliberate final-test scoring and must still demand the variable.

**Q3: the cutoff date.** Recommendation: `2026-07-01`, the day after the only leaderboard fold's
`val_end` (`2026-06-30`). `CvConfig` validates that `final_test_start` is later than every
leaderboard fold's `val_end`. **#960 has not decided the final-test design, so the field is a
guard, not a sealed test year:** the data after the cutoff is about three months, not an
independent year, and nothing this issue writes (docs, field docstring, error message) calls it a
"final-test year" or claims independence. If #960 moves the date, only `conf/cv/default.yaml`
changes.

**Q4: what is the expected row set?** (Revised after the simplicity review: the cross-series regularity half is dropped, and the reference-experiment alternative is offered as option B.) The issue says rows `(time_series_id, power_fcst_init_time,
valid_time)` "the fold's eligible series should cover", but nothing defines the expected
initialisation times independent of the forecast itself. Recommendation (option A), a check per
`(experiment_name, fold_id)` group in leaderboard scope that raises `MissingForecastRowsError`:

- **Series coverage:** every series in the fold's `eligible_time_series` partition appears in the
  group, and each has a forecast row for every `valid_time` in the fold window that has an
  observed actual for that series. Dropping a hard series, or hard timestamps of one series, fails.
- **Rows inside the window, and nothing else.** Every row's `valid_time` lies in the fold's
  `[val_start, val_end]`. `_resolve_eval_window` returns those dates, but only to stamp them on the
  metric rows; neither `_score_forecast_group` nor `compute_metrics` filters the scored rows to them,
  so without this check a predictions file carrying rows past `FINAL_TEST_START` would be scored and
  labelled as the fold's window. The date guard below is therefore meaningful in leaderboard scope
  only through this check.

Gap option A leaves: a study that drops hard initialisation times for every series, while still
covering every valid time through other initialisation times, passes, because nothing defines a
canonical initialisation-time grid.

**Option B (offered, not recommended as the default):** require the study's distinct
`(time_series_id, power_fcst_init_time, valid_time)` keys to equal a named reference experiment's
(the XGBoost baseline) for the same fold, using an anti-join each way. It closes the gap above,
needs no read of `eligible_time_series`, and refuses out-of-window rows as extras. It costs a
dependency on the baseline having been materialised for the fold, and changes the scored
population from the eligible series to the baseline's trained series. Option A follows the issue's
wording ("the fold's eligible series"); choose B if you prefer the stronger check.

Under option A, an experiment whose `trained_time_series_ids` is smaller than the eligible set now
fails the check. That is intended; the existing `xgboost` fold must still pass.

## What changes, file by file

**`conf/cv/default.yaml`, `packages/contracts/src/contracts/config_schemas.py`.** Add
`final_test_start: date` to `CvConfig` (required, `2026-07-01`) with a model validator that it
exceeds every leaderboard fold's `val_end`. Docstring says "guard", not "test year".

**`src/nged_substation_forecast/defs/cv_assets.py`** (only `metrics`, `MetricsConfig`,
`_resolve_eval_window`, `_score_forecast_group`, and new helpers; the out-of-bounds assets
`eligible_time_series`, `effective_capacity`, `trained_cv_model`, `cv_power_forecasts` are not
touched). PR #1040 (open, #1019) edits other parts of this file; expect a rebase, nothing else.

- `_resolve_eval_window` returns the window as today. A new `_require_window_within_guard(window_end,
  fold_id, final_test_start)` raises when `window_end >= final_test_start` and
  `NGED_FINAL_TEST != "1"` and `fold_id != "live"`; in ad-hoc scope `window_end` is the observed
  maximum `valid_time`, so this is the check that stops an ad-hoc run reaching past the cutoff. In
  leaderboard scope the window is the fold's `val_end`, so the leaderboard check is the row-window
  check below. It is
  called in `_score_forecast_group` before any scoring or write, so a refused window leaves no
  `forecast_metrics` rows or MLflow runs. R&D, so it raises.
- New `_require_complete_row_set(group_scan, eligible_ids, actuals_lf, window)` per Q4 option A
  (series coverage plus rows inside `[val_start, val_end]`), called in
  `_score_forecast_group` for `leaderboard` scope only, before the batched scoring. It streams
  distinct `(time_series_id, valid_time)` and per-series pair counts rather than collecting rows.
  `ad_hoc` scope is exempt because it has no eligible population. Failures use `ValueError`
subclasses defined next to `NoOverlappingActualsError`, whichever module holds that.
- `metrics` reads the fold's `eligible_time_series` partition (read-only; the table is written by
  an out-of-bounds asset, which this issue does not change) and passes the ids down.
- Rows with `experiment_name` starting `study/` are written to `forecast_metrics` and logged under
  their prefixed name; nothing else changes. `promotion_assets.py` reads `experiment_name` only from
  MLflow runs that have a registered model, which a `study/` group never has, so no code is needed;
  the PR body records this.

**`scripts/score_study.py`** (new). Arguments: predictions parquet path, study name, fold id. It
validates the file as `PowerForecast`, sets `experiment_name = "study/<name>"` (rejecting a name
that already begins `study/` or contains a path separator), writes the rows to `power_forecasts`
via `delta_store.power_forecasts.write_power_forecasts(replace_partition=...)`, then calls
`dagster.materialize([metrics], run_config=...)` in-process with `evaluation_scope="leaderboard"`
and a `PopulationFilter` pinned to that experiment and fold, following
`scripts/forecasting/run_baseline_experiment.py`. It takes no code and no actuals as input. Why it
is safe from `main`: the sysadmin step (separate issue) runs it as the maintainer's user from the
`main` checkout.

**`packages/studies/src/studies/power.py`** (extend) adds `scan_power()`, returning the power
Delta as a lazy frame filtered to `time < FINAL_TEST_START` (read from the CV config). Migrate the
two direct readers inside `packages/studies`: `pv_dataset.py` and `wind_product_frames.py`. The
issue's audit is a list in the PR body of the other direct `scan_delta` readers on the power table
under `studies/` (`weather_downloads/fetch_open_meteo_previous_runs.py`,
`past_weather/cerra_wind_levels.py`, `beam_diffuse_split/stamp_alignment.py`,
`beam_diffuse_split/site_e_commissioning.py`; rechecked at implementation). Those published scripts
are left alone, because migrating them would silently cut their input at the cutoff and could
change reproduced results or a download range. No bypass-scan test is added. The reader guards only
callers that route through it, not the data; against an autonomous session the protection is the
truncated copy under Q1(b).

**`pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`.** Add
`import-linter` to the dev group and a `[tool.importlinter]` block with one `forbidden` contract:
`ml_core.metrics` and `ml_core.cv_helpers` may not import `dagster`, `mlflow`, or `studies`
(`include_external_packages = true`, indirect imports included). Add a `uv run lint-imports` step to
CI after `ty`. If the strict indirect check trips on `contracts`, the contract is narrowed to direct
imports and the reason recorded in the TOML. `uv.lock` is also edited by #147: whichever lands
second re-runs `uv lock`.

**Sysadmin steps (not in this PR).** The follow-up issue carries a runbook page the maintainer runs
by hand: the Unix user, `setfacl` on the data folders, setgid directories with
umask 002, the research user's own `uv` cache and credentials, tightening `/mnt/data` (mode 2777
today), the narrow `sudo` rule naming `scripts/score_study.py`, and the truncated power copy (Q1b).

## Splitting

**Recommendation: split the sysadmin steps into a new issue, and keep one PR for the code.** The
code half is testable and reviewable in the diff; the sysadmin half is a checklist only the
maintainer can run and verify (`getfacl`, a failed `cat` as the research user), and its closing
condition is a human confirming it works. Bundling them would hold #958 open until the maintainer
has done sysadmin work, or close it with the layer-1 claim unverified. The new issue is blocked by
this one (the `sudo` rule names the script) and by the Q1 decision. This PR updates the issue's
checklist to point at it.

## Design-philosophy check

All of this runs in R&D, where `inherent-stability.md` says to fail fast: the guard, the row-set
check, and `score_study.py` raise, and none adds an asset check (so the WARN/non-blocking rule is not
engaged). Nothing touches `live_forecast_assets.py` or the serving path; the `live` exemption keeps
production monitoring unaffected. Hypothesis H2 is reworded in the docs (confirmed promotions per
month, not experiments per month). The design principle traded away is principle 3's "research runs
on the production pipeline" for autonomous studies, bought back by "one scoring path for all
research"; the doc edits state this.

## Tests

Each assertion fails on `main` today.

- `tests/test_metrics.py` (or `test_cv_assets.py`, wherever `metrics` is already exercised):
  a leaderboard-scope group missing one eligible series raises `MissingForecastRowsError`; on `main`
  it scores what it has. A group missing a subset of one series' valid times raises. A complete group still scores.
- An ad-hoc group whose observed `valid_time` maximum is at or after `final_test_start` raises
  without `NGED_FINAL_TEST=1` and scores with it (`monkeypatch.setenv`). The `fold_id == "live"` ad-hoc group scores with the variable unset.
  A refused window writes no `forecast_metrics` rows.
- `packages/contracts/tests`: `CvConfig` rejects a `final_test_start` that is not after every
  leaderboard `val_end`.
- `packages/studies/tests`: `scan_power()` returns no row at or after the cutoff on a small Delta
  table written in `tmp_path` (the cutoff injected).
- A leaderboard-scope group with a row whose `valid_time` is past `val_end` raises; on `main` it
  scores. This is the test that proves the guard is not vacuous.
- `tests/test_score_study.py`: with a moto-free local Delta, `score_study` rejects a predictions
  file with a missing eligible series, writes `study/<name>` rows into `forecast_metrics` for a
  complete file, and refuses a name beginning `study/`.
- Import linter: CI runs the real contract. No test of `import-linter` itself is written; the
  mutation-testing review adds `import mlflow` to `ml_core.metrics` once and confirms
  `lint-imports` fails.
- The existing `xgboost` leaderboard fold path (`tests/test_cv_assets.py`) keeps passing.

## Docs to update

Written in the present tense, no history:

- `docs/design-philosophy/design-principles.md` principle 3; `studies/README.md`: reviewed research
  keeps principle 3; autonomous studies get one scoring path, and a finding reaches production only
  as a written spec plus reviewed re-implementation.
- `docs/design-philosophy/engineering-hypotheses.md` H2 wording (labels never renumbered).
- `docs/roadmap/metrics-and-leaderboard.md`: the `study/` prefix, the guard and its cutoff, the
  row-set refusal; delete the shipped steps of "Implementation details — final-test window" and keep
  step 3 (#960's conditional reservation).
- `docs/ml_experimentation/index.md`: the autonomous-study route and promotion path.
- `.claude/skills/study/SKILL.md` and the reviewer checklist: a study's leaderboard number comes only
  from `scripts/score_study.py`, and every number on a study page traces to a `forecast_metrics`
  row. CLAUDE.md's skills table is checked for a stale summary.
- Aligning layer 1 in `docs/roadmap/auto-research.md` and #1035 with the Q1 outcome moves to the
  follow-up issue, because it depends on Q1.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
uv run lint-imports
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

Plus the pydoclint and docs-link checks from CI, run locally, and `ls plans` empty before ship.

## Risks and open questions

- Q1-Q4 above need the maintainer's decision; Q1 is the one that changes scope.
- The `import-linter` strict indirect check may fail through `contracts`; fallback stated above.
- The row-set check reads `eligible_time_series`, which #1019 may change (cleaned power). It reads
  only the Delta table's `(fold_id, time_series_id)` columns, which `EligibleTimeSeries` fixes.
- A stale `eligible_time_series` partition (not re-materialised after a roster change) makes the
  check refuse correct forecasts. This is the intended fail-fast, and the error names the fold.

## Review log

**Simplicity review (Opus), accepted:** the vacuous leaderboard guard (departure 4); dropping the
cross-series regularity check; shrinking the study reader migration; moving the runbook and
layer-1 docs to the follow-up issue; dropping the import-linter self-test and the bypass-scan
test; recording that `promotion_assets.py` cannot see `study/` groups.

**Simplicity review, rejected:**

- **Drop the `NGED_FINAL_TEST` variable and the `live` exemption in favour of a row-window check
  alone.** The issue specifies the variable and its test, and ad-hoc scope (the maintainer's route)
  has no window except the observed rows. Kept for ad-hoc scope; the row-window check is added for
  leaderboard scope.
- **Replace Q4 with an equality check against a reference experiment.** Offered as option B in Q4
  for the maintainer, because it changes the scored population and depends on the baseline.
- **Replace `import-linter` with a `sys.modules` pytest.** The issue names import-linter and a CI
  contract; the pytest cannot see imports inside a function the way a static check can. Kept.
- **Defer the `import-linter` work entirely.** Not proposed; noted only because of the `uv.lock`
  collision with #147, which the plan already handles.
