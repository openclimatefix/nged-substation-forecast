# Plan: protect the leaderboard scorer for autonomous research (#958)

**Problem.** Anything that scores a forecast today can also edit the scoring code, drop the rows it
is bad at, or score on actuals it should not have seen. An autonomous research session would hit all
three. The issue also bundles two kinds of work: code changes in this repository, and sysadmin steps
(Unix user, ACLs, a `sudo` rule) that only the maintainer can run.

**Solution.** The `metrics` asset becomes the only source of a leaderboard number and protects
itself. For study forecasts it refuses any row-key set that differs from a reference experiment's, and it refuses a window reaching
past `FINAL_TEST_START` unless `NGED_FINAL_TEST=1` is set. A new `scripts/forecasting/score_study.py` scores a
study's predictions file by running the asset from `main`. The shared study power reader (issue #1082) stops at the
same date. An `import-linter` contract keeps the scorer free of `dagster`, `mlflow`, and `studies`.
The sysadmin steps move into a maintainer-run runbook and a separate issue.

## Verdict, size and departures

**Verdict: worth implementing, with three departures from the issue body.** Premises checked on
`main`: `metrics` scores whatever rows it is given (`_score_forecast_group` joins to actuals and
skips series with no overlap); `_resolve_eval_window` takes dates from the fold config; no
`FINAL_TEST_START` exists; `packages/studies` has no shared power reader (`pv_dataset.py`,
`wind_product_frames.py`, and several `studies/*` scripts call `pl.scan_delta` directly), which
[issue #1082 (Make every study read cleaned power through one reader)](https://github.com/openclimatefix/nged-substation-forecast/issues/1082) adds first; no
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
- **Callers not nameable without searching:** fires. The required `final_test_start` field touches six `CvConfig(...)` constructions in tests, and the `study/` prefix reaches the promotion path in `ml_core/mlflow_runs.py`.

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
5. **Split the study-reader migration into [issue #1082 (Make every study read cleaned power through one reader)](https://github.com/openclimatefix/nged-substation-forecast/issues/1082)**, a separate PR that reads cleaned power with no cutoff. This plan keeps only the cutoff.
6. **Move the runbook page and the layer-1 doc alignment into the follow-up issue**, because both depend on Q1.

## Decisions for the maintainer

**Q1: in-window power (the issue's last comment). Decided by the maintainer: option (b).** Options (a) and (c) are kept below as the alternatives considered. Under (b) the truncated copy is built by the follow-up sysadmin issue.

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
- **(c) One Delta table, partitioned at the cutoff.** The power tables gain a derived partition
  column, such as `after_cutoff` (true or false), so the rows after `FINAL_TEST_START` sit in their
  own directory, and an ACL denies the research user that directory. Code running as the maintainer
  sees the one normal table. Delta has no setting that splits files at a date, so the partition
  column is the only mechanism. What happens to the manifest: `_delta_log/` stays one shared log
  that the research user must read, and it lists every data file, including the denied ones, with
  row counts and per-column minimum and maximum values. That leaks the post-cutoff row counts and
  the range of `power` per file; setting the table's statistics columns to skip `power` removes
  most of the range leak. A scan that filters `after_cutoff = false` prunes the denied directory
  and works; an unfiltered scan fails with a permission error, which fails closed but is opaque, so
  `scan_power()` would apply the filter. Costs: partitioning an existing table is a one-off rewrite,
  moving `FINAL_TEST_START` rewrites it again (and #960 has not settled the date), the writer in
  `defs/assets.py` (issue #1020's territory) and the cleaning asset must derive the column on every
  write, and the cleaned table needs the same layout because studies now read it. Buys: no copy to
  refresh and no second table. Not chosen now because it edits out-of-bounds files and is
  expensive to redo when the date moves; it is the better end state if the cutoff proves stable.
- **Consequence of (b) for this issue:** the research user needs a copy of the power table
  truncated at `FINAL_TEST_START`, because Delta files cannot be hidden row by row. Writing that
  copy is a `scripts/maintenance/` job the maintainer runs (it belongs to the sysadmin issue). Layer 1
  in the issue, the auto-research page, and #1035 then say the same thing.

**Q2: which windows does the `FINAL_TEST_START` guard cover? Decided: as recommended.** every group except
`fold_id == "live"`. Without the exception, scoring live output past the cutoff would need
`NGED_FINAL_TEST=1`, which breaks the ad-hoc production-monitoring route; live rows are forecasts
of the future, not a held-out set. The guard applies to both `leaderboard` and `ad_hoc` scopes,
because `ad_hoc` is the route for deliberate final-test scoring and must still demand the variable.

**Q3: the cutoff date. Decided: as recommended.** `2026-07-01`, the day after the only leaderboard fold's
`val_end` (`2026-06-30`). `CvConfig` validates that `final_test_start` is later than every
leaderboard fold's `val_end`. **#960 has not decided the final-test design, so the field is a
guard, not a sealed test year:** the data after the cutoff is about three months, not an
independent year, and nothing this issue writes (docs, field docstring, error message) calls it a
"final-test year" or claims independence. If #960 moves the date, only `conf/cv/default.yaml`
changes.

**Q4: what is the expected row set?** (Revised after the simplicity review: the cross-series regularity half is dropped, and the reference-experiment alternative is offered as option B.) The issue says rows `(time_series_id, power_fcst_init_time,
valid_time)` "the fold's eligible series should cover", but nothing defines the expected
initialisation times independent of the forecast itself. Option A (rejected by the maintainer after the correctness review, findings 1 and 3), a check per
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
population from the eligible series to the baseline's trained series. **Decided by the maintainer: option B.** Option A would refuse the existing
XGBoost fold, because `cv_power_forecasts` builds each run's half-hourly grid only between that
run's first and last native NWP step, so the last valid times of `val_end` (21:30 to 23:30 with
3-hour steps) have actuals but no forecast row for any series. Option A also admits extra
non-eligible series and short-lead-only submissions, which the "all" horizon slice pools. Option B
inherits the baseline's own coverage, so both disappear. Under B the check is: the study's keys
equal the reference experiment's keys (anti-join each way, ensemble members included), and the
study contains no series outside the reference's. If you choose A instead, add "no series outside
the eligible set", define "observed actual" from the scorer's own `actuals_lf` (never a separate
power scan), restrict the valid times required to those inside some run's native range, and
accept the lead-time gap.

**What option B means in code (decided).** The check applies to groups whose `experiment_name` starts with `study/`; reviewed experiments are not checked, because their code is reviewed. A new `reference_experiment_name` field on `CvConfig` names the reference (the XGBoost baseline; the implementer reads its exact name from `scripts/forecasting/run_baseline_experiment.py`). For each series batch the check anti-joins the study's keys `(time_series_id, power_fcst_init_time, valid_time, ensemble_member)` against the reference experiment's keys for the same fold, in both directions, and also compares the two series sets up front. Batching by series, as scoring already does, bounds memory; one fold is about 364M rows. A missing reference partition raises with a message naming the experiment to materialise.

## What changes, file by file

**`conf/cv/default.yaml`, `packages/contracts/src/contracts/config_schemas.py`.** Add
`final_test_start: date` to `CvConfig` (required, `2026-07-01`) with a model validator that it
exceeds every leaderboard fold's `val_end`. Docstring says "guard", not "test year".

**`src/nged_substation_forecast/defs/cv_assets.py`** (only `metrics`, `MetricsConfig`,
`_resolve_eval_window`, `_score_forecast_group`, and new helpers; the out-of-bounds assets
`eligible_time_series`, `effective_capacity`, `trained_cv_model`, `cv_power_forecasts` are not
touched). PR #1040 (#1019) has merged: `metrics` now reads actuals through `scan_cleaned_power` and depends on `clean_nged_power_data`. The row-set check uses that same `actuals_lf`.

- `_resolve_eval_window` returns the window as today. A new `_require_window_within_guard(window_end,
  fold_id, final_test_start)` raises when `window_end >= final_test_start` and
  `NGED_FINAL_TEST != "1"` and `fold_id != "live"`; in ad-hoc scope `window_end` is the observed
  maximum `valid_time`, so this is the check that stops an ad-hoc run reaching past the cutoff. In
  leaderboard scope the window is the fold's `val_end`, so the leaderboard check is the row-window
  check below. It is
  called in `_score_forecast_group` before any scoring or write, so a refused window leaves no
  `forecast_metrics` rows or MLflow runs. R&D, so it raises.
- New pure function `require_same_row_keys` in `ml_core/metrics.py` (option B above), called from
  `_score_forecast_group` for `study/` groups before the batched scoring, and called by
  `score_study.py` before it writes, so a rejected file leaves no partition behind. Plus a
  leaderboard-scope check, for every group, that each row's `valid_time` lies in
  `[val_start, val_end]`. `ad_hoc` scope is exempt because it has no fold. Failures use `ValueError`
  subclasses defined next to `NoOverlappingActualsError`, whichever module holds that.
- Rows with `experiment_name` starting `study/` are written to `forecast_metrics` and logged under
  their prefixed name; nothing else changes. `promotion_assets.py` reads `experiment_name` only from
  MLflow runs that have a registered model, which a `study/` group never has, so no code is needed;
  the PR body records this.

**`scripts/forecasting/score_study.py`** (new). Arguments: predictions parquet path, study name, fold id. It
validates the file as `PowerForecast`, sets `experiment_name = "study/<name>"` (rejecting a name
that already begins `study/` or contains a path separator), writes the rows to `power_forecasts`
via `delta_store.power_forecasts.write_power_forecasts(replace_partition=...)`, then calls
`dagster.materialize([metrics], run_config=...)` in-process with `evaluation_scope="leaderboard"`
and a `PopulationFilter` pinned to that experiment and fold, following
`scripts/forecasting/run_baseline_experiment.py`. It takes no code and no actuals as input. Why it
is safe from `main`: the sysadmin step (separate issue) runs it as the maintainer's user from the
`main` checkout.

**`packages/studies/src/studies/power.py`** (after issue #1082 merges) adds the cutoff to `scan_power()`: one filter, `time < FINAL_TEST_START`, with the date read from the CV config. If this plan is implemented first, `scan_power()` does not exist yet and #1082 comes first. Q5 is the only open question: whether the reader truncates at all.

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
today), the narrow `sudo` rule naming `scripts/forecasting/score_study.py`, and the truncated power copy (Q1b).

## Changes from the correctness review

Accepted (each verified or marked for verification at implementation):

- **`score_study.py` must not trust its inputs.** `delta_store/power_forecasts.py` and
  `forecast_metrics.py` build the overwrite predicate by f-string, so a study name containing a
  quote could overwrite other experiments' rows. The script whitelists the study name with
  `^[a-z0-9_-]{1,64}$`, requires the fold id to be in `leaderboard_fold_ids` (so `live` and
  `smoke_test` are refused), checks that the file's `fold_id` column agrees, refuses to overwrite
  an existing `study/<name>` partition unless `--replace` is given (so submissions are recorded),
  and refuses to run if `CV_CONFIG_PATH`, `NGED_FINAL_TEST`, any `*_DATA_PATH`, or
  `MLFLOW_TRACKING_URI` arrives in the caller's environment. The f-string predicates themselves are
  an out-of-scope defect, reported to the maintainer rather than fixed here.
- **Check every group before scoring any.** `metrics` runs the date guard and the row-set check
  for all groups first, so a refusal never leaves earlier groups' rows or MLflow runs behind. The
  test uses two groups.
- **Put the pure checks in `ml_core/metrics.py`** (`require_complete_row_set`,
  `require_window_within_guard`), where the import-linter contract, #1035's protected paths, and
  Dagster-free tests all apply. `cv_assets.py` only calls them.
- **No new asset dependency.** Option B reads only `power_forecasts`, so `metrics`' `deps` and its
  read of `eligible_time_series` are unchanged.
- **Promotion path.** `ml_core/mlflow_runs.py:list_promotable_runs` lists every fold run in every
  MLflow experiment, so a `study/` fold run would reach the `promotable_model_runs` candidates. The
  plan filters `study/` experiments there and refuses them in `promoted_model`, with tests. This
  replaces the earlier claim that no code is needed.
- **Study reader.** Moved to issue #1082. The cutoff in `scan_power()` changes the input of every later re-run of a published study, which is Q5.
- **Existing constructions.** A required `final_test_start` breaks `CvConfig(...)` in
  `tests/test_jobs.py` and five places in `packages/contracts/tests/test_config_schemas.py`; the plan
  updates them, and `_score_forecast_group`'s positional callers in `tests/test_metrics.py`.
- **Test fixtures.** A fixture whose actuals span the whole `val_end` day, with forecasts stopping at
  21:00 as production's do, is added; it fails an option-A check on `main`'s data shape. The MLflow
  file-store fixtures in `tests/test_metrics.py` move to `conftest.py` so the `score_study` test
  can use them. The ad-hoc guard test writes rows after the cutoff directly or monkeypatches
  `cv_assets._cv_config`. Regression tests that already pass on `main` (a complete group scores, a
  `live` group scores) are labelled as such.
- **Docs added:** `docs/ml_experimentation/dagster-workflow.md` Step 9,
  `docs/ml_experimentation/cross-validation-folds.md` (the new field), and
  `docs/roadmap/auto-research.md` lines about the refusal and promotion paths, which would otherwise
  be false.
- **Risk to verify:** `score_study` commits to `power_forecasts` while `live_forecasts` overwrites a
  different partition hourly; the implementer checks that delta-rs resolves the disjoint-partition
  commits, or the script retries.

**Q5: should `scan_power()` also truncate at the cutoff? Decided: yes, as recommended.** Truncating means a re-run of a published study loses about 3 months of power (2026-07-01 to October), which can change site lists, seeded anonymised labels, and numbers. Recommended yes: a reader that studies can read past the cutoff is no guard. Published pages keep the numbers they were computed with.

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

- `tests/test_metrics.py` (or `test_cv_assets.py`, wherever `metrics` is already exercised): a
  `study/` group missing one series of the reference experiment raises `MissingForecastRowsError`;
  a group missing a subset of one series' keys raises; a group with an extra series or extra keys
  raises; a group identical to the reference scores. On `main` the first three score what they are
  given. A fixture whose reference forecasts stop at 21:00 on the last day shows that the check
  does not demand rows the reference lacks.
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
  file with a missing reference series, leaving no partition, writes `study/<name>` rows into `forecast_metrics` for a
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
- `scripts/README.md`: a line for `forecasting/score_study.py` under the existing `forecasting/` section, saying what it is for and who runs it.
- `.claude/skills/study/SKILL.md` and the reviewer checklist: a study's leaderboard number comes only
  from `scripts/forecasting/score_study.py`, and every number on a study page traces to a `forecast_metrics`
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

Plus `uv run pytest --run-studies -n auto packages/studies` (a plain `pytest` skips those tests; main now gates them behind that flag and a `studies_tests.yml` workflow), plus the pydoclint and docs-link checks from CI, run locally, and `ls plans` empty before ship.

## Risks and open questions

- Q1 to Q5 are decided (see above). No decision is open.
- The `import-linter` strict indirect check may fail through `contracts`; fallback stated above.
- The row-set check depends on the reference experiment having been materialised for the fold. A missing reference partition raises with a message naming the experiment. The reference experiment's own forecasts are trusted: a reviewed experiment, produced by the maintainer.
- Concurrent Delta commits: `score_study.py` writes `study/` partitions of `power_forecasts` while `live_forecasts` overwrites the `live` partition hourly. The implementer verifies that delta-rs resolves the disjoint-partition commits, or the script retries.

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

**Correctness review (Opus), rejected or adjusted:** none rejected outright. Finding 14 (concurrent
Delta commits) is kept as a verification item rather than a design change. Finding 9 is adopted
partly: only the two pure checks move to `ml_core/metrics.py`, and `cv_assets.py` keeps the
orchestration.

## Main merged after the plan was written (2026-10-07)

Checked against `origin/main`: the plan's design is unchanged. Updated facts: #1040 has merged
(see `cv_assets.py` above); `pv_dataset.py` and `wind_product_frames.py` still read raw power with
`scan_delta`, so the reader decision stands; there is still no `import-linter` in `pyproject.toml`;
`uv.lock` changed heavily, so the lock is re-resolved at implementation; the studies tests now run
behind `--run-studies`; the repository renamed "roster" to "site list" and "metadata table", and
this plan follows.
