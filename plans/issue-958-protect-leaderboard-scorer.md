# Plan: protect the leaderboard scorer for autonomous research (#958)

**Problem.** Anything that scores a forecast today can also edit the scoring code, drop the rows it
is bad at, or score on actuals it should not have seen. An autonomous research session would hit all
three. The issue also bundles two kinds of work: code changes in this repository, and sysadmin steps
(Unix user, ACLs, a `sudo` rule) that only the maintainer can run.

**Solution.** The `metrics` asset becomes the only source of a leaderboard number and protects
itself. A study's forecasts must carry exactly the same row keys as a reference experiment's, so a
study cannot abstain on hard rows. Scoring refuses any window reaching past `FINAL_TEST_START`
unless `NGED_FINAL_TEST=1` is set. A new `scripts/forecasting/score_study.py` scores a study's
predictions file by running the asset from `main`. The shared study power reader (issue #1082) stops
at the same date. An `import-linter` contract keeps the scorer free of `dagster`, `mlflow`, and
`studies`. The sysadmin steps move into a separate maintainer-run issue.

## Verdict, size and departures

**Verdict: worth implementing, with the departures below.** Premises checked on `main`: `metrics`
scores whatever rows it is given; `_resolve_eval_window` takes dates from the fold config and only
stamps them on the output rows; no `FINAL_TEST_START` exists; there is no `import-linter` in
`pyproject.toml`, CI, or `uv.lock`; `packages/studies` has no shared power reader, which
[issue #1082 (Make every study read cleaned power through one reader)](https://github.com/openclimatefix/nged-substation-forecast/issues/1082)
adds first. PR #1028 and PR #1040 have merged. [Issue #960 (Design rolling-origin CV folds, assuming at least monthly retraining)](https://github.com/openclimatefix/nged-substation-forecast/issues/960)
is open and has not decided the final-test design; the issue says the scorer protections are not
blocked on it.

**Size: complex.** One line per trigger:

- **What gets stored:** fires. New `final_test_start` and `reference_experiment_name` fields on
  `CvConfig`, a `study/` `experiment_name` prefix in `power_forecasts` and `forecast_metrics`, and a
  predictions-file route into `power_forecasts`.
- **Production serving path:** does not fire. Only R&D assets and scripts change.
- **Degradation rule:** does not fire. These are R&D assets, which fail fast.
- **More than one defensible design:** fires. The row-set definition, the guard's scope, and
  in-window power each admit several designs.
- **Callers not nameable without searching:** fires. The two required `CvConfig` fields touch six
  `CvConfig(...)` constructions in tests, and the `study/` prefix reaches the promotion path in
  `ml_core/mlflow_runs.py`.

That buys the plan, both plan reviews, and both diff reviews.

**Departures from the issue body:**

1. **No second import-linter contract for "no package outside `packages/studies` may import
   `studies`".** `packages/studies/tests/test_study_boundaries.py` already enforces it.
2. **Skip the optional `.claude/settings.json` deny rule and the CODEOWNERS file.** The issue itself
   says a deny rule is bypassed by Bash, and #1035's submit command restores protected paths from
   the session's starting commit. Adding `features/_lags.py` to a deny list is still unagreed.
3. **Split the sysadmin steps into their own issue** (see "Splitting").
4. **Check the scored rows' `valid_time`.** The issue's guard tests the configured window, which in
   leaderboard scope is always `val_end` and so can never fire.
5. **Split the study-reader migration into issue #1082**, which reads cleaned power with no cutoff.
   This plan keeps only the cutoff.
6. **Move the runbook page and the layer-1 documentation alignment into the follow-up issue**,
   because both depend on the Q1 decision.
7. **Define the expected row set by a reference experiment** (Q4), not by the fold's eligible series.

## Decisions by the maintainer

**Q1: in-window power. Decided: option (b).** The research user reads all power before
`FINAL_TEST_START` and none after. Layer 1's wording becomes "cannot read power after
`FINAL_TEST_START`". What stops a worker using in-window future power is the leakage test in the
submit command (#1035), which perturbs power after each cut-off and rejects any node whose earlier
forecasts change; the same test catches a worker that hard-codes validation actuals. The remaining
cost is that a worker can fit to validation actuals it can see, which is the selection-bias risk
#960 and the Ladder guard address. The research user needs a copy of the power table truncated at
`FINAL_TEST_START`, because Delta files cannot be hidden row by row; the follow-up issue builds it
with a `scripts/maintenance/` script. Alternatives considered:

- **(a) Staging harness.** A maintainer-user harness stages, per initialisation time, only the power
  observed before it, and runs the worker's code as the research user against the staged copy. It
  needs a harness that copies or views the power table per initialisation time (about 17,500 hourly
  steps over the fold), and the worker's code must write nothing the session can read back. By the
  last initialisation time the staged copies cover almost the whole window, so the secrecy is thin
  where it matters.
- **(c) One Delta table, partitioned at the cutoff.** The power tables gain a derived partition
  column, such as `after_cutoff`, so the rows after the cutoff sit in their own directory and an
  ACL denies the research user that directory. Delta has no setting that splits files at a date, so
  a partition column is the only mechanism. `_delta_log/` stays one shared log that the research
  user must read, and it lists every data file, including the denied ones, with row counts and
  per-column minimum and maximum values; skipping statistics on `power` removes most of that leak.
  A scan filtered on `after_cutoff = false` prunes the denied directory; an unfiltered scan fails
  with a permission error. Costs: a one-off rewrite of the table, and another rewrite whenever
  `FINAL_TEST_START` moves; the writer in `defs/assets.py` (issue #1020's territory) and the
  cleaning asset must derive the column on every write; the cleaned table needs the same layout.
  Not chosen now because it edits out-of-bounds files and is expensive to redo when the date moves.

**Q2: which windows the date guard covers. Decided: every group except `fold_id == "live"`.** The
guard applies to both `leaderboard` and `ad_hoc` scopes, because `ad_hoc` is the route for
deliberate final-test scoring and must still demand the variable. Live rows are forecasts of the
future, not a held-out set, and exempting them keeps ad-hoc production monitoring working.

**Q3: the cutoff date. Decided: `2026-07-01`,** the day after the only leaderboard fold's `val_end`
(`2026-06-30`). `CvConfig` validates that `final_test_start` is later than every leaderboard fold's
`val_end`. **Issue #960 has not decided the final-test design, so the field is a guard, not a sealed
test year:** the data after the cutoff is about three months, not an independent year, and nothing
this change writes (documentation, field docstring, error message) calls it a "final-test year" or
claims independence. If #960 moves the date, only `conf/cv/default.yaml` changes. This plan is not
delayed until #960 resolves.

**Q4: the expected row set. Decided: option B, a reference experiment.** The check applies to
groups whose `experiment_name` starts with `study/`; reviewed experiments are not checked, because
their code is reviewed. A new `reference_experiment_name` field on `CvConfig` names the reference
(the XGBoost baseline; its exact name is `xgboost_no_power_lags` in
`scripts/forecasting/run_baseline_experiment.py`, to be re-read at implementation). Why not the
fold's eligible series: the issue's wording would refuse the existing XGBoost fold, because
`cv_power_forecasts` builds each run's half-hourly grid only between that run's first and last
native NWP step, so the last valid times of `val_end` (21:30 to 23:30 with 3-hour steps) have
actuals but no forecast row for any series. Option B inherits the reference's own coverage.

- **The key is `(time_series_id, power_fcst_init_time, valid_time)`, without `ensemble_member`.**
  The reference is a roughly 51-member ensemble, and the scorer treats single-member forecasts as
  first-class, so requiring the same member keys would refuse any deterministic study or any study
  with a different member count. Dropping members drops no hard row. (A refinement of option B made
  after the third correctness review; flagged to the maintainer.)
- **A study must use the reference's initialisation-time grid.** The reference's initialisation
  times are the ECMWF ENS run time plus the publication delay, so a study built on another weather
  product with different initialisation times is refused. Scoring such a study needs a different
  reference, which is a decision for when one arrives.
- **The check is an anti-join in both directions** on the distinct keys, per series batch with the
  same batch size scoring already uses, plus an up-front comparison of the two series sets. Batching
  bounds memory; one fold is about 364M rows. The streaming engine is used throughout.
- **The reference scan is a fresh `pl.scan_delta`** filtered on `experiment_name ==
  reference_experiment_name`, `fold_id`, and the batch's `time_series_id` values, with the same
  `valid_time_min`/`valid_time_max` filter the operator's `PopulationFilter` carries. It cannot come
  from `_group_scan(pruned_scan, ...)`, because `pruned_scan` is already pinned to the study.
- **A missing reference partition raises** with a message naming the experiment to materialise. The
  reference's own forecasts are trusted: they come from a reviewed experiment.
- **Rows outside the fold window.** For every leaderboard-scope group, each row's `valid_time` must
  lie in `[val_start, val_end]`. For `study/` groups the key equality already implies this.
- **Stale study groups.** A study group goes stale when the reference is re-materialised or renamed.
  An unfiltered leaderboard run skips `study/` groups with a warning naming them, so a stale study
  cannot block scoring of reviewed experiments; `score_study.py` pins its own experiment, so the
  check still raises there. Each study metric row's MLflow run is tagged with the reference
  partition's Delta version.

**Q5: whether `scan_power()` truncates at the cutoff. Decided: yes.** A re-run of a published study
then loses about 3 months of power (2026-07-01 to October), which can change site lists, seeded
anonymised labels, and numbers. Published pages keep the numbers they were computed with.

## What changes, file by file

**`conf/cv/default.yaml`, `packages/contracts/src/contracts/config_schemas.py`.** Add
`final_test_start: date` (`2026-07-01`) and `reference_experiment_name: str` to `CvConfig`, both
required. Validators: `final_test_start` exceeds every leaderboard fold's `val_end`;
`reference_experiment_name` does not start with `study/`. Docstrings say "guard", not "test year".
Update the six existing `CvConfig(...)` constructions in `tests/test_jobs.py` and
`packages/contracts/tests/test_config_schemas.py` (lines 15, 48, 71, 113, and 171).

**`packages/ml_core/src/ml_core/metrics.py`** gains the pure checks, so the import-linter contract
and #1035's protected paths cover them and they test without Dagster: `require_window_within_guard`
and `require_same_row_keys`, raising `FinalTestWindowError` and `RowKeyMismatchError` (both
`ValueError` subclasses next to `NoOverlappingActualsError`).

**`src/nged_substation_forecast/defs/cv_assets.py`** (only `metrics`, `MetricsConfig`,
`_resolve_eval_window`, `_score_forecast_group`, and new helpers; `eligible_time_series`,
`effective_capacity`, `trained_cv_model`, and `cv_power_forecasts` are not touched).

- **Validate every group before scoring any.** `metrics` first resolves each group's window and
  runs the date guard and the key check for all groups, then scores. A refusal on group N therefore
  never leaves groups 1 to N-1's rows or MLflow runs behind. This moves `_resolve_eval_window` ahead
  of scoring, where today it runs after, and changes `_score_forecast_group`'s signature (its
  positional callers in `tests/test_metrics.py` are updated).
- **Date guard.** Raises when the window's end is at or after `final_test_start`,
  `NGED_FINAL_TEST != "1"`, and `fold_id != "live"`. In ad-hoc scope the window end is the observed
  maximum `valid_time`. In leaderboard scope the window is the fold's `val_end`, so the leaderboard
  protection is the row-window check above.
- **Key check** per Q4, for `study/` groups in leaderboard scope. `ad_hoc` scope is exempt from both
  the key check and the row-window check, because it has no fold.
- **`metrics` keeps its `deps`** (`cv_power_forecasts`, `effective_capacity`,
  `clean_nged_power_data`) and reads actuals through `scan_cleaned_power`; it does not read
  `eligible_time_series`.
- **Promotion path.** `ml_core/mlflow_runs.py:list_promotable_runs` lists every fold run in every
  MLflow experiment, so a `study/` fold run would reach the `promotable_model_runs` candidates. The
  plan filters `study/` experiments there and refuses them in `promoted_model`.

**`scripts/forecasting/score_study.py`** (new). Arguments: predictions parquet path, study name,
fold id. It takes no code and no actuals.

- **Validate inputs.** The study name matches `^[a-z0-9_-]{1,64}$`, because the delta writers build
  their overwrite predicate by f-string and a quote could overwrite other experiments' rows (an
  out-of-scope defect reported to the maintainer, not fixed here). The fold id is in
  `leaderboard_fold_ids`, which refuses `live` and `smoke_test`. The file's `fold_id` column agrees.
  An existing `study/<name>` partition is not overwritten unless `--replace` is passed.
- **Clean environment.** The script clears its environment and keeps only `PATH`, `HOME`, `LANG`,
  and the `UV_*` variables, instead of checking a deny list: `Settings` reads `DATA_PATH_INTERNAL`,
  `DATA_PATH_DELIVERY`, `LOCAL_ARTIFACTS_PATH`, `METADATA_PATH`, `DATA_STORE_*`, `CV_CONFIG_PATH`,
  `MLFLOW_TRACKING_URI`, and `NGED_FINAL_TEST`, and any of them could repoint the actuals or the
  guard.
- **Batched, not whole-file.** A full fold is about 364M rows, and one `write_power_forecasts` call
  on the whole frame sorts and converts it to Arrow, which already exhausts a 29 GB machine. The
  script reads the file lazily per series batch, runs `require_same_row_keys`, then writes the first
  chunk with `replace_partition` and appends the rest, as `cv_power_forecasts` does. Key checks
  finish before the first write, so a rejected file leaves no partition behind.
- **Score.** It calls `dagster.materialize([metrics], run_config=...)` in-process with
  `evaluation_scope="leaderboard"` and a `PopulationFilter` pinned to `study/<name>` and the fold,
  following `scripts/forecasting/run_baseline_experiment.py`.
- **Experiment name with a slash.** `study/<name>` becomes a partition value in `power_forecasts`
  and `forecast_metrics` and an MLflow experiment name. The implementer verifies that delta-rs
  handles the slash; if it nests the partition directory, the prefix becomes `study__`.

**`packages/studies/src/studies/power.py`** (after issue #1082 merges) adds the cutoff to
`scan_power()`: one filter, `time < FINAL_TEST_START`. Studies find their data through
`REPO_DATA_DIR`, never through `Settings`, so `power.py` loads the date with
`contracts.config_schemas.load_cv_config` on `conf/cv/default.yaml` under the repository root
(`CV_CONFIG_PATH` is not consulted). If this plan is implemented first, #1082 lands first.

**`pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`.** Add `import-linter` to the dev group
and a `[tool.importlinter]` block with one `forbidden` contract: `ml_core.metrics` and
`ml_core.cv_helpers` may not import `dagster`, `mlflow`, or `studies` (`include_external_packages =
true`, indirect imports included). Add a `uv run lint-imports` step to CI after `ty`. If the strict
indirect check trips on `contracts`, narrow the contract to direct imports and record why in the
TOML. `uv.lock` is also edited by #147 and #1082; whichever lands later re-runs `uv lock`.

**Sysadmin steps (follow-up issue, not this PR).** A runbook the maintainer runs by hand: the Unix
user, `setfacl` on the data folders, setgid directories with umask 002, the research user's own
`uv` cache and credentials, tightening `/mnt/data` (mode 2777 today), the narrow `sudo` rule naming
`scripts/forecasting/score_study.py` with `env_reset` on, the truncated power copy, and **no write
access for the research user to `power_forecasts` or `forecast_metrics`**, because option B trusts
the reference partition.

## Splitting

**Split the sysadmin steps into a new issue, and keep one PR for the code.** The code half is
testable and reviewable in the diff. The sysadmin half is a checklist only the maintainer can run
and verify (`getfacl`, a failed `cat` as the research user), and its closing condition is a human
confirming it works. Bundling the two would hold #958 open until the maintainer has done sysadmin
work, or close it with the layer-1 claim unverified. The new issue is blocked by this one (the
`sudo` rule names the script). This PR updates the issue's checklist to point at it.

## Design-philosophy check

All of this runs in R&D, where `inherent-stability.md` says to fail fast: the guard, the key check,
and `score_study.py` raise, and none adds an asset check, so the WARN/non-blocking rule is not
engaged. The one exception is deliberate: an unfiltered leaderboard run skips stale `study/` groups
with a warning instead of raising, so a stale study cannot block reviewed experiments. Nothing
touches `live_forecast_assets.py` or the serving path, and the `live` exemption keeps production
monitoring unaffected. Hypothesis H2 is reworded in the docs (confirmed promotions per month, not
experiments per month). The design principle traded away is principle 3's "research runs on the
production pipeline" for autonomous studies, bought back by "one scoring path for all research".

## Tests

Tests marked *regression* pass on `main` today and guard against over-refusal; all others fail on
`main`.

- **Key check** (`tests/test_metrics.py`): a `study/` group missing one reference series raises
  `RowKeyMismatchError`; a group missing a subset of one series' keys raises; a group with an extra
  series or extra keys raises; a deterministic single-member study with the reference's keys scores
  (*regression*); a study with a different member count scores (*regression*); a missing reference
  partition raises naming the experiment; the same `valid_time_min`/`valid_time_max` filter applies
  to the reference scan.
- **Row window** (leaderboard scope): a group with a row whose `valid_time` is past `val_end`
  raises. This proves the date guard is not vacuous.
- **Date guard:** an ad-hoc group whose observed `valid_time` maximum is at or after
  `final_test_start` raises without `NGED_FINAL_TEST=1` and scores with it (`monkeypatch.setenv`);
  a `fold_id == "live"` ad-hoc group scores with the variable unset (*regression*).
- **No partial writes:** with two groups where the second is refused, the first writes no
  `forecast_metrics` rows and no MLflow run.
- **Stale study groups:** an unfiltered leaderboard run skips a `study/` group whose reference
  partition is missing, with a warning, and still scores the reviewed experiment.
- **`CvConfig`:** rejects a `final_test_start` not after every leaderboard `val_end`; rejects a
  `reference_experiment_name` starting `study/`.
- **Promotion:** `list_promotable_runs` omits `study/` experiments, and `promoted_model` refuses
  one.
- **`scan_power()`** (`packages/studies/tests`): returns no row at or after the cutoff on a small
  Delta table in `tmp_path`, with the cutoff injected.
- **`score_study`** (`tests/test_score_study.py`): rejects a name with a quote or a leading `study/`,
  a non-leaderboard fold id, and a file whose keys differ from the reference, leaving no partition;
  scores a matching file; ignores a `DATA_PATH_INTERNAL` set in the caller's environment. The MLflow
  file-store fixtures in `tests/test_metrics.py` move to `conftest.py` for it.
- **Import linter:** CI runs the real contract. The mutation-testing review adds `import mlflow` to
  `ml_core.metrics` once and confirms `lint-imports` fails.
- **Existing XGBoost leaderboard fold path** (`tests/test_cv_assets.py`) keeps passing (*regression*).

## Docs to update

Written in the present tense, with no history:

- `docs/design-philosophy/design-principles.md` principle 3 and `studies/README.md`: reviewed
  research keeps principle 3; autonomous studies get one scoring path, and a finding reaches
  production only as a written spec plus reviewed re-implementation.
- `docs/design-philosophy/engineering-hypotheses.md` H2 wording (labels never renumbered).
- `docs/roadmap/metrics-and-leaderboard.md`: the `study/` prefix, the guard and its cutoff, the
  reference-key refusal; delete the shipped steps of "Implementation details — final-test window"
  and keep step 3 (#960's conditional reservation).
- `docs/ml_experimentation/index.md`, `docs/ml_experimentation/dagster-workflow.md` (Step 9, the
  `metrics` description), and `docs/ml_experimentation/cross-validation-folds.md` (the two new
  fields): the autonomous-study route and promotion path.
- `docs/roadmap/auto-research.md`: the sentences about the refusal (it now compares keys with a
  reference experiment) and the promotion paths. Aligning layer 1 with Q1 moves to the follow-up
  issue.
- `scripts/README.md`: a line for `forecasting/score_study.py` under the existing `forecasting/`
  section.
- `.claude/skills/study/SKILL.md` and the reviewer checklist: a study's leaderboard number comes
  only from `scripts/forecasting/score_study.py`, and every number on a study page traces to a
  `forecast_metrics` row. CLAUDE.md's skills table is checked for a stale summary.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
uv run pytest --run-studies -n auto packages/studies
uv run lint-imports
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

Plus the pydoclint and docs-link checks from CI, run locally, and `ls plans` empty before ship. A
plain `pytest` skips the `packages/studies` tests, which `--run-studies` runs.

## Risks

- The `import-linter` strict indirect check may fail through `contracts`; the fallback is above.
- The key check depends on the reference experiment having been materialised for the fold, and a
  study is refused if the reference is re-materialised after the study was scored.
- `score_study.py` commits to `power_forecasts` while `live_forecasts` overwrites the `live`
  partition hourly. The implementer verifies that delta-rs resolves disjoint-partition commits, or
  the script retries.
- The slash in `study/<name>` as a partition value (see `score_study.py`).
- Requiring the reference's initialisation-time grid excludes studies built on another weather
  product until a second reference exists.

## Review log

**Simplicity review (Opus), accepted:** the date guard could never fire in leaderboard scope (added
the row-window check); dropped the cross-series regularity check; moved the study-reader migration
out; moved the runbook and layer-1 documentation to the follow-up issue; dropped the import-linter
self-test and the bypass-scan test.

**Simplicity review, rejected:** dropping the `NGED_FINAL_TEST` variable and the `live` exemption
(the issue specifies the variable, and ad-hoc scope has no window except the observed rows);
replacing `import-linter` with a `sys.modules` test (the issue names import-linter, and a static
check sees imports inside a function); the reference-experiment equality check was first offered as
an option and the maintainer later chose it.

**Correctness review 1 (Opus), accepted:** the fold-window row check; name and fold validation in
`score_study.py` against the f-string overwrite predicate; validating every group before scoring
any; moving the pure checks into `ml_core/metrics.py`; filtering `study/` runs from the promotion
path; the six `CvConfig(...)` constructions; the additional documentation pages; the Delta-commit
risk. Switched the recommendation from the eligible-series check to the reference experiment because
the eligible-series check would refuse the existing XGBoost fold.

**Correctness review 2 (Opus), accepted:** dropped `ensemble_member` from the key; a batched
`score_study.py`; the reference scan source and its filters; stale study groups skipped in
unfiltered runs; a cleared environment instead of a deny list; no research-user write access to the
forecast tables; `reference_experiment_name` validated and listed as a stored-data change; the
test list completed; the stale sentences rewritten; the slash risk; how `power.py` finds the CV
config. Nothing rejected.

**Main merged after the plan was written (2026-10-07):** PR #1040 changed `metrics` to read cleaned
power; the studies tests now run behind `--run-studies`; `uv.lock` changed heavily and is re-locked
at implementation.
