# Plan: retry `ecmwf_ens` when instantaneous variables have whole-slice gaps (#1020)

**Problem.** When Dynamical's IFS ENS store is still being filled, whole (ensemble member, lead time) slices of an instantaneous variable read as NaN. The conversion step then fails in `Nwp.validate` through base Patito validation, with a `DataFrameValidationError`. The `ecmwf_ens` asset does not retry that error, so the partition fails at once. The failure happened on 2026-10-01 and 2026-10-02, but only 2026-10-02 is shown to be recoverable by a retry: its store was filled about 3 hours 28 minutes after the 08:30 UTC attempt. The cause of 2026-10-01 is not established (its null counts differed per variable, and a re-run a day later passed), and the retry may or may not have helped.

**Solution.** After the download and before the conversion, a new check in `dynamical_data` looks for any (member, lead time) slice of an instantaneous variable that is NaN at every grid point. If it finds one, it raises the existing `NwpRunNotYetAvailable`, which the asset already turns into a `RetryRequested` on the existing ladder (8 retries, 30 minutes apart). A contract violation, such as a wrong dtype or an out-of-range value, still fails at once, because it never triggers the check. The retry budget stays as it is, because one attempt costs about 81 seconds and the ladder covers about 4 hours 12 minutes. When the retries run out, the partition stays unmaterialised, as today.

## Verdict, size and departures

**Verdict:** worth doing, as described. On `main`, `assets.py` catches only `NwpRunNotYetAvailable` and `NwpVariableWhollyMissing`, and instantaneous gaps reach base validation first (`weather_schemas.py`, `Nwp.validate`).

**Size: complex.** One line per trigger:

- **What gets stored:** nothing changes in this plan. Writing rows with nulls would change it, and that is not planned (see "After the retries run out").
- **Production serving path:** no change to `live_forecasts`. `ecmwf_ens` is the production ingest asset, though, and its failure mode decides which NWP run `live_forecasts` reads.
- **A degradation rule:** yes. The plan edits the retry ladder's coverage and states what happens when it is exhausted, both of which are rules in `inherent-stability.md`.
- **More than one defensible design:** yes. Four detectors are possible (see "Catching the gap failure").
- **Callers I could not name without searching:** no. `download_ecmwf_ens_data` and `NwpRunNotYetAvailable` each have one production caller, in `ecmwf_ens`.

Reviews: **both plan reviews and both diff reviews** (the degradation-rule trigger alone makes it complex).

**Departures from the issue body:**

- The issue suggested "a dedicated exception raised before base validation". The plan puts the check on the raw downloaded grid and reuses `NwpRunNotYetAvailable`, with no new exception and no change to the `Nwp` contract.
- The issue's "optional: use Dynamical's feed for diagnosis" item is left out. The issue says the feed's schema is undocumented, and the retry does not need it.
- The issue's second comment lists "the recorded age of the weather on the forecast row" as part of the fix. The stored rows already carry `nwp_init_time` and `power_fcst_init_time`, so the age is derivable and no contract change is needed (see open question 2). The plan narrows the issue by leaving any surfacing of the age out.

## Measurements

- **Cost of one attempt**, measured on 2026-10-06 on the workstation (idle, load average 0.8) for the 2026-10-05 run, with `open_ecmwf_ens_run`, `download_ecmwf_ens_data` and `convert_nwp_xarray_dataset_to_polars_dataframe` run in sequence: open 4 s, download 38 s, convert 39 s, 7,243,785 rows. That is about 81 s per attempt. `production-deployment.md` budgets about 10 minutes per attempt, so the measurement leaves wide room. A busy production day may be slower, which is why the plan does not tighten anything. Because the new check runs before the conversion, an attempt that fails on a gap costs about 42 s (open and download only).
- **Coverage of the ladder:** 9 attempts, 8 waits of 1,800 s, plus the work of each attempt, is about 4 hours 12 minutes after the 10:30 UTC start (about 4 hours 5 minutes when every attempt fails at the new check). The store of 2026-10-02 was filled about 3 hours 28 minutes after the 08:30 UTC attempt, so a series of attempts starting at 08:30 would have found the filled store with about 37 minutes to spare. Dynamical's maintainer has said a fix deployed on 2026-10-02 cuts recovery to about 20 minutes. The retry still matters when ECMWF's own run is late, and when that fix does not work. Recommendation: keep 8 retries at 30 minutes. Raising them moves the 16-hour deadline in `live_forecasts_are_healthy`, so any raise needs that deadline reviewed together.

## What changes, file by file

- **`packages/dynamical_data/src/dynamical_data/ecmwf_ens/download.py`**
    - Add a check of about 6 lines. It runs on the downloaded dataset: `ds[sorted(ECMWF_ENS_INSTANTANEOUS_VARS)].isnull().all(dim=["latitude", "longitude"])` gives one boolean per variable, member and lead time. If any is true, raise `NwpRunNotYetAvailable` naming each variable and its count of empty slices. Make it a small public function in `download.py` that the asset calls between `download_ecmwf_ens_data` and the conversion, inside the existing `try`. It does not go inside `download_ecmwf_ens_data`, which would put retry policy into the fetch and make a study caller that only wants raw-grid null counts receive `NwpRunNotYetAvailable`.
    - Widen the `NwpRunNotYetAvailable` docstring to "not in the catalog, or present with whole slices of an instantaneous variable still unwritten".
- **`src/nged_substation_forecast/defs/assets.py`:** comments and docstrings only. The `except` tuple is unchanged. Update the `_ECMWF_ENS_MAX_RETRIES` docstring (its inference that a `NwpRunNotYetAvailable` retry costs no download becomes false, because the new check raises it after the download; add the measured 81 s per attempt), the comment above the `try` in `ecmwf_ens`, the `ecmwf_ens` docstring, and `_NWP_INSTANTANEOUS_CHECK_DESCRIPTION` (a whole-slice gap never reaches the check, because the asset retries it first; scattered nulls still do).
- **`src/nged_substation_forecast/defs/schedules.py`:** the `ecmwf_ens_schedule` docstring and the `ecmwf_ens_job` description name the third retried failure.
- **`packages/contracts/src/contracts/weather_schemas.py`:** one docstring edit. The `NwpVariableWhollyMissing` docstring says instantaneous gaps are rejected "with no retry", which becomes false.
- **`docs/design-philosophy/inherent-stability.md`:** the failure-modes table (around lines 168-169) has a row "A whole ECMWF slice corrupt → Landed" that is wrong for an instantaneous variable. Correct it and add a row for "instantaneous whole-slice gap → retried for up to about 4 hours → missed run if still unfilled".
- **`docs/architecture/ecmwf-ens-known-issues.md`:** also the passage around line 359 about the 2026-07-14 run ("the day becomes a missed run"), which omits the retry. `docs/live_service/operations.md` lines 296-299 are accurate only for de-accumulated variables; update them if they claim no retry for instantaneous gaps. Main edits: rewrite the "A wholly-missing variable, and instantaneous nulls (fatal)" section so that whole-slice instantaneous gaps are retried and scattered ones still fail. Rewrite "A wholly-missing variable is retried, not failed outright" to cover both retried patterns, the 81 s measurement, and the 2026-10-02 timeline. Check `production-deployment.md` (the ladder-timing paragraph) for stale statements.

## Catching the gap failure without catching a contract bug

Four detectors were considered:

1. **Catch `DataFrameValidationError` in the asset and retry.** Rejected: it retries every contract violation (wrong dtype, out-of-range value, unit bug) for 4 hours.
2. **Retry on any null in an instantaneous column.** Rejected: it is broader than the evidence supports.
3. **A new `Nwp.validate` pre-check and a new exception, retrying only when every null is in a whole null slice.** Rejected after review: it adds an exception, a `ClassVar` duplicating `ECMWF_ENS_INSTANTANEOUS_VARS`, and a contract-side check, to defend against scattered nulls that have never been seen in an instantaneous variable.
4. **Chosen: any whole empty slice on the raw grid raises `NwpRunNotYetAvailable`.** The signature of "a worker has not committed yet" is a slice that is empty at every grid point, and every count in the issue is a multiple of 1,671 cells. A contract violation never makes a slice empty at every grid point, so it is not retried. A run that has an empty slice *and* a genuine scattered fault waits out the ladder, then fails in base validation exactly as today. The raw dataset is already cropped to the H3 grid's bounding box by `open_ecmwf_ens_run`, so "every grid point" means every grid point the run uses.

Two residual risks follow. A bug of our own that empties whole slices, or a permanently defective upstream run, would be retried for about 4 hours while holding an `ECMWF` pool slot, and then fail as it does today. A run with both a gap and a genuine contract violation reaches Sentry as `NwpRunNotYetAvailable` until the gap clears, which hides the violation for that time. `_sentry.py` already unwraps an exhausted `RetryRequested` to its cause, so Sentry would name `NwpRunNotYetAvailable` and its message, which lists the variables and slice counts.

## After the retries run out

- **Recommendation, and the issue's own third comment agrees: keep the partition unmaterialised.** Dagster fails the run after the last retry, as it does today. `live_forecasts` reads the freshest NWP run in the store and `live_forecasts_are_healthy` counts the missed runs, so serving degrades and nothing raises in the serving path. No contract change.
- **Alternative: drop the null slices and write the rest.** It needs `Nwp` or the conversion step to admit missing slices, and a half-published run would land once and never be re-ingested, the hazard #1021 describes. It belongs to #1021 and needs the maintainer's agreement, so this plan does not take it.

## Design-philosophy check

- `ecmwf_ens` is production code, so the change moves one input-absent state from "fail at once" to "wait, then degrade", which is the direction `inherent-stability.md` asks for. Malformed input (wrong types, out-of-range values) is still rejected at the Patito boundary.
- No asset check is added or changed beyond a description string. All three stay `WARN` with `blocking=False`, and their bodies are untouched, so they cannot newly raise.
- The new check runs inside the existing retry `try` and raises a typed exception, which is the existing pattern.
- The implementer should check `engineering-hypotheses.md` for the labels this supports (the aim is fewer manual re-runs) before citing any.

## Tests

- `packages/dynamical_data/tests/test_download.py` (beside the existing `NwpRunNotYetAvailable` test at line 91):
    - One (member, lead time) slice all-NaN raises `NwpRunNotYetAvailable` (on `main`: no function, so no raise). Parametrise over all nine instantaneous variables (catches checking only one) and put the empty slice at lead-0 (catches copying `exclude_lead_0` from `upstream_nulls`).
    - A slice that is NaN at every grid point except one does not raise (passes once the function exists; pins that the reduction is over both `latitude` and `longitude`, since a NaN row would wrongly raise if only one dimension were reduced).
    - An all-NaN slice in a de-accumulated variable, and an all-NaN `categorical_precipitation_type_surface` column, do not raise (pins the variable set; the categorical column is legitimately all-null before 2024-11-13).
    - The test dataset comes from `conftest.py::_build_ens_dataset`, which takes arrays broadcastable to `(lead, member, lat, lon)` on a 2×2 grid.
- `tests/test_assets.py`: parametrise `test_ecmwf_ens_retries_when_run_not_yet_available` (line about 1164) so `NwpRunNotYetAvailable` is raised from the new check function as well as from `open_ecmwf_ens_run`. Every `ecmwf_ens` test monkeypatches the download away, so without this nothing pins that the check sits inside the `try`.
- No network test. The 81 s measurement is recorded here and in the docs, not asserted.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest packages/dynamical_data packages/contracts tests/test_assets.py
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
uv run pydoclint .   # and the docs-link checker, per the run-every-CI-step-locally note
```

## Considered and rejected from the simplicity review

- **Do nothing**, because the 10:30 UTC schedule and Dynamical's faster re-run may be enough. Rejected: the retry still protects against a late ECMWF run and against Dynamical's fix failing, and the check is about 6 lines.
- **Put the check inside `download_ecmwf_ens_data`.** Rejected in the correctness review: it mixes fetching with retry policy.
- **Generalise `NwpVariableWhollyMissing` into one `NwpRunIncomplete` and put the detector in the contract.** Rejected as larger than the raw-grid check, which needs no new type.

## Risks and open questions

1. **Confirm: keep the partition unmaterialised after the retries run out?** Recommended, and consistent with the issue's third comment.
2. **NWP age on the forecast row needs no contract change.** `PowerForecast` already carries `nwp_init_time` and `power_fcst_init_time`, and the forecaster copies both onto every row. In the stored `power_forecasts` table (checked on 2026-10-07: 1,339,807,383 rows, 0 null `nwp_init_time`), the age of the weather run is `power_fcst_init_time - nwp_init_time`, for example 18 hours, 1 day, or 1 day 6 hours for the slots of 2026-10-05 and 2026-10-06. The issue's second comment says the table has no column recording the age, which is true only of a stored age column. No separate issue is needed for the contract; what remains is whether the age should be surfaced (a documented query, or a dashboard column), which is outside this issue.
3. **Detection is on the raw bounding box, not H3 cells.** A slice empty only over the H3 footprint but not the whole box would not be retried. No upstream mechanism known to us produces that, so the risk is theoretical.
