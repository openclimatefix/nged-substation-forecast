# Plan: retry `ecmwf_ens` when instantaneous variables have whole-slice gaps (#1020)

**Problem.** When Dynamical's IFS ENS store is still being filled, whole (ensemble member, valid time) slices of an instantaneous variable read as null. `Nwp.validate` rejects the frame through base Patito validation with a `DataFrameValidationError`, which the `ecmwf_ens` asset does not retry, so the partition fails at once. This happened on 2026-10-01 and 2026-10-02. On 2026-10-02 the store recovered about 3 hours 28 minutes after the 08:30 UTC attempt, so a retry would have recovered the run unaided.

**Solution.** Before base validation, `Nwp.validate` checks whether every null in an instantaneous column sits in a slice that is null from end to end. If so it raises a new exception, `NwpInstantaneousSlicesMissing`, and the asset retries it on the existing ladder (8 retries, 30 minutes apart). Any other null pattern still reaches base validation and fails at once, so a genuine contract bug is not retried. The retry budget stays as it is, because one attempt costs about 81 seconds and the ladder covers about 4 hours 12 minutes. When the retries run out, the partition stays unmaterialised, as today.

## Verdict, size and departures

**Verdict:** worth doing, as described. The issue's premise holds: the code on `main` raises `DataFrameValidationError` for instantaneous gaps (`weather_schemas.py` `Nwp.validate`) and `assets.py` catches only `NwpRunNotYetAvailable` and `NwpVariableWhollyMissing`.

**Size: complex.** One line per trigger:

- **What gets stored:** nothing changes in this plan. Writing rows with nulls would change it, and that is proposed as an open question below, not planned.
- **Production serving path:** no change to `live_forecasts`. `ecmwf_ens` is the production ingest asset, though, and its failure mode decides which NWP run `live_forecasts` reads.
- **A degradation rule:** yes. The plan edits the retry ladder and says what happens when it is exhausted, both of which are rules in `inherent-stability.md`.
- **More than one defensible design:** yes. Three detectors are possible (see "Catching the gap failure"), and there are two defensible end states after the retries run out.
- **Callers I could not name without searching:** no. `Nwp.validate` has one production caller, `convert_nwp_xarray_dataset_to_polars_dataframe`, plus tests.

Reviews: **both plan reviews and both diff reviews** (the degradation-rule trigger alone makes it complex).

**Departures from the issue body:**

- Its first candidate was "a dedicated exception raised before base validation when the nulls form whole slices". The plan takes that candidate, with a precise definition of "whole slice".
- The "optional: use Dynamical's feed for diagnosis" item is left out. The issue itself says the feed schema is undocumented, and the retry does not need it.
- Recording the NWP age on the forecast row changes `PowerForecast` and `live_forecast_assets.py`, which are out of bounds for this session. It is proposed as a separate issue (see open questions).

## Evidence for the measurements

- **Cost of one attempt, measured on 2026-10-06 on the workstation** for the 2026-10-05 run, with `open_ecmwf_ens_run`, `download_ecmwf_ens_data` and `convert_nwp_xarray_dataset_to_polars_dataframe` run in sequence: open 4 s, download 38 s, convert 39 s, 7,243,785 rows. That is about 81 s per attempt, on an idle machine. `production-deployment.md` budgets about 10 minutes per attempt, so the measurement leaves about 7 times that budget in hand. Production on a busy day may be slower, which is why the plan does not tighten anything.
- **Coverage of the ladder:** 9 attempts, 8 waits of 1,800 s, plus 9 × 81 s of work, is about 4 hours 12 minutes after the 10:30 UTC start, so the last attempt starts at about 14:30 UTC. The 2026-10-02 outage ran from Dynamical's last good update at 08:05 UTC to a filled store at about 11:58 UTC, which is 3 hours 53 minutes. The store was filled about 3 hours 28 minutes after the 08:30 UTC attempt, so an attempt series starting at the outage's start would be covered with about 45 minutes to spare, and one starting later is covered more easily. Dynamical's maintainer has said a fix deployed on 2026-10-02 cuts recovery to about 20 minutes.

## What changes, file by file

- **`packages/contracts/src/contracts/weather_schemas.py`**
    - Add `NwpInstantaneousSlicesMissing(ValueError)`, beside `NwpVariableWhollyMissing`, with a docstring saying it means "the upstream run is still being filled".
    - Add `Nwp.instantaneous_var_names` as the set of weather fields that are non-nullable (all weather variables minus `deaccumulated_var_names` and `categorical_var_names`), so the check does not hard-code names.
    - Add `Nwp._check_instantaneous_nulls_are_whole_slices(dataframe)`, called at the top of `validate` before `super().validate`. For each instantaneous column present in the frame, group by `(init_time, ensemble_member, valid_time)` and compare each slice's null count with its row count. If at least one null exists and every null-bearing slice is entirely null, raise `NwpInstantaneousSlicesMissing`, naming the variables and the number of whole slices. If any null-bearing slice is only partly null, return and let base validation fail as it does today.
    - Update the `NwpVariableWhollyMissing` docstring, which says the instantaneous variables are rejected with no retry.
- **`src/nged_substation_forecast/defs/assets.py`**
    - Import the new exception and add it to the `except (NwpRunNotYetAvailable, NwpVariableWhollyMissing)` tuple in `ecmwf_ens`; update the comment above that `try`.
    - Update the `_ECMWF_ENS_MAX_RETRIES` docstring ("the two ways" becomes three, and the measured 81 s per attempt replaces the "22.5 s at best" remark), the `ecmwf_ens` docstring, and `_NWP_INSTANTANEOUS_CHECK_DESCRIPTION`. The check description says a red result is "a mail to Dynamical.org, not a re-run: the run has landed". That stays true for scattered nulls, so it gains a sentence saying whole-slice gaps never reach the check because the asset retries them first.
    - `_ECMWF_ENS_MAX_RETRIES` and `_ECMWF_ENS_RETRY_DELAY_SECONDS` keep their values.
- **`src/nged_substation_forecast/defs/schedules.py`:** the `ecmwf_ens_schedule` docstring and `ecmwf_ens_job` description gain the third retried failure.
- **`docs/architecture/ecmwf-ens-known-issues.md`:** rewrite the "A wholly-missing variable, and instantaneous nulls (fatal)" section so that whole-slice instantaneous gaps are retried and scattered ones still fail at once. Rewrite "A wholly-missing variable is retried, not failed outright" to cover both retried patterns, the 81 s measurement, and the 2026-10-02 timeline. Check `production-deployment.md` (the ladder-timing paragraph) and `inherent-stability.md` for stale statements of "no retry".
- **`packages/dynamical_data`:** no change is planned. The detector belongs in the contract because `Nwp.validate` is where the other two retried patterns are detected, and `dynamical_data` only calls it.

## Catching the gap failure without catching a contract bug

Three detectors were considered:

1. **Catch `DataFrameValidationError` in the asset and retry.** Rejected: it retries every contract violation (a wrong dtype, an out-of-range value, a unit bug) for 4 hours, and hides real bugs behind retries.
2. **Retry on any null in an instantaneous column.** Rejected: a scattered null is not a sign of an unfinished run (it has never been seen in an instantaneous variable), and retrying cannot repair it.
3. **Retry only when every null is in a slice that is null from end to end (chosen).** The evidence supports it. Every null count in the issue is a multiple of 1,671 H3 cells (43,446 = 26 × 1,671; 3,509,100 = 2,100 × 1,671), and Dynamical's store held whole member-by-step slices when it was truncated. The signature of "worker has not committed yet" is a whole slice. Everything else still raises `DataFrameValidationError` at once.

The residual risk is a bug that nulls whole slices for a reason of our own (for example a mis-indexed conversion). It would be retried for 4 hours and then fail as it does today, and the Sentry event would still name the fault. That cost is accepted, and the same risk already exists for `NwpVariableWhollyMissing`.

## What happens when the retries run out (proposal for the maintainer)

- **Recommended: keep the partition unmaterialised.** Dagster fails the run after the last retry, as it does today. `live_forecasts` reads the freshest NWP run in the store and `live_forecasts_are_healthy` counts the missed runs, so serving degrades and nothing raises in the serving path. No contract changes.
- **Alternative: drop the null slices and write the rest.** It needs `Nwp` to admit missing slices or the conversion step to drop them. It also lands a half-published run once and never re-ingests it, which is the hazard #1021 describes. It belongs to #1021, so this plan does not take it.
- **Telemetry on exhaustion.** The failure message names the partition and the variables and slice counts, so the Sentry event names the fault instead of only `RetryRequested`'s eventual error. The implementer should check `_sentry.py` to see how an exhausted `RetryRequested` is reported, and add a tag only if the current event does not name the series and run.

## Design-philosophy check

- `ecmwf_ens` is production code, so the change moves one input-absent state from "fail at once" to "wait, then degrade", which is the direction `inherent-stability.md` asks for. Malformed input (scattered nulls, wrong types) is still rejected at the Patito boundary.
- No asset check is added or changed beyond a description string, and all three stay `WARN` with `blocking=False`. The check bodies are untouched, so they cannot newly raise.
- The new detector runs inside `Nwp.validate`, which is already inside the retry `try`, and raising a typed exception there is the existing pattern.
- It supports hypothesis `H1` only indirectly (fewer manual re-runs); the implementer should confirm which labels in `engineering-hypotheses.md` apply before citing any.

## Tests

- `packages/contracts/tests/test_weather_schemas_validation.py`
    - A frame with one instantaneous variable null across whole `(member, valid_time)` slices raises `NwpInstantaneousSlicesMissing` (on `main`: raises `DataFrameValidationError`).
    - A frame with a single null row in one slice raises `DataFrameValidationError` and not the new type (passes on `main`, and pins the "genuine bug" half so a later edit cannot widen the retry).
    - A frame mixing a whole null slice and a partly null slice raises `DataFrameValidationError`.
    - The 2026-10-02 shape: 50 of 51 members null for the last 42 steps, across all nine variables, raises the new type.
    - A de-accumulated whole-column null still raises `NwpVariableWhollyMissing`, and a frame with no nulls validates.
- `tests/test_assets.py`: extend the existing recording-`RetryRequested` test (near line 437) with a parametrised case where conversion raises `NwpInstantaneousSlicesMissing` and the asset raises `RetryRequested` with `max_retries=_ECMWF_ENS_MAX_RETRIES`; a `DataFrameValidationError` from conversion must still propagate.
- No network test. The 81 s measurement is a one-off recorded above and in the docs, not a test.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest packages/contracts packages/dynamical_data tests/test_assets.py
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
uv run pydoclint .   # and the docs-link checker, per the run-every-CI-step-locally note
```

## Risks and open questions

1. **After the retries run out: keep the partition unmaterialised (recommended), or write the run without the null slices?** The second option is a contract change and overlaps #1021, so it needs your decision.
2. **NWP age on the forecast row.** Recording how old the weather run was on each `PowerForecast` row changes `power_schemas.py` and `live_forecast_assets.py`, which this session may not edit. Recommendation: a separate issue, so the degradation record exists once.
3. **NaN versus null.** The implementer should confirm that the converted frame carries gaps as nulls and not NaN before relying on `is_null`.
4. **Retry budget.** The recommendation is to keep 8 retries at 30 minutes. Raising it moves the 16-hour deadline in the freshness check, so any raise needs that deadline reviewed together.
5. **The sibling failure on 2026-10-01** has no established cause. Its counts were also whole slices (26 and 3 slices per variable, so a partly published run), and the detector treats it the same way.
