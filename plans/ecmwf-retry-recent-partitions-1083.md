# Plan: retry ecmwf_ens only for recent partitions, and warn Sentry on the first failed attempt (#1083)

**Problem.** `ecmwf_ens` retries a failing run for about 4 hours whatever the age of the partition, so a backfill over bad runs holds an `ECMWF` pool slot for 4 hours per partition. In a scheduled run, Sentry hears nothing until the last retry fails.

**Solution.** Two changes in the `except (NwpRunNotYetAvailable, NwpVariableWhollyMissing)` branch of `ecmwf_ens`. If the run's `init_time` is more than 36 hours old, the asset re-raises the original exception at once instead of requesting a retry. On the first failed attempt (`context.retry_number == 0`) of a run that will retry, the asset sends the caught exception to Sentry at warning level through a thin new wrapper, `report_asset_retry`, tagged `retrying_asset`, with a note naming the partition.

## Verdict, size and departures

**Verdict:** worth doing; the maintainer chose options 3 and 4 after discussion of #1020. No departures from the issue.

**Size: complex.**

- **What gets stored:** nothing changes.
- **Production serving path:** no change to `live_forecasts`.
- **A degradation rule:** yes. The change edits which failures retry, which is a rule in `inherent-stability.md`.
- **More than one defensible design:** yes. The retry could be limited by age or by backfill tag, and the warning could be a message event or the caught exception.
- **Callers I could not name without searching:** no. `ecmwf_ens` has one retry site and `_sentry.py` helpers are called from named places.

Reviews: the maintainer asked for the PR without a separate plan approval, so the plan is reviewed by one simplicity sub-agent and one correctness sub-agent before code, and the diff by two sub-agents after code, as the size requires.

## What changes, file by file

- **`src/nged_substation_forecast/defs/assets.py`**
    - Add `_ECMWF_ENS_MAX_RETRY_AGE: Final[timedelta] = timedelta(hours=36)`, documented as: it must exceed the age of a healthy run at its last retry (about 15 hours: the 10:30 UTC schedule plus the 4-hour ladder).
    - Add a pure helper `_is_too_old_to_retry(nwp_init_time: datetime, now: datetime) -> bool`, so a test can pin the 36-hour boundary with a fixed `now`.
    - In `ecmwf_ens`, at the top of the `except` branch: `exc.add_note(f"ecmwf_ens partition {partition_date_str}")`, so every Sentry event for the failure, including the run-failed event after the ladder, names the partition. Then, if `_is_too_old_to_retry(nwp_init_time=nwp_init_time, now=datetime.now(UTC))`, log a warning and `raise` (no `RetryRequested`). Otherwise, if `context.retry_number == 0`, call `report_asset_retry(asset_name="ecmwf_ens", exc=exc)`, then request the retry.
    - Update the `_ECMWF_ENS_MAX_RETRIES` docstring, the `ecmwf_ens` docstring (which tells the operator what a long-running materialisation means), and the comment above the `try`.
- **`src/nged_substation_forecast/_sentry.py`:** add `report_asset_retry(asset_name, exc)`, a three-line wrapper over `_capture_tagged` with tag `retrying_asset` (distinct from `fault_category=run_failed` and from `degraded_asset`, which means "caught and carried on"). Give `_capture_tagged` an optional `level` argument, set on the forked scope, so this event is a warning and every existing caller stays at error level. No fingerprint: the exception is genuinely caught, so its stack trace groups the event.
- **`src/nged_substation_forecast/defs/schedules.py`:** the `ecmwf_ens_schedule` docstring says a run that is not usable is retried; add the 36-hour limit.
- **`docs/architecture/ecmwf-ens-known-issues.md`:** in "Why only these three failures are retried", replace the sketch of a first-attempt warning with what the code does, and add the 36-hour rule and its reason. Update the "up to 4h" row in `docs/design-philosophy/inherent-stability.md` (line about 170) and check `docs/live_service/operations.md`.

## Design-philosophy check

- Both changes are production-side and neither adds a raise: the age branch re-raises the exception that would have been raised after the ladder, only sooner, and the warning helper never raises and is a no-op without a DSN. This keeps the always-output path unchanged: an unmaterialised partition leaves `live_forecasts` reading the freshest run.
- Design principle 16 supports the warning: the first failed attempt names the partition and the exception message.
- No asset check changes.

## Tests

- `tests/test_assets.py`
    - The two existing retry tests (`test_ecmwf_ens_retries_when_run_not_yet_available`, `test_ecmwf_ens_retries_when_a_variable_is_wholly_missing`) use old partition keys (2024-05-01 and 2024-12-01), which the 36-hour rule now fails at once. Move them to today's partition key (`datetime.now(UTC).date().isoformat()`), which is always 0 to 24 hours old and is accepted because `end_offset=1`.
    - New: an old partition key with a raising download re-raises the original exception and not `RetryRequested` (fails on `main`: it raises `RetryRequested`).
    - New, through `materialize` with `_ECMWF_ENS_RETRY_DELAY_SECONDS` patched to 0, as `test_power_time_series_and_metadata_gives_up_after_its_retry_budget` does: a young partition whose download keeps raising calls `report_asset_retry` exactly once across all attempts (fails on `main`: the helper does not exist), and an old partition never calls it. Direct invocation cannot test `retry_number`, because `context.retry_number` raises `AttributeError` on `DirectOpExecutionContext`.
    - New: `_is_too_old_to_retry` is false at exactly 36 hours and true one second later.
- `tests/test_sentry.py`: add one row for `report_asset_retry` to the existing parametrised `test_degradation_reporters_capture_the_exception_and_tag_the_name`, and assert the event level is warning for this reporter and error for the others.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run --all-packages ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
uvx pydoclint --config=pyproject.toml packages src
```

## Risks and open questions

1. **36 hours.** It must exceed about 15 hours and should include the previous day's partition for a manual re-run the next morning; 36 hours does both. A manual re-run of an older partition fails at once if the run is not ready, which is acceptable because a person is watching.
2. **Warning only on the first attempt, as the issue says.**

## Considered and rejected from the simplicity review

- **Warn on every failed attempt, to drop the `retry_number` branch.** Rejected: the issue asks for the first attempt, and nine events on a bad day add noise.
- **Skip the retry for backfill runs (the `dagster/backfill` tag) instead of by age.** Rejected: it departs from the issue, misses a manual re-run of an old partition, and would not retry a backfill that includes today's partition.
