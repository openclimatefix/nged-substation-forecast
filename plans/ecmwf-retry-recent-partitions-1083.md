# Plan: retry ecmwf_ens only for recent partitions, and warn Sentry on the first failed attempt (#1083)

**Problem.** `ecmwf_ens` retries a failing run for about 4 hours whatever the age of the partition, so a backfill over bad runs holds an `ECMWF` pool slot for 4 hours per partition. In a scheduled run, Sentry hears nothing until the last retry fails.

**Solution.** Two changes in the `except (NwpRunNotYetAvailable, NwpVariableWhollyMissing)` branch of `ecmwf_ens`. If the run's `init_time` is more than 36 hours old, the asset re-raises the original exception at once instead of requesting a retry. On the first failed attempt (`context.retry_number == 0`) of a run that will retry, the asset sends a Sentry warning through a new helper, `report_upstream_retry`.

## Verdict, size and departures

**Verdict:** worth doing; the maintainer chose options 3 and 4 after discussion of #1020. No departures from the issue.

**Size: complex.**

- **What gets stored:** nothing changes.
- **Production serving path:** no change to `live_forecasts`.
- **A degradation rule:** yes. The change edits which failures retry, which is a rule in `inherent-stability.md`.
- **More than one defensible design:** yes. The age test could use the partition key or the run's `init_time`; the warning could be a new message-level event or reuse `report_asset_degradation`.
- **Callers I could not name without searching:** no. `ecmwf_ens` has one retry site and `_sentry.py` helpers are called from named places.

Reviews: the maintainer asked for the PR without a separate plan approval, so the plan is reviewed by one simplicity sub-agent and one correctness sub-agent before code, and the diff by two sub-agents after code, as the size requires.

## What changes, file by file

- **`src/nged_substation_forecast/defs/assets.py`**
    - Add `_ECMWF_ENS_MAX_RETRY_AGE: Final[timedelta] = timedelta(hours=36)`, documented as: it must exceed the age of a healthy run at its last retry (about 15 hours: the 10:30 UTC schedule plus the 4-hour ladder).
    - Add `_utc_now() -> datetime`, so tests can patch the clock.
    - In `ecmwf_ens`, in the retry branch: if `_utc_now() - nwp_init_time > _ECMWF_ENS_MAX_RETRY_AGE`, log a warning and `raise` the original exception (no `RetryRequested`); otherwise, if `context.retry_number == 0`, call `report_upstream_retry(settings=settings, asset_name="ecmwf_ens", partition_key=partition_date_str, exc=exc)`, then request the retry.
    - Update the `_ECMWF_ENS_MAX_RETRIES` docstring, the `ecmwf_ens` docstring (which tells the operator what a long-running materialisation means), and the comment above the `try`.
- **`src/nged_substation_forecast/_sentry.py`:** add `report_upstream_retry(settings, asset_name, partition_key, exc)`. It sends a warning-level message naming the asset, the partition, and the exception message, on a forked scope with tag `retrying_asset=<asset_name>` (not `fault_category=run_failed`) and fingerprint `[UPSTREAM_RETRY_FINGERPRINT, asset_name, settings.sentry_environment]`. It is a no-op when Sentry is uninitialised and never raises. Add the fingerprint constant with its docstring.
- **`src/nged_substation_forecast/defs/schedules.py`:** the `ecmwf_ens_schedule` docstring says a run that is not usable is retried; add the 36-hour limit.
- **`docs/architecture/ecmwf-ens-known-issues.md`:** in "Why only these three failures are retried", replace the sketch of a first-attempt warning with what the code does, and add the 36-hour rule and its reason. Check `docs/live_service/operations.md` and `inherent-stability.md` for statements of "retries up to 4h" that now need the age limit.

## Design-philosophy check

- Both changes are production-side and neither adds a raise: the age branch re-raises the exception that would have been raised after the ladder, only sooner, and the warning helper never raises and is a no-op without a DSN. This keeps the always-output path unchanged: an unmaterialised partition leaves `live_forecasts` reading the freshest run.
- Design principle 16 supports the warning: the first failed attempt names the partition and the exception message.
- No asset check changes.

## Tests

- `tests/test_assets.py` (new, beside `test_ecmwf_ens_retries_when_run_not_yet_available`):
    - A run older than 36 hours that raises `NwpRunNotYetAvailable` re-raises it and does not raise `RetryRequested` (fails on `main`: it raises `RetryRequested`).
    - A run younger than 36 hours still raises `RetryRequested` (passes on `main`; it pins the other side of the boundary, with the clock patched to just inside 36 hours and just outside).
    - On `retry_number == 0`, `report_upstream_retry` is called once with the partition key and the exception; on `retry_number == 1`, it is not called; for an old partition, it is not called (fails on `main`: the helper does not exist).
- `tests/test_sentry.py`: `report_upstream_retry` sends one warning-level message with tag `retrying_asset`, the fingerprint, and a message naming the asset, partition, and exception text; it is a no-op without a DSN; it does not raise when `capture_message` raises.

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
2. **Clock in the asset.** The age test reads the wall clock, which is the only new nondeterminism; `_utc_now` isolates it for tests.
