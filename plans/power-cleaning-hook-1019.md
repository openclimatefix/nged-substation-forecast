# Plan: a power-cleaning hook in the Dagster pipeline (#1019)

**The problem.** Cleaning algorithms for NGED's power telemetry are being written now, and the
pipeline has no place to put them. Every asset that reads observed power — training, CV prediction,
live forecasting, eligibility, and effective capacity — scans the raw `power_time_series` Delta
table directly, so a cleaning function today would have to be wired into five places by hand, and
nothing would report what it removed.

**The planned solution.** Add one function, `clean_power_time_series`, in a new module
`nged_data.cleaning`. The function takes a lazy `PowerTimeSeries` frame, applies an ordered tuple of
plain cleaning functions (empty today, so the frame passes through unchanged), and returns the
cleaned lazy frame together with a small dictionary of counts: rows and series in, rows and series
out, and rows dropped by each cleaning function. Every asset that reads power calls it at read time
and adds the counts to its own Dagster output metadata. The raw table is never rewritten and no new
table is created. The production caller, `live_forecasts`, wraps the call so that a failing cleaning
function degrades the slot to raw power and reports to Sentry; the research callers let the
exception propagate and fail the run.

## Verdict, size and departures

**Verdict: worth implementing, roughly as described.** The issue is current: no cleaning step exists
anywhere (`docs/roadmap/data-cleaning.md` says versions 0.1 to 0.3 train on uncleaned telemetry
apart from the timestamp repair at ingest).

**Size: complex.** The five triggers:

- What gets stored: **no** — read-time cleaning writes no new table and changes no contract.
- The production serving path: **yes** — `live_forecasts` gains a call and a fallback.
- A degradation rule: **yes** — a new degrade-on-failure path in production.
- More than one defensible design: **yes** — a materialised cleaned-power asset is the obvious
  alternative (settled below).
- Callers not nameable without searching: no — the five read sites are named below.

Complex buys both plan reviews (simplicity, then correctness) and both diff reviews.

**Departures from the issue body.**

- The issue speaks of a cleaning "asset". The plan makes it a read-time step, not an asset — see the
  next section for why.
- The issue says the function "returns the Polars LazyFrame it receives". The plan's function
  returns that frame unchanged plus a counts dictionary, because a lazy frame alone cannot carry the
  "rows dropped" statistic the issue asks for.

## Decision: a read-time step, not a materialised cleaned-power asset

**Clean at read time, because the policy for a failure differs between the production and research
callers, and only the caller knows which it is.** A materialised `cleaned_power_time_series` asset
would be one asset shared by `live_forecasts` (must degrade) and `trained_cv_model` (must fail
fast). Either the asset degrades and research silently trains on raw data when cleaning breaks, or
the asset fails and the live forecast loses its fresh power. The read-time step lets each caller
apply its own policy around one shared function.

The other reasons, in order of weight:

- **No new staleness path in production.** A materialised table becomes a second hop between ingest
  and `live_forecasts`. If that hop lags, the live forecast reads stale power although fresh raw
  power is on disk, and the job wiring that would schedule the hop lives in `defs/schedules.py`,
  which another session (#1020) owns.
- **Principle 15 ("transform in feature engineering, not in the ingest, unless it saves a lot of
  storage").** Cleaning throws rows away and saves no storage, so the principle puts it on the read
  side, where changing a cleaning rule means re-running an experiment rather than rebuilding a table.
- **Today the step is the identity.** A materialised copy of a table identical to its input is a
  second copy of the power archive bought for nothing.

**The cost of this choice is that a cleaning function sees only the caller's read window, not the
whole history.** `live_forecasts` reads 15 days; `trained_cv_model` reads the training window plus
the longest power lag. A function that is row-local (drop rows before a commissioning cut-off, drop
values outside a physical range) gives the same answer in every window. A detector that needs long
history — a stuck-value run crossing a window edge, a ramp detected against a neighbour's median —
would give a different answer in live than in training. That is training–serving skew, and is the
first open question below.

## What changes, file by file

### `packages/nged_data/src/nged_data/cleaning.py` (new)

- `PowerCleaningFunction` — a type alias for
  `Callable[[pt.LazyFrame[PowerTimeSeries]], pt.LazyFrame[PowerTimeSeries]]`. The interface a
  colleague drops a function into: take a lazy power frame, return a lazy power frame with rows
  dropped (or values changed), no other side effects, no `collect`.
- `POWER_CLEANING_FUNCTIONS: Final[tuple[PowerCleaningFunction, ...]] = ()` — the ordered registry.
  Adding a cleaning rule means writing a function and appending it here. The docstring states the
  row-local constraint from the decision above.
- `clean_power_time_series(power, cleaning_functions=None) -> CleanedPower` — applies each function
  in order (`None` resolves to `POWER_CLEANING_FUNCTIONS` at call time, so a test can monkeypatch the
  module attribute). Computes the counts with a single `pl.collect_all` over one
  `select(pl.len(), pl.col("time_series_id").n_unique())` per stage, which is an aggregate, not a
  materialisation, so principle 11 holds. Returns a `NamedTuple` `CleanedPower(power, stats)`.
- `stats` is a flat `dict[str, int]` with keys `power_cleaning/n_rows_in`,
  `power_cleaning/n_rows_out`, `power_cleaning/n_time_series_in`,
  `power_cleaning/n_time_series_out`, and `power_cleaning/n_rows_dropped_by/<function __name__>`
  per function. Flat string keys go straight into `context.add_output_metadata`, and the prefix
  keeps them clear of each asset's existing keys (Dagster raises on a duplicate key).

How this later feeds the delivery table that reports data problems is not built now: the per-stage
frames already exist inside `clean_power_time_series`, so an anti-join of each stage against the
one before yields the rows each function removed, labelled by function name. That needs no change
to the `PowerCleaningFunction` signature.

`nged_data` is the home because the step cleans NGED's telemetry and the package already depends on
`contracts`. A new package is only worth it if the cleaning functions grow dependencies (`scipy`, a
solar-geometry library) that `nged_data` should not carry; that is cheap to do later.

### `src/nged_substation_forecast/defs/cv_assets.py` (research — fail fast)

Out of bounds and untouched: `metrics` and its helpers (#958).

- `trained_cv_model`: call `clean_power_time_series(power_ts)` on the frame
  `load_engineering_inputs` returns, use `.power`, and merge `.stats` into the existing
  `add_output_metadata` call.
- `cv_power_forecasts`: the same inside the `init_time` chunk loop. The power window is identical in
  every chunk, so keep the stats from the last chunk and report them once after the loop.
- `effective_capacity`: clean the full-table scan before `compute_effective_capacity`; merge stats.
- `eligible_time_series`: today it calls `nged_data.storage.time_series_coverage(path)`, which scans
  the path itself. Split that function: a new `coverage_from_power(power: pl.LazyFrame)` holds the
  `group_by` + streaming `collect`, and `time_series_coverage(path)` keeps its existence check and
  calls it. `eligible_time_series` scans the table, cleans it, and calls `coverage_from_power` on
  the cleaned frame. `time_series_coverage`'s other callers (the `power_data_is_fresh` check and the
  ingest's `select_new_rows`) keep reading raw coverage, which is right: freshness and de-duplication
  are about what NGED delivered.

No `try` in any of these: an exception from a cleaning function propagates and fails the run.

### `src/nged_substation_forecast/defs/live_forecast_assets.py` (production — degrade)

- After `load_engineering_inputs`, call a private helper `_clean_or_degrade(power_ts, context)`
  that, inside the same `BaseException` guard the control-member probe uses (re-raising
  `KeyboardInterrupt`, `SystemExit`, and `DagsterExecutionInterruptedError`):
    - calls `clean_power_time_series`, then collects the cleaned frame and returns it re-wrapped
      as lazy, so a cleaning function that only fails at collect time fails *inside* the guard
      rather than later in `predict`. The live power window is 15 days of the trained series, a few
      MB at V2's 2,500 series, so this one collect is a measured exception to principle 11;
    - on failure, logs with `context.log.exception`, calls
      `report_asset_degradation(asset_name="live_forecasts", exc=exc)`, and returns the raw frame
      with `{"power_cleaning/degraded": True}` as its stats.
- On success the stats carry `power_cleaning/degraded: False`. Merge the stats into the existing
  `add_output_metadata` call.
- The forecast rows record no degradation flag: `PowerForecast` has no degradation column today, and
  adding one is a contract change outside this issue. The degradation is recorded in Sentry and the
  materialisation metadata, as the control-member probe does.

### `_engineering_inputs.py`

Unchanged. Keeping cleaning out of the shared loader keeps the failure policy at the caller, and the
loader keeps returning raw power.

## Design-philosophy check

- **Production degrades, research fails fast** (inherent-stability rules 1 and 7): `live_forecasts`
  catches, falls back to raw power, logs, and reports with the `degraded_asset=live_forecasts` tag
  (principle 16 — the telemetry names the asset). The research assets have no guard.
- **No asset check is added.** The counts go to output metadata. A `WARN`/`blocking=False` check
  would need the counts persisted somewhere a separate op can read them, and nothing yet defines a
  threshold to warn at. When the delivery table of data problems exists, a check over it is the
  natural place for a warning.
- **Principle 3 (one execution path)**: research and production call the same
  `clean_power_time_series`; only the failure handling differs.
- **Principle 11**: the counts are aggregates; the one `collect` is the live path's degrade guard,
  stated above.
- **Principle 15**: honoured by cleaning at read time.

## Tests

In `packages/nged_data/tests/test_cleaning.py` (new):

- `test_no_cleaning_functions_returns_input_unchanged` — the cleaned frame equals the input and
  `n_rows_in == n_rows_out`. Fails on `main` because the module does not exist; it pins the
  identity default the issue asks for.
- `test_rows_dropped_are_attributed_to_each_function` — two injected functions, one dropping two
  rows and one dropping a whole series; asserts each `n_rows_dropped_by/<name>` and
  `n_time_series_out`. This is the test that would catch a counting off-by-one or stats attributed
  to the wrong stage.

In `packages/nged_data/tests/test_storage.py`:

- The existing `time_series_coverage` tests keep passing unchanged, which is the evidence the split
  is a pure refactor. No new test.

In `tests/test_cv_assets.py`:

- `test_effective_capacity_reads_cleaned_power` — monkeypatch
  `nged_data.cleaning.POWER_CLEANING_FUNCTIONS` with a function dropping one series; assert the
  `effective_capacity` table lacks that series and the materialisation metadata carries the drop
  count. Fails on `main`: nothing applies the registry.
- `test_eligible_time_series_reads_cleaned_power` — same shape for eligibility.
- `test_a_failing_cleaning_function_fails_research` — a raising cleaning function makes
  `effective_capacity` fail.

In `tests/test_live_forecasts.py`:

- `test_live_forecasts_reads_cleaned_power` — a function dropping one trained series; assert the
  metadata's `power_cleaning/n_time_series_out` is one fewer than `n_time_series_in`. The written
  forecasts are not the assertion, because a series with no power can still be forecast from
  weather-only features.
- `test_a_failing_cleaning_function_degrades_the_slot_instead_of_failing_it` — a raising function
  (raising at collect time, via a `map_batches` that raises, to prove the guard's collect is what
  catches it); assert forecasts are written, `report_asset_degradation` was called with
  `asset_name="live_forecasts"`, and the metadata has `power_cleaning/degraded: True`. Modelled on
  `test_a_failing_control_member_probe_degrades_the_slot_instead_of_failing_it`.

`trained_cv_model` and `cv_power_forecasts` get no dedicated wiring test: their harnesses train real
models and the call is two lines. The first diff review should judge whether that is enough.

## Docs to update

- `docs/roadmap/data-cleaning.md` — a short section saying where cleaning functions go
  (`nged_data.cleaning.POWER_CLEANING_FUNCTIONS`), the row-local constraint, and that production
  falls back to raw power on failure while research fails.
- `docs/design-philosophy/inherent-stability.md` — add the cleaning fallback to the degradation
  ladder if the ladder enumerates per-input fallbacks (check while implementing).
- `packages/nged_data/README.md` — one line for the new module.
- The docstrings of the five assets touched, which are operator docs in the Dagster UI.

## Verification commands

The `implement-issue` green-before-push set (`ruff check`, `ruff format`, `ty check`, `pytest`,
the `pymarkdown` scan), plus `pydoclint` and the docs-link checker as CI runs them, plus
`uv run mkdocs build --strict` for the roadmap edit.

## Risks and open questions

1. **Window-dependent cleaning functions.** Recommendation: state the row-local constraint in the
   `PowerCleaningFunction` docstring now. When the first detector needs full history, it becomes a
   materialised asset of flagged rows (`power_data_problems`, computed over the whole table, which
   also feeds the delivery table), and the read-time step drops the flagged rows. That is a separate
   issue; this plan's interface does not block it.
2. **Should `metrics` score against cleaned actuals?** Probably yes for rows that describe a plant
   that no longer exists (the commissioning ramp), but `metrics` belongs to #958. Recommendation:
   leave it untouched here and raise it on #958 or a follow-up issue.
3. **Should `power_data_is_fresh` and the ingest de-duplication see cleaned power?** Recommendation:
   no — both are about what NGED delivered, not about what is fit to train on.
4. **Package home.** `nged_data.cleaning` unless the cleaning functions bring heavy dependencies;
   then a new `power_cleaning` package. Confirm with whoever is writing the functions.
