# Plan: a power-cleaning hook in the Dagster pipeline (#1019)

**The problem.** Cleaning algorithms for NGED's power telemetry are being written now, and the
pipeline has no place to put them. Every asset that reads observed power — training, CV prediction,
live forecasting, eligibility, and effective capacity — scans the raw `power_time_series` Delta
table directly, so a cleaning function today would have to be wired into five places by hand, and
nothing would report what it removed.

**The planned solution.** Add a new module `nged_data.cleaning` holding an ordered tuple of
cleaning steps, empty today. Each step pairs a cleaning function (a lazy power frame in, a lazy
power frame out) with an optional logging function. `clean_power_time_series` applies the steps in
order and stays lazy, so with no steps the frame passes through unchanged. `power_cleaning_stats`
reports, per step, how many rows the step dropped, how many series it touched, and the time range
it dropped, plus whatever the step's own logging function computes from the rows it dropped (for
example the min and max of the extreme values removed). All of it is one lazy query collected once.
Five assets that read power call the cleaning at read time, and four add the stats to their own
Dagster output metadata. The raw table is never rewritten and no new table is created. In
production, `live_forecasts` guards the cleaning and the stats separately: a failing cleaning
function falls back to raw power, a failing logging function loses only the stats, and both report
to Sentry. The research assets let either exception propagate and fail the run.

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

- The issue describes one function that both cleans and reports. The plan splits those into two
  functions, so a caller that runs the cleaning many times on the same window
  (`cv_power_forecasts`, once per `init_time` chunk) does not compute the stats each time.

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

**The interface a colleague writes against is a cleaning step: a cleaning function, optionally
paired with a logging function.**

- `PowerCleaningStep` — a frozen dataclass with two fields:
    - `clean: Callable[[pt.LazyFrame[PowerTimeSeries]], pt.LazyFrame[PowerTimeSeries]]` — takes a
      lazy power frame and returns one with rows dropped or values changed. No side effects and no
      `collect`, and it must be deterministic, because `power_cleaning_stats` re-applies it to
      build the frames it compares. The step's name is `clean.__name__`, so a `lambda` is not
      allowed (its name would be `<lambda>`).
    - `summarise: Callable[[CleaningStepFrames], pl.LazyFrame] | None = None` — the optional
      logging function. It returns a lazy query that collects to exactly one row; each column
      becomes one metadata entry. Returning a lazy query rather than Python values is what lets
      every step's summary run in one collect, and stops a logging function forcing the data into
      memory on its own.
- `CleaningStepFrames` — a frozen dataclass of three lazy frames, the logging function's input:
    - `before` — the frame going into this step.
    - `after` — the frame coming out.
    - `dropped` — the rows of `before` whose `(time_series_id, time)` key is absent from `after`
      (an anti-join). Most logging functions read only `dropped`; `before` and `after` are there
      for a step that changes values rather than dropping rows, whose logging function joins the
      two to summarise the change.
- `POWER_CLEANING_STEPS: Final[tuple[PowerCleaningStep, ...]] = ()` — the ordered registry. Adding a
  cleaning rule means writing a function (and optionally a logging function) and appending a step
  here. The docstring states the row-local constraint from the decision above.
- `clean_power_time_series(power, steps=None) -> pt.LazyFrame[PowerTimeSeries]` — applies each
  step's `clean` in order and stays lazy. `None` resolves to `POWER_CLEANING_STEPS` at call time, so
  a test can monkeypatch the module attribute.
- `power_cleaning_stats(power, steps=None) -> dict[str, int | float | str | bool]` — takes the raw
  frame, rebuilds each step's `CleaningStepFrames`, and runs one `pl.collect_all` over:
    - the whole run: `power_cleaning/n_rows_in`, `power_cleaning/n_rows_out`,
      `power_cleaning/n_time_series_in`, and `power_cleaning/n_time_series_out`;
    - every step, built in, whether or not it has a logging function:
      `power_cleaning/<step>/n_rows_dropped`, `power_cleaning/<step>/n_time_series_affected`, and
      `power_cleaning/<step>/first_time_dropped` and `.../last_time_dropped` as ISO-8601 strings
      (empty when nothing was dropped);
    - every step's logging function, if it has one: `power_cleaning/<step>/<column>` per column.

  The flat string keys go straight into `context.add_output_metadata`, and the prefix keeps them
  clear of each asset's existing keys (Dagster raises on a duplicate key). The function raises
  `ValueError` on our own bugs: two steps sharing a name, a logging function whose result is not
  exactly one row, or a logging-function column colliding with a built-in key. Every value is
  aggregated, so principle 11 holds; the anti-joins are the cost, one per step over the caller's
  read window.

Because each step's `dropped` frame is labelled by the step's name, the later delivery table of data
problems can be built from those frames without changing the interface.

`nged_data` is the home because the step cleans NGED's telemetry and the package already depends on
`contracts`. A new package is only worth it if the cleaning functions grow dependencies (`scipy`, a
solar-geometry library) that `nged_data` should not carry; that is cheap to do later.

### `src/nged_substation_forecast/defs/cv_assets.py` (research — fail fast)

Out of bounds and untouched: `metrics` and its helpers (#958).

- `trained_cv_model`: call `clean_power_time_series` on the frame `load_engineering_inputs`
  returns, and merge `power_cleaning_stats` of that raw frame into the existing
  `add_output_metadata` call.
- `cv_power_forecasts`: call `clean_power_time_series` inside the `init_time` chunk loop and report
  no stats. The power window is the validation window in every chunk, and `trained_cv_model`
  already reports the training window's stats.
- `effective_capacity`: clean the full-table scan before `compute_effective_capacity`; merge the
  stats. This is the one caller whose stats query runs over the whole table, so each step's
  anti-join runs over the whole table too.

- `eligible_time_series`: today it calls `nged_data.storage.time_series_coverage(path)`, which scans
  the path itself. Split that function: a new `coverage_from_power(power: pl.LazyFrame)` holds the
  `group_by` + streaming `collect`, and `time_series_coverage(path)` keeps its existence check and
  calls it. `eligible_time_series` keeps today's absent-table behaviour (an empty population, not a
  `TableNotFoundError`) by checking `delta_table_exists` first; when the table exists it scans the
  table, cleans it, and calls `coverage_from_power` on the cleaned frame. The `coverage_from_power`
  docstring notes that a cleaning function using `map_batches` or `.over()` may push the streaming
  engine back to in-memory. `time_series_coverage`'s other callers (the `power_data_is_fresh` check
  and the ingest's `select_new_rows`) keep reading raw coverage, which is right: freshness and
  de-duplication are about what NGED delivered.

No `try` in any of these: an exception from a cleaning function propagates and fails the run.

### `src/nged_substation_forecast/defs/live_forecast_assets.py` (production — degrade)

**Two guards, not one, so a failing logging function costs the stats and never the cleaning.**
Logging is not what the forecast depends on: a bug in a logging function that switched the live
forecast to raw power would degrade the forecast for a fault that has nothing to do with the
forecast. Both guards sit right after `load_engineering_inputs` and copy the shape of the existing
control-member guard (the same `BaseException` catch, re-raising `KeyboardInterrupt`,
`SystemExit`, and `DagsterExecutionInterruptedError`). No shared helper for the three guards: short
inline blocks read more clearly than an abstraction.

- **Cleaning guard.** Call `clean_power_time_series`, collect the cleaned frame and re-wrap it as
  lazy. The collect means a cleaning function that only fails at collect time fails *inside* the
  guard rather than later in `predict`. The live power window is 15 days of the trained series, a
  few MB at V2's 2,500 series, so this one collect is a deliberate exception to principle 11.
    - On failure: `context.log.exception("Power cleaning failed; forecasting this slot from
      uncleaned power")`. The guard also covers the raw Delta read the collect triggers, so the
      message must not claim the cleaning function was at fault. Then call
      `report_asset_degradation(asset_name="live_forecasts", exc=exc)`, carry on with the raw
      frame, set `power_cleaning/degraded: True`, and skip the stats guard.
- **Stats guard**, run only when cleaning succeeded. Call `power_cleaning_stats` on the raw frame.
    - On failure: `context.log.exception("Power cleaning statistics failed; the slot was still
      forecast from cleaned power")`, call `report_asset_degradation(asset_name="live_forecasts",
      exc=exc)`, and set `power_cleaning/stats_failed: True` in place of the stats.
- On success the metadata carries `power_cleaning/degraded: False`, `power_cleaning/stats_failed:
  False`, and the stats, merged into the existing `add_output_metadata` call.

- The forecast rows record no degradation flag: `PowerForecast` has no degradation column today, and
  adding one is a contract change outside this issue. The degradation is recorded in Sentry and the
  materialisation metadata, as the control-member probe does.

### `_engineering_inputs.py`

Unchanged. Keeping cleaning out of the shared loader keeps the failure policy at the caller, and the
loader keeps returning raw power.

## Design-philosophy check

- **Production degrades, research fails fast** (inherent-stability rules 1 and 7): `live_forecasts`
  falls back to raw power when cleaning fails, drops only the stats when logging fails, logs, and
  reports with the `degraded_asset=live_forecasts` tag (principle 16 — the telemetry names the
  asset, and the log message names which of the two steps failed). The research assets have no
  guard.

- **No asset check is added.** The counts go to output metadata. A `WARN`/`blocking=False` check
  would need the counts persisted somewhere a separate op can read them, and nothing yet defines a
  threshold to warn at. When the delivery table of data problems exists, a check over it is the
  natural place for a warning.
- **Principle 3 (one execution path)**: research and production call the same
  `clean_power_time_series`; only the failure handling differs.

- **Principle 11**: the stats are aggregates collected once; the one `collect` of a frame is the
  live cleaning guard's, stated above.

- **Principle 15**: honoured by cleaning at read time.

## Tests

In `packages/nged_data/tests/test_cleaning.py` (new):

- `test_no_cleaning_steps_returns_input_unchanged` — the cleaned frame equals the input and
  `n_rows_in == n_rows_out`. Fails on `main` because the module does not exist; it pins the
  identity default the issue asks for.
- `test_cleaning_steps_run_in_order` — two injected steps where the second only has an effect if the
  first ran before it; asserts the result.
- `test_rows_dropped_are_attributed_to_each_step` — a step dropping zeros, then a step dropping
  values outside a plausible range whose lower bound is above zero, on data where one series has
  two zeros and another has one extreme value.
  Asserts each step's `n_rows_dropped`, `n_time_series_affected`, and first and last dropped time,
  and the run's four totals. Would fail if `dropped` were computed against the raw frame rather than
  the step's own `before`, which would double-count a row both steps match; the data includes such
  a row (a zero, which the second step's range also excludes) to pin that.
- `test_a_logging_function_summarises_the_rows_its_step_dropped` — the extreme-value step paired
  with a logging function returning the min and max of `dropped.power`; asserts both metadata
  values.
- `test_a_logging_function_sees_values_a_step_changed` — a step that clips values rather than
  dropping rows, with a logging function joining `before` to `after` to count changed rows; asserts
  `n_rows_dropped == 0` and the changed count.
- `test_power_cleaning_stats_rejects_our_own_bugs` — parametrised over the three `ValueError` cases:
  duplicate step names, a two-row summary, and a summary column named `n_rows_dropped`.

In `packages/nged_data/tests/test_storage.py`:

- The existing `time_series_coverage` tests keep passing unchanged, which is the evidence the split
  is a pure refactor. No new test.

In `tests/test_cv_assets.py`:

- `test_effective_capacity_reads_cleaned_power` — monkeypatch
  `nged_data.cleaning.POWER_CLEANING_STEPS` with a step dropping one series; assert the
  `effective_capacity` table lacks that series and the materialisation metadata carries the step's
  `n_rows_dropped`. Fails on `main`: nothing applies the registry.

- `test_eligible_time_series_reads_cleaned_power` — same shape for eligibility, dropping series 1:
  the fixture makes only series 1 eligible for `FOLD_ID`, so dropping any other series would leave
  eligibility unchanged and the population assertion would pass on `main`.

In `tests/test_trained_cv_model.py`:

- `test_trained_cv_model_reads_cleaned_power` — the existing harness already materialises
  `trained_cv_model` with two eligible series. Drop one through the registry and assert the
  metadata's `power_cleaning/n_time_series_out` is one fewer than `n_time_series_in`. Fails if the
  cleaning call is deleted from `trained_cv_model`.

In `tests/test_live_forecasts.py`:

- `test_a_failing_cleaning_function_degrades_the_slot_instead_of_failing_it` — a raising cleaning
  function (raising at collect time, via a `map_batches` that raises, to prove the guard's collect
  is what catches it); assert forecasts are written, `report_asset_degradation` was called with
  `asset_name="live_forecasts"`, and the metadata has `power_cleaning/degraded: True`. Modelled on
  `test_a_failing_control_member_probe_degrades_the_slot_instead_of_failing_it`. The test also
  proves the live wiring: if `live_forecasts` never called the cleaning, nothing would raise and
  `report_asset_degradation` would not be called.
- `test_a_failing_logging_function_keeps_the_cleaned_power` — a step whose cleaning drops one
  trained series and whose logging function raises. Assert forecasts are written,
  `report_asset_degradation` was called, and the metadata has `power_cleaning/degraded: False` and
  `power_cleaning/stats_failed: True`. Fails if the two guards are merged into one, because the
  merged guard would set `degraded: True` and fall back to raw power.
- The existing happy-path live test gains one assertion: the metadata carries
  `power_cleaning/degraded: False`, `power_cleaning/stats_failed: False`, and equal in/out row
  counts.

`cv_power_forecasts` gets no wiring test: the call is one line, it reports no counts, and a test
would need a frame-level assertion on forecasts that weather-only features can produce without
power. The mutation-testing diff review should judge whether that is enough.

## Docs to update

- `docs/roadmap/data-cleaning.md` — a short section saying where cleaning steps go
  (`nged_data.cleaning.POWER_CLEANING_STEPS`), what a logging function receives and returns, the
  row-local constraint, and that production falls back to raw power when cleaning fails, and loses
  only the stats when logging fails, while research fails.

- `packages/nged_data/README.md` — one line for the new module, and one for `coverage_from_power`
  beside the existing `time_series_coverage` entry.
- `TimeSeriesCoverage`'s docstring in `nged_data/storage.py`, which names CV eligibility as a
  caller of `time_series_coverage`; eligibility now goes through `coverage_from_power`.
- The docstrings of the five assets touched, which are operator docs in the Dagster UI.

## Verification commands

The `implement-issue` green-before-push set (`ruff check`, `ruff format`, `ty check`, `pytest`,
the `pymarkdown` scan), plus `pydoclint` and the docs-link checker as CI runs them, plus
`uv run mkdocs build --strict` for the roadmap edit.

## Risks and open questions

1. **Window-dependent cleaning functions.** Recommendation: state the row-local constraint in the
   `PowerCleaningStep` docstring now. When the first detector needs full history, it becomes a
   materialised asset of flagged rows (`power_data_problems`, computed over the whole table, which
   also feeds the delivery table), and the read-time step drops the flagged rows. That is a separate
   issue; this plan's interface does not block it.

2. **Should `metrics` score against cleaned actuals?** Probably yes for rows that describe a plant
   that no longer exists (the commissioning ramp), but `metrics` belongs to #958. Recommendation:
   leave it untouched here and raise it on #958 or a follow-up issue.
3. **Should `power_data_is_fresh` and the ingest de-duplication see cleaned power?** Recommendation:
   no — both are about what NGED delivered, not about what is fit to train on.

4. **Settled by the maintainer: frame-to-frame cleaning functions, each with an optional logging
   function.** The simplicity review's alternative — each rule a named `pl.Expr` predicate — would
   have made per-rule stats cheaper, but cannot change values or join to another frame. The
   anti-join per step is the price of keeping both.
5. **The stats on `effective_capacity` run one anti-join per step over the whole power table.**
   Cheap today with no steps; at V2's 2,500 series and several steps it becomes the most expensive
   part of that asset. Recommendation: measure when the first steps land, and drop
   `effective_capacity`'s stats (keeping `trained_cv_model`'s) if they are slow.

6. **Package home.** `nged_data.cleaning` unless the cleaning functions bring heavy dependencies;
   then a new `power_cleaning` package. Confirm with whoever is writing the functions.

## Review log

### Simplicity review (fresh Opus sub-agent)

Accepted:

- Split the cleaning from the counting, so `cv_power_forecasts` stops recounting the same window
  per chunk and reports no counts.
- Drop per-function attribution until the first cleaning function exists.
- Inline the live guard instead of a `_clean_or_degrade` helper.
- Drop the conditional `inherent-stability.md` edit, the `NamedTuple`, the research fail-fast test
  (it tests the absence of a `try`), and the separate live wiring test (the degrade test already
  proves the wiring).

Kept as the plan had it: the `coverage_from_power` split (the reviewer agreed it earns its place),
and both research wiring tests, because they exercise different code paths (a full-table scan
versus the coverage split).

Passed to the human reviewer: predicate rules instead of frame-to-frame functions (open question 4).

Rejected architectures, agreeing with the reviewer: cleaning inside `load_engineering_inputs` (the
live fallback would need raw and cleaned frames back, and two of the five readers do not use the
loader); cleaning in the feature engineer (misses eligibility and effective capacity, hides the
counts from Dagster); a materialised `power_data_problems` asset now (a new stored contract and
schedule wiring in `schedules.py`, which is out of bounds, bought before any detector needs it).

### Correctness review (fresh Opus sub-agent)

Accepted, all four defects:

- `eligible_time_series` would have raised on an absent power table instead of writing an empty
  population; the plan now keeps the `delta_table_exists` check.
- Two stale prose sites (the `TimeSeriesCoverage` docstring and the `nged_data` README) added to the
  docs list.
- Added a `trained_cv_model` wiring test (the reviewer showed the harness already exists) and a
  `degraded: False` assertion on the live happy path.
- The eligibility wiring test now names series 1, the only one eligible in the fixture's fold.

Also taken from the reviewer's caveats: the streaming engine on the counts query, and a live log
message naming the cleaning step (principle 16).

Confirmed by the reviewer, and so unchanged: the description of current code, that monkeypatching
the module attribute reaches in-process materialisations, that a raising `map_batches` surfaces at
the guard's collect, that `/` keys and mixed bool/int metadata work on Dagster 1.13, and that the
change is a no-op with an empty registry.

### Maintainer-directed redesign of the stats

The maintainer asked for stats expressive enough to tell zeros dropped from extreme values dropped,
and to summarise the extreme values themselves. Per-step attribution, which the simplicity review
had deferred, is back, and each cleaning function can be paired with an optional logging function
that receives `before`, `after`, and `dropped` for its step. The live path now has a second guard
so a failing logging function costs only the stats.
