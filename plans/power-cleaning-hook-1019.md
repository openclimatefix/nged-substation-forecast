# Plan: a power-cleaning hook in the Dagster pipeline (#1019)

**The problem.** Cleaning algorithms for NGED's power telemetry are being written now, and the
pipeline has no place to put them. Every asset that reads observed power — training, CV prediction,
live forecasting, eligibility, and effective capacity — scans the raw `power_time_series` Delta
table directly, so a cleaning function today would have to be wired into five places by hand, and
nothing would report what it removed.

**The planned solution.** Add two functions in a new module `nged_data.cleaning`.
`clean_power_time_series` takes a lazy `PowerTimeSeries` frame, applies an ordered tuple of plain
cleaning functions (empty today, so the frame passes through unchanged), and returns the cleaned
lazy frame. `power_cleaning_stats` compares the raw and cleaned frames and returns a small
dictionary of counts: rows and series in, rows and series out. Five assets that read power call the
first at read time, and four of them add the counts to their own Dagster output metadata. The raw
table is never rewritten and no new table is created. The production caller, `live_forecasts`, wraps
the call so that a failing cleaning function degrades the slot to raw power and reports to Sentry;
the research callers let the exception propagate and fail the run.

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
  (`cv_power_forecasts`, once per `init_time` chunk) does not pay for the counts each time.

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
- `clean_power_time_series(power, cleaning_functions=None) -> pt.LazyFrame[PowerTimeSeries]` —
  applies each function in order and stays lazy. `None` resolves to `POWER_CLEANING_FUNCTIONS` at
  call time, so a test can monkeypatch the module attribute.
- `power_cleaning_stats(raw, cleaned) -> dict[str, int]` — one `pl.collect_all(...,
  engine="streaming")` over `select(pl.len(), pl.col("time_series_id").n_unique())` on each frame.
  The query is an aggregate, not a materialisation, so principle 11 holds. Keys:
  `power_cleaning/n_rows_in`, `power_cleaning/n_rows_out`, `power_cleaning/n_time_series_in`, and
  `power_cleaning/n_time_series_out`. Flat string keys go straight into
  `context.add_output_metadata`, and the prefix keeps them clear of each asset's existing keys
  (Dagster raises on a duplicate key).

Rows dropped per cleaning function is deliberately left out until the first function exists: with
an empty registry there is nothing to attribute, and attribution is the largest piece of logic the
module would otherwise carry. It can be added then, together with the delivery table of data
problems, without changing the `PowerCleaningFunction` signature — an anti-join of each stage
against the one before yields the rows each function removed.

`nged_data` is the home because the step cleans NGED's telemetry and the package already depends on
`contracts`. A new package is only worth it if the cleaning functions grow dependencies (`scipy`, a
solar-geometry library) that `nged_data` should not carry; that is cheap to do later.

### `src/nged_substation_forecast/defs/cv_assets.py` (research — fail fast)

Out of bounds and untouched: `metrics` and its helpers (#958).

- `trained_cv_model`: call `clean_power_time_series` on the frame `load_engineering_inputs`
  returns, and merge `power_cleaning_stats(raw, cleaned)` into the existing `add_output_metadata`
  call.
- `cv_power_forecasts`: call `clean_power_time_series` inside the `init_time` chunk loop and report
  no counts. The power window is the validation window in every chunk, and `trained_cv_model`
  already reports the training window's counts.
- `effective_capacity`: clean the full-table scan before `compute_effective_capacity`; merge the
  counts.
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

- Right after `load_engineering_inputs`, an inline `try` block, copying the shape of the existing
  control-member guard (the same `BaseException` catch, re-raising `KeyboardInterrupt`,
  `SystemExit`, and `DagsterExecutionInterruptedError`). No shared helper for the two guards: two
  short inline blocks read more clearly than an abstraction with one reuse.
    - Inside the guard: call `clean_power_time_series`, collect the cleaned frame and re-wrap it as
      lazy, then call `power_cleaning_stats`. The collect means a cleaning function that only fails
      at collect time fails *inside* the guard rather than later in `predict`. The live power
      window is 15 days of the trained series, a few MB at V2's 2,500 series, so this one collect is
      a deliberate exception to principle 11.
    - On failure: log with `context.log.exception("Power cleaning failed; forecasting this slot
      from uncleaned power")`, so the log names the cleaning step rather than only the exception
      type. The guard also covers the raw Delta read the collect triggers, so an object-store fault
      during that read is reported the same way, which the log message has to allow for. Then call
      `report_asset_degradation(asset_name="live_forecasts", exc=exc)`, carry on with the raw
      frame, and set `power_cleaning/degraded: True`.
- On success the metadata carries `power_cleaning/degraded: False` and the counts, merged into the
  existing `add_output_metadata` call.
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
- `test_cleaning_functions_run_in_order` — two injected functions where the second only has an
  effect if the first ran before it; asserts the result.
- `test_stats_count_rows_and_series_removed` — an injected function dropping two rows of one series
  and every row of another; asserts all four counts.

In `packages/nged_data/tests/test_storage.py`:

- The existing `time_series_coverage` tests keep passing unchanged, which is the evidence the split
  is a pure refactor. No new test.

In `tests/test_cv_assets.py`:

- `test_effective_capacity_reads_cleaned_power` — monkeypatch
  `nged_data.cleaning.POWER_CLEANING_FUNCTIONS` with a function dropping one series; assert the
  `effective_capacity` table lacks that series and the materialisation metadata carries the drop
  count. Fails on `main`: nothing applies the registry.
- `test_eligible_time_series_reads_cleaned_power` — same shape for eligibility, dropping series 1:
  the fixture makes only series 1 eligible for `FOLD_ID`, so dropping any other series would leave
  eligibility unchanged and the population assertion would pass on `main`.

In `tests/test_trained_cv_model.py`:

- `test_trained_cv_model_reads_cleaned_power` — the existing harness already materialises
  `trained_cv_model` with two eligible series. Drop one through the registry and assert the
  metadata's `power_cleaning/n_time_series_out` is one fewer than `n_time_series_in`. Fails if the
  cleaning call is deleted from `trained_cv_model`.

In `tests/test_live_forecasts.py`:

- `test_a_failing_cleaning_function_degrades_the_slot_instead_of_failing_it` — a raising function
  (raising at collect time, via a `map_batches` that raises, to prove the guard's collect is what
  catches it); assert forecasts are written, `report_asset_degradation` was called with
  `asset_name="live_forecasts"`, and the metadata has `power_cleaning/degraded: True`. Modelled on
  `test_a_failing_control_member_probe_degrades_the_slot_instead_of_failing_it`. The test also
  proves the live wiring: if `live_forecasts` never called the cleaning, nothing would raise and
  `report_asset_degradation` would not be called.

- The existing happy-path live test gains one assertion: the metadata carries
  `power_cleaning/degraded: False` and equal in/out row counts.

`cv_power_forecasts` gets no wiring test: the call is one line, it reports no counts, and a test
would need a frame-level assertion on forecasts that weather-only features can produce without
power. The mutation-testing diff review should judge whether that is enough.

## Docs to update

- `docs/roadmap/data-cleaning.md` — a short section saying where cleaning functions go
  (`nged_data.cleaning.POWER_CLEANING_FUNCTIONS`), the row-local constraint, and that production
  falls back to raw power on failure while research fails.
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
   `PowerCleaningFunction` docstring now. When the first detector needs full history, it becomes a
   materialised asset of flagged rows (`power_data_problems`, computed over the whole table, which
   also feeds the delivery table), and the read-time step drops the flagged rows. That is a separate
   issue; this plan's interface does not block it.
2. **Should `metrics` score against cleaned actuals?** Probably yes for rows that describe a plant
   that no longer exists (the commissioning ramp), but `metrics` belongs to #958. Recommendation:
   leave it untouched here and raise it on #958 or a follow-up issue.
3. **Should `power_data_is_fresh` and the ingest de-duplication see cleaned power?** Recommendation:
   no — both are about what NGED delivered, not about what is fit to train on.
4. **A cleaning function as a frame-to-frame function, or as a named "bad row" predicate?** The
   simplicity review proposed `POWER_CLEANING_RULES: dict[str, pl.Expr]`, each expression True for
   a row to drop (`.over("time_series_id")` allowed). Cleaning becomes one
   `filter(~pl.any_horizontal(...))`; per-rule counts and the delivery table of data problems come
   from one aggregate pass with the rule name attached. What it gives up: a rule cannot rewrite
   values or join to another frame (the neighbour-median ramp detector in
   `docs/roadmap/data-cleaning.md`). Recommendation: keep frame-to-frame functions, the more general
   interface and the one the issue describes, unless whoever writes the cleaning functions finds
   predicates fit everything they plan; this decides what they write, so it is their call.
5. **Package home.** `nged_data.cleaning` unless the cleaning functions bring heavy dependencies;
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
