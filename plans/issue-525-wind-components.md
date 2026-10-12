# Store wind components before interpolation — issue 525

The ECMWF ingest averages native wind components onto H3 cells, converts them to speed and
direction, then discards the components. The feature pipeline interpolates those bearings as
ordinary numbers: wind crossing North can become southerly wind. Storing vectors also prevents
future spatial and ensemble arithmetic on the stored columns from inheriting that problem.
Derived bearings remain circular; their model encoding and any circular lookback redesign are
separate work.

Measure the storage cost of three representations and the scalar-versus-vector speed gap first.
Subject to that evidence, store eastward `u` and northward `v`, interpolate them, and derive the
existing speed and meteorological direction features afterwards. Preserve model feature names,
weather lookbacks, and missing-input behavior. Rebuild the archive separately before cutover.

## Verdict, size, and scope

**Worth implementing, roughly as described.** The premise matches `origin/main` at `5ac2e16a`.
Issue 525 is open with no comments; no open PR or issue branch was found on 12 October 2026.
The audit in `docs/studies/forecasts/production-feature-faults.md` also recommends vector
interpolation. Its diagnostic reconstructs bearings along the shorter arc across North while
leaving speeds unchanged. That exposes direction errors but does not measure the speed changes
that follow from interpolating vectors.

**Complex; two plan reviews and three implementation reviews.** The triggers are:

- Stored data: replaces four `Nwp` fields and rebuilds the Delta archive.
- Production serving path: changes `_engineer_features`, used for live prediction and training.
- Degradation rule: must preserve null weather and single-run missing-control-member behavior.
- Multiple defensible designs: u/v alone or u/v plus scalar-mean speed; archive rewrite or download.
- Callers requiring a search: contracts, converter, storage, features, dashboard, fixtures, and
  scripts reading the production archive.

Run the simplicity review, then the correctness/testability review. Implementation gets the
cut-down, correctness, and mutation reviews; mutate changed production logic, not documentation.
Numerical results intended for publication additionally follow the `study` skill's scientific
reviews. This plan approves no code or archive operation yet.

Departures and boundaries:

- Derive direction too: `conf/model/xgboost.yaml` requests both bearings today. Keeping the current
  selected feature names isolates this change from the separate raw-u/v representation experiment.
- Derive polar columns before analysis-proxy selection and weather lookbacks, rather than merely
  adding speed alongside post-join windchill. Otherwise speed/direction lags lose their sources.
- Use a new archive location. `write_nwp` uses partition overwrite with schema overwrite; replacing
  columns in an existing table does not convert its other partitions. An in-place rolling rebuild
  would expose mixed physical schemas to readers.
- Do not claim experiments still cost zero: the production-feature study now exists. Preserve its
  input snapshots and treat its published numbers as historical evidence.
- Coordinate with 526 before implementation. Its planning worktree exists locally, although no
  PR is open. Keep its radiation and precipitation work separate; no circular class is needed for
  the four Cartesian columns. If components are rejected, its wind arm is required again.

## Solution

### Walk through

#### The happy path

First run the storage measurement described below on identical raw ECMWF runs and H3 weights.
Once the representation is chosen, `download_ecmwf_ens_data` continues supplying native wind
components to `convert_nwp_xarray_dataset_to_polars_dataframe`. The converter keeps their
area-weighted H3 means instead of calling `_calc_wind_speed` and `_calc_wind_direction` and
dropping the components. `Nwp.validate` checks the new signed fields. `write_nwp` rounds them
to `NWP_SIGNIFICAND_BITS`, sorts by `NWP_SORT_COLS`, and writes member-aligned Parquet as today.

`TabularFeatureEngineer.engineer` still attaches the containing H3 cell. `_engineer_features`
renames `init_time`, then `_upsample_nwp_to_half_hourly` interpolates the components within the
existing series/run/member groups. A small `_derive_wind_features` helper in
`tabular_feature_engineer.py` adds speed and direction to this processed frame using expressions
owned by `_parsed_features.py`. Speed is `sqrt(u² + v²)`; direction is
`(atan2(u, v) * 180/pi + 180) % 360`, the current meteorological convention.

Derivation precedes `select_analysis_proxy`, both power/NWP join modes, and weather lag/rolling
features, because all three need the derived columns at their source timestamps. Windchill then
uses the already-derived speed in its existing post-join stage. The selected output remains
`AllFeatures` with the existing polar feature names; components need not become model inputs.

#### The main branches

An empty destination archive is created by the first `write_nwp`; later runs replace only their
own partition. A rerun has the same idempotency as today. Existing readers continue using the old
archive throughout the build. After validation, switch the archive setting and matching code
together, keeping the previous archive and compatible code/model available for rollback.

Prefer u/v alone. If measured scalar-mean speed merits storage and the maintainer accepts it,
calculate each grid point's magnitude before H3 aggregation and retain its weighted scalar mean
as `wind_scalar_mean_speed_10m` and `wind_scalar_mean_speed_100m`. Keep the existing
`wind_speed_*` feature defined as vector magnitude; adding a stored scalar diagnostic must not
silently change model inputs. Its model use is a separate measured decision.

#### The main error-handling paths

Malformed signed components remain contract errors; research measurements fail on inconsistent
rows, failed writes, or incomplete paired coverage rather than reporting a favorable subset.
Keep the converter's existing per-variable null normalization and contributing-weight behavior.
Missing joined weather propagates to null polar features, including direction. Empty NWP and a
missing control member in single-run mode retain existing behavior; bulk weather-lag training
still rejects a missing control member. No new warning or fallback mechanism is introduced.

At exactly zero vector magnitude, preserve the existing formula
(`atan2(0, 0) + 180`, normally 180 degrees) and document that it carries no physical bearing.
Test opposing vectors with exact Cartesian inputs;
near-zero vectors use their computed bearing without an arbitrary calm threshold.

A failed archive partition leaves the old archive serving. Record rebuild failures and retry
them; do not switch settings until the replacement's required partition inventory is validated.

## What changes, file by file

### 1. Measure before choosing the schema

Add `studies/wind_storage/measure_wind_storage.py` and a short README. Its outputs live under
`data/studies/wind_storage/`, outside Git. Commit the sample manifest and print source/commit,
H3-grid identity, rows, members, horizons, schema, package versions, and write settings in the
report. Planned contrasts: polar versus u/v; u/v versus u/v plus scalar speed; weighted scalar
mean versus vector magnitude at both heights.

Use 24 full daily runs, one near the middle and one near the end of each month in a complete
available archive year. Freeze dates before measuring; report unavailable dates, and choose
replacement dates by a documented date-only rule before inspecting values. Use every member,
lead time, and production H3 cell of each run. This is a sample, not a new archive-wide census.

Run three minimal experimental schema/converter variants in isolated worktrees: unchanged polar,
u/v, and u/v plus scalar means. Each validates its own complete `Nwp` contract and invokes the
real `delta_store.nwp.write_nwp` into a fresh local table. Run the same committed measurement
driver in each variant; retain variant diffs and SHAs with the report. This avoids introducing
a configurable production writer just for a benchmark. Do not duplicate rounding, sorting,
compression, row-group sizing, or target file-size logic in the driver. The issue's named
`NWP_WRITER_PROPERTIES` constant is stale: `_writer_properties` is now built per write.

Use Parquet metadata to sum compressed column-chunk bytes and row counts across all row groups
and active files; separately report file bytes and Delta-log overhead. Report per-column
bytes/row, paired per-run totals, seasonal spread, and row-weighted sample totals for all three
representations. Label sample and archive-wide numbers distinctly.

For the same raw grid and weights, report weighted scalar-minus-vector speed by height, season,
and H3 cell: absolute quantiles, maxima, and relative differences with calm denominators reported
separately. Use a common valid-u/v footprint for this comparison; also report coverage lost to
missing pairs. Do not infer scalar means from stored polar columns. A scalar mean alone does
not recover mean turbine power through a nonlinear power curve. Treat size and speed-gap results
as descriptive measurements, not a leaderboard or turbine-yield result.

Present the report before retaining a permanent schema change. Recommend u/v alone unless the
gap and incremental bytes justify the third column; no threshold has been agreed in issue 525.
If the storage cost is rejected, retain polar storage and re-scope 526's wind work explicitly.

### 2. Contracts and ingest

- `packages/contracts/src/contracts/weather_schemas.py`: replace stored speed/direction with
  signed Float32 `wind_u_10m`, `wind_v_10m`, `wind_u_100m`, `wind_v_100m`, in metres per second,
  with bounds -200 to 200 and current non-nullability. Preserve the old magnitude corrupt-feed
  ceiling by validating `u² + v² <= 200²` at each height. Describe eastward/northward signs.
- Keep `WeatherFeature` as the existing model-weather vocabulary, including derived polar names;
  document that it describes model inputs rather than stored columns. `AllFeatures` already
  declares those fields. Replace the literal-equals-Nwp test with an explicit mapping invariant:
  substitute the four component fields for the four polar feature names. No second feature
  vocabulary or raw-component model experiment is needed.
- `packages/dynamical_data/src/dynamical_data/ecmwf_ens/convert_to_polars.py`: retain components;
  remove ingest-only polar helpers after moving their expressions into feature engineering.
  If scalar columns are accepted, derive point magnitudes after NaN normalization but before
  weighted aggregation, requiring both components at that point.
- `packages/delta_store/src/delta_store/nwp.py`: keep the writer algorithm unchanged. Correct
  documentation claiming validated inputs make column dropping impossible: this replacement
  is a deliberate breaking schema change requiring a fresh destination.
- `packages/dynamical_data/src/dynamical_data/ecmwf_ens/download.py`: verify existing native names
  and dataset contents; no download-variable change is needed for u/v alone.

### 3. Feature engineering and consumers

- `_parsed_features.py`: own the four fixed wind expressions next to windchill. Keep the existing
  polar names in `WeatherFeature`, so parsing, weather detection, lag/rolling types, and
  `requires_weather_data` keep working unchanged. `StaticFeature` stays windchill alone.
  Do not add a second registry or a general dependency resolver.
- `tabular_feature_engineer.py`: derive all four polar columns once on `processed_nwp`, immediately
  after interpolation. Keep those expressions out of `_apply_post_join_features`' single
  windchill `with_columns` call; siblings cannot depend on aliases made in that same call.
  Remove raw components from the final selected frame unless a future feature explicitly asks
  for them. Preserve direct, lag-only, rolling-only, windchill-only, and power-only requests.
- `_nwp.py`: existing generic interpolation should work once the stored fields are Cartesian.
  Update its account of wind handling without changing radiation, precipitation, or time grids.
- `packages/dashboard/src/dashboard/forecast_chart.py`: replace the four raw polar plot choices
  with signed component choices and labels, preserving the continuous-variable coverage invariant.
  Model feature plots still show derived polar features. Extra raw-panel polar views are outside
  this change.
- `tests/_nwp_test_data.py`, its validator test, converter fixtures, Delta/S3 fixtures,
  `tests/test_assets.py`, and affected feature/dashboard tests: migrate raw NWP records; retain
  polar expectations for engineered features and model serialization.
- `studies/nwp_forecast_comparison/fetch_ens_forecast_horizons.py`: derive polar values at native
  steps before its existing output projection; it directly scans the production archive.
  Search for other readers by archive path and `Nwp.scan_delta`, not wind names alone. Update
  actual production-table readers only; Open-Meteo/reanalysis frames and frozen study caches
  use independent schemas and should retain their names and provenance.

### 4. Archive cutover and measurement of forecasts

For u/v alone, add `scripts/maintenance/rewrite_nwp_wind_components.py`: read the old Delta archive
partition by partition through a plain Polars scan pinned to a source version, reconstruct
`u = -speed * sin(direction)` and `v = -speed * cos(direction)`, remove polar fields, validate
against the new `Nwp`, and write through `write_nwp` to a new configured destination. Reject
identical source/destination paths. Record a source-version and completed-partition manifest;
restart by safely rewriting partitions. The measured reconstruction error is below stored
precision, so this avoids a full download and upstream revisions. Validate it on the seasonal
raw-input sample before choosing it. If scalar means are accepted, use the existing `ecmwf_ens`
backfill path to re-download native inputs into the fresh destination instead. Estimate elapsed
time and volume before scheduling either path; do not promise an overnight finish.

Inventory required partitions first. Validate new key coverage, dtypes, members, steps, H3 cells,
null patterns, and magnitude ranges per partition. Compare unchanged weather fields within the
writer's precision. Record source revisions and exceptions: re-downloading revised upstream
data need not reproduce the original archive bit for bit. Switch settings only with the matching
feature code and compatible trained model; keep old artifacts for rollback and rebuild derived
feature caches. Readers must never scan old and new schemas as one table.

Preserve the pre-change commit and old table version/path as the reference. Register the u/v arm
through the existing experiment workflow with the same selected feature names, training recipe,
folds, seeds, evaluation population, and scenario settings. Score on the same machinery and rows,
with horizon/type slices including the roadmap's 3–10 day band. Compare interpolation ordering,
not a simultaneous raw-component feature experiment. Later scoring is permitted if required
baselines/metrics are unavailable; record the pending arm and immutable reference explicitly.

## Tests and assertions

Each new regression below must fail on current main; existing fixture migrations are maintenance.

- `packages/contracts/tests/test_weather_schemas.py`: a complete signed-component frame validates
  without stored polar fields; legacy-only input is rejected; a component beyond bounds and a
  pair whose magnitude exceeds 200 are rejected. Update the feature/storage mapping test and
  keep the continuous/categorical variable-partition test.
- `packages/dynamical_data/tests/test_convert_to_polars.py`: asymmetric weighted raw vectors
  emerge as their component means and no polar fields. In the conditional scalar variant, two
  opposing vectors have zero vector speed and nonzero scalar mean; missing-pair normalization
  uses only paired contributions. Preserve current per-variable null tests.
- `packages/delta_store/tests/test_nwp.py`: write/read negative and zero components through
  `write_nwp`, asserting contract fields, significand rounding, and two-partition readability.
  Rematerializing one partition must retain the other and introduce no duplicate keys.
- `packages/ml_core/tests/test_features.py`: equal-speed bearings 350 and 10 degrees yield a
  midpoint at North and reduced vector magnitude, not 180 degrees and unchanged speed. At both
  heights verify cardinal signs, native endpoints, exact calm/opposing vectors, and nulls.
  Feed raw Cartesian rows to the pipeline so pre-change code fails on the actual new schema.
- Feature/parser and cross-mode tests: direct polar, speed/direction lag-only and rolling-only,
  and windchill-only selections work without explicitly requesting their dependencies. Assert
  numeric midpoint speeds feed windchill and lookbacks, and matched bulk/single-run rows agree.
  Mixed runs/members/series must not contaminate each other's component interpolation.
- Add a regression using an empty new-schema NWP frame in single-run mode: requested polar
  features are null on retained forecast rows. Re-run existing missing-control-member warning
  and power-only tests; add a component-schema case that current main cannot process.
- `packages/dashboard/tests/test_forecast_chart.py`: plot registry equals the new continuous
  stored variables; negative components plot with signed units. Test the archive-reader study's
  native polar projection on synthetic cardinal vectors if extracting that projection is needed.
- Archive-rewrite tests under `tests/`: old polar cardinal vectors become correctly signed
  components in a different destination; all non-wind columns and row keys survive; a rerun
  has no duplicates; source data stays untouched; a two-partition result reads under the new
  contract. These test the new migration tool, whose absence is the failure on current main.
- Benchmark checks use tiny synthetic frames to verify equal key coverage, column-chunk byte
  accounting, and the distinction between scalar and vector means. They are method checks for
  new measurement code, not regressions claimed to fail before that code exists. Confirm real
  three-variant runs actually call `write_nwp` and use identical complete-run manifests.

## Design-philosophy check

Principles 12 and 15 require measuring the storage trade and moving reversible transformations
to feature engineering. Principle 8 requires an attributable comparison; principle 10 rules out
an archive left half converted. No permanent writer abstraction or model-selection experiment
is needed. The signed-component magnitude check preserves the existing malformed-input rule.

This path serves production: preserve missing-weather nulls and existing degradation reporting,
without adding a new exception or inventing a new degradation reason. R&D measurements fail fast
on invalid comparisons. No asset check is added; existing checks remain WARN, `blocking=False`,
and protected so their bodies cannot fail the ingest. Run their regressions after fixture changes.
This supports H1/T1.2 through preserved degradation and H4/T4.1 through measured storage cost;
it does not claim either hypothesis has been demonstrated by this change.

## Documentation and ship-time triage

Update `docs/architecture/nwp-variable-conventions.md` with the chosen stored representation,
signs, calm/null convention, scalar-mean decision, measured storage scope, and transformation
order. Preserve historical measurements with their dates/sample labels. Keep the old
`wind-is-stored-as-speed-and-direction-and-why` anchor as an explicit compatibility anchor if
the heading changes, because code comments and the issue already link to it.

Update relevant ECMWF API/package README material, `CLAUDE.md` feature/storage descriptions,
and dashboard docs naming raw plot choices. Publish the reproducible measurement report under
`docs/studies/` with the study reviews if it warrants a separate page; link from the conventions
page. Record production-feature-audit recommendations as addressed without rewriting its cached
results as though they used the new code.

Update the wind-storage item and measurement sequence in `docs/roadmap/xgboost-improvements.md`.
Promote surviving decisions to architecture docs and remove completed implementation details,
while retaining pending leaderboard scoring as pending. Do not mark the whole XGBoost roadmap
complete. On shipping, transfer this plan or its summary to the PR body and delete the plan file.

## Verification commands

For implementation: `uv run ruff check .`; `uv run ruff format .`;
`uv run --all-packages ty check`; `uv run pytest --run-studies`;
`uv run pytest --run-network -m network` for the convention-sensitive converter;
`uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md`;
`uv run mkdocs build --strict`, then inspect rendered HTML and changed anchors.
Run the frozen three-arm storage measurement, inspect Parquet metadata and the report, and
validate the completed replacement archive before cutover. Plot changed wind features against
observed power to check signs and chronology. A green unit suite does not replace those checks.

For this planning-only PR: Markdown lint of this plan and `git diff --check`; no production
tests or data rebuild are needed before reviewing the design.

## Risks and open questions

- Is the measured storage premium acceptable? Recommend u/v alone for structural correctness,
  subject to the larger paired measurement; do not treat the old two-run 6% figure as final.
- Does scalar-mean speed earn two extra stored columns? Recommend measuring first, then having
  the maintainer decide before schema retention or re-ingest; it is not mean turbine power.
- Are 24 runs sufficient to establish seasonal compression cost? Recommend the fixed sample
  above, expanding by a date-only rule if between-season spread makes the decision ambiguous.
- When should the archive/code switch happen, and which model must be retrained? Recommend
  staged rebuild and a coordinated switch after inventory validation, with old artifacts retained.
- Does 526 land first? Recommend agreeing its scope before either implementation; components
  remove wind wrap work, while period-ending radiation remains independently necessary.
- Does a measurement fail because upstream data changed? Recommend recording the mismatch and
  re-establishing paired inputs, never interpreting different row populations as storage effects.

## Review record

Simplicity review accepted: preserve the existing `WeatherFeature` model vocabulary instead of
introducing derived static categories; prefer a compute-only fresh-table rewrite for u/v alone;
preserve the existing calm-direction formula; limit the structural wrap-safety claim to stored
components. Scalar storage remains conditional on the requested measurement.

Rejected: extract a private configurable writer and call it from the study. Study imports must
use public package APIs, and the issue explicitly asks for `write_nwp` itself. The three small
experimental variants keep the production write path literal without widening its public API.
Revisit this cost only if the benchmark implementation shows the variants are harder to maintain.
The correctness/testability review is pending.
