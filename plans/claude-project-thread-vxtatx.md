# Plan: serve the step-5 demand network for the 16 Primaries (#1706)

**The problem.** The live service forecasts every series with the naive forecaster, while research
has a neural network for the 16 Primaries that beats the manual heuristic and climatology at every
lead band. The network is step 5 of research experiment 677 at monthly refits. Training stays in
`nged-ml-research`, so production needs inference only, from a plain directory of weights and
constants that research experiment 705 exports. The goal is a daily forecast for the Primaries in
the live service before the Christmas 2026 break.

**The planned solution.** A new workspace package, `substation_demand_forecaster`, rebuilds the
network's 60 input columns from the `nwp` and `cleaned_power_time_series` Delta tables, runs a
forward pass re-implemented in numpy for each of the three seeds, and reads the 13 NGED delivery
quantile levels out of each seed's Gaussian mixture. A new daily Dagster asset,
`live_primary_substation_forecasts`, runs the package at the 12:00 UTC origin on the day's 00Z ECMWF
ENS run and writes one row per Primary and half-hour to `power_forecasts`, with the 13 quantile
columns that a separate issue adds to `PowerForecast` (see "Dependency on the contract change"). The
bundle is copied into the image beside the existing champion model. The 6-hourly `live_forecasts`
asset keeps running, as the fallback.

## Verdict, size, and departures

**Verdict: worth implementing, as the issue describes.** The maintainer chose the step-5 network
over the full Candidate on 2026-11-18. Experiment 677 measured a normalised MAE of the median of
7.39 and a CRPS of 4.19 points of capacity for step 5 at monthly refits, against 7.48 and 4.26 for
the Candidate. Step 5 also needs no daily K1 run, no stored lag-free table, and no neighbour list.

**Size: complex.** The five triggers:

- **What gets stored:** fires. New rows in `power_forecasts` under a new `experiment_name`, and a
  new directory in the image.
- **The production serving path:** fires. A new live asset.
- **A degradation rule:** fires. The asset needs rules for a missing ENS run, missing members,
  missing history runs, and missing lags.
- **More than one defensible design:** fires. Whether to subclass `BaseForecaster`, and how to get
  the solar position, both have alternatives (see "Risks and open questions").
- **Callers I could not name without searching:** does not fire. `select_analysis_proxy` and
  `_upsample_nwp_to_half_hourly` are called unchanged, and every other function is new.

**Reviews: both plan reviews (simplicity, then correctness), and all three diff reviews.**

**Dependency on the contract change.** The maintainer decided on 2026-11-18 that `PowerForecast`
itself changes from one `power_fcst` per `ensemble_member` to 13 `Float32` quantile columns at the
levels agreed with NGED for the delivery tables (`DELIVERY_QUANTILES`: 0.01, 0.02, 0.05, 0.10, 0.20,
0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 0.99), keyed by
(`time_series_id`, `power_fcst_init_time`, `valid_time`). That change touches every forecaster, the
metrics, and the `power_forecasts` table, so it is its own issue and pull request. This plan only
needs it before step 2 of "Implementation order" below; the package in step 1 does not depend on
it. The design follows the research page "Delivering uncertainty to NGED's power-flow studies"
(nged-ml-research, 2026-11-10): one quantile forecast per series and half-hour, with no member
structure.

## Solution

### Walk through

#### The happy path

1. **A schedule fires `live_primary_substation_forecasts` at 12:00 UTC each day.** The asset reuses
   `ecmwf_ens_partitions` (`defs/assets.py`), so its partition maps one-to-one to the day's ENS
   run. The partition's date gives `power_fcst_init_time` = that date at 12:00 UTC and the native
   origin = 09:00 UTC, the time the network was trained to treat the 00Z run as available. The
   12:00 slot is later than `ecmwf_ens_schedule` (10:30 UTC), so on a normal day the run is already
   on disk. The two schedules couple through data at rest, never through run status (design
   principle 14).
2. **The asset loads the bundle** with `substation_demand_forecaster.bundle.load_bundle(path)` from
   the new `Settings.primary_substation_bundle_path`. `load_bundle` checks that the manifest's
   `bundle_format_version` equals `BUNDLE_FORMAT_VERSION: Final = "2"` (the manifest stores it as a
   string), reads each seed's weights from `components/step5/weights/seed<N>.npz` with `numpy.load`,
   and returns a frozen `Bundle` dataclass. The dataclass holds the config, the standardisers, the
   column order, the offset knots, the model release number from the manifest's new `model_release`
   field (0 until a release is cut), and the 16 Primaries from `primaries.json` (identifier, H3
   cell, latitude, longitude, `effective_capacity_mw`, `capacity_mw`, embedding index). The
   Primaries and their capacities come from the bundle, never from today's `TimeSeriesMetadata`
   table, for the same reason `live_forecasts` uses its frozen metadata copy.
3. **The asset checks that today's 00Z run is present** with the existing
   `_available_nwp_init_times`. If it is not, the asset raises `RetryRequested` every 30 minutes,
   up to 6 times, so until 15:00 UTC. That is 30 minutes after the last retry of `ecmwf_ens`
   (8 retries from 10:30 UTC), which can itself take about 10 minutes to finish.
4. **The asset reads the NWP it needs in one scan** of the `nwp` Delta table, pruned to the 16 H3
   cells and to the runs initialised from 13 days before the native origin to today's 00Z run. The
   13 days are set by the cold-snap column of the thermal-lag group. Research keeps 12 days of
   history in front of the origin (`LOOKBACK_STEPS = 12 * 48` in experiment 437's
   `e00437_features.py`, a 10-day cold-snap cap plus a 24-hour mean window). The first day of run
   D−13 is the earliest that reaches 12 days before the origin. The scan also covers the
   expected-weather rows, which reach back 192 hours.
5. **`substation_demand_forecaster.ens_summary.summarise_runs(nwp, primaries)` builds the ENS
   summary per run.** It first joins each H3 cell to the Primaries' `time_series_id`, drops
   `h3_index`, and renames `init_time` to `nwp_init_time`, because `_upsample_nwp_to_half_hourly`
   raises on any other column. It then calls `_upsample_nwp_to_half_hourly` on each member
   separately, with the Primaries' coordinates as `solar_coordinates` and
   `strict_radiation_schedule=False`, so that one run with an unusual radiation lead degrades
   instead of failing the asset. It derives `wind_speed_10m = sqrt(u² + v²)` per member, and only
   then takes the member mean and the population standard deviation (`ddof=0`). Experiment 705 found
   that averaging first is wrong by up to 4.24 m/s in wind speed. Precipitation gets `log1p` of the
   value clipped at zero, applied to both the mean and the standard deviation, as the bundle's
   `ens_summary_recipe` states.
6. **`substation_demand_forecaster.features.build_input_matrix(...)` builds the 60 columns** for
   each (Primary, valid half-hour) from the native origin plus 0.5 hours to plus 339 hours, in the
   order of `components/step5/columns.json`:
    - the 14 ENS means and standard deviations of today's run;
    - the 21 calendar and sun columns, computed from the valid time. The sun columns are the
      apparent zenith cosine and the azimuth sine and cosine at the half-hour's stamp minus 15
      minutes, computed with pvlib's SPA on numpy arrays (`pvlib.spa.solar_position_numpy`), to
      match research's `pvlib.solarposition.get_solarposition`. The calendar columns are the
      local-hour sines and cosines at 24 and 12 hours in Europe/London, a one-hot day of week, the
      COVID flag, three Christmas columns, and three bank-holiday columns from the `holidays`
      package's England calendar, as research experiment 87 computes them;
    - the 2 lead columns (lead over 336 hours, and its square root);
    - the 14 thermal-lag and solar-gain ("B1") columns, built over the analysis-proxy history
      before the native origin and today's run after it, with the two previous-day CAMS columns set
      to zero (raw 0, as arm `drop` of experiment 704 does);
    - the 8 lag columns: day and week lag value, missing flag, and age code, plus the two grid
      indices. The lag times follow the rule in the bundle's config (`target − 48 × ceil((target −
      (origin − 48)) / 48)` in half-hour steps, and the same with 336). The lags read cleaned power
      from `nged_data.storage.scan_cleaned_power`, divided by `effective_capacity_mw`;
    - the knot position (months since 2019-09 plus the fraction of the month, clamped to the knot
      range).

   The analysis proxy comes from `substation_demand_forecaster.features.analysis_proxy`. It filters
   the ENS summary means to leads strictly above 9 hours and strictly below 33.5 hours, labels the
   frame `ensemble_member = NWP_ANALYSIS_MEMBER`, and calls `select_analysis_proxy` unchanged;
   experiment 705 showed that this window reproduces the research history exactly. The
   expected-weather rows come from `build_expected_rows(...)`. For each half-hour g in (native
   origin − 192 h, native origin − 24 h], the row is the ENS summary row of the freshest run whose
   native 09:00 origin is at or before g: its 7 weather means (precipitation `log1p`) and 21
   date-time columns recomputed from g, with the 14 B1 slots set to 0. Standardisation follows the
   rules in `standardisers.json`: float32 arithmetic, clip at ±5, and NaN to 0 for the 37 lag-free
   columns; float64 and a clip for the lag values; non-finite to 0 and no clip for the B1 columns;
   float16 rounding of the expected-weather rows.
7. **`substation_demand_forecaster.network.forecast(bundle, matrix, expected_rows, levels)`** runs
   each seed's forward pass in numpy, with `scipy.special.erf` for the exact GELU and the mixture's
   cumulative distribution. The forward pass is the functional `forward` and `mlp` of
   `reference_check.py`, ported line by line and applying the stored weights by their `state_dict`
   names, so the port diffs against the reference and imports nothing from research. `forecast`
   reads the quantiles at `levels` (the 13 delivery levels in production, and the 11 research levels
   in the golden tests) from each seed's mixture by 40 bisection steps over ±40 standard deviations,
   sorts each seed's quantiles, converts them to a fraction of capacity with the per-Primary target
   mean and standard deviation, averages the three seeds level by level, and sorts within each row
   again, as research does.
8. **The asset relabels the origin to 12:00 UTC, multiplies by `capacity_mw`, and keeps the 671
   valid times strictly after 12:00 UTC, the last being 12:00 UTC 14 days later.** Research's own
   contract rows start at 12:00 UTC itself; `PowerForecast.valid_time` must be strictly after
   `power_fcst_init_time`, so that row is dropped. The asset then clamps every quantile of the eight
   MVA Primaries at zero, as the backtest files do. It builds a `PowerForecast` frame with the 13
   quantile columns, `power_fcst_model_name = "primary_substation_demand_forecaster"`,
   `experiment_name = "primary_substation_demand_forecaster"`, `power_fcst_model_version` from the
   bundle's `model_release`, `nwp_init_time` = today's 00Z run, and `fold_id = "live"`.
9. **The asset writes with `write_power_forecasts`**, replacing exactly the partition
   (`experiment_name`, `"live"`) and this `power_fcst_init_time`, so a rerun never duplicates rows.
   No Sentry check-in monitor is added: a failed run reaches Sentry through
   `sentry_capture_failure`, a missing run through the degradation report, and a dead daemon through
   the existing `live-forecasts` monitor.

#### The main branches off the happy path

- **A backfill of a past partition** reads the NWP runs initialised up to that day's 00Z run, so it
  sees the same weather the live run did. Its lags can differ, because telemetry that arrived late
  is now on disk; the runbook says so. A backfill whose 00Z run is more than 36 hours old does not
  retry, which mirrors `_is_too_old_to_retry` in `ecmwf_ens`. The asset has no `availability_mode`
  config.
- **A Primary whose day or week lag is missing** (no cleaned reading at the lag time, or a stale
  telemetry feed) gets the missing encoding the network was trained with: value 0, flag 1, and the
  largest age code. The network trained with 30% of origin days lag-masked, so this degrades the
  forecast towards the lag-free forecast rather than failing.
- **A missing daily run in the 13-day history.** The windowed proxy then has a gap of about a day,
  because consecutive runs' first-day windows only touch. The B1 columns are trailing aggregates
  that research computes over the finite values only (`trailing_mean`, `np.fmin`, and `np.fmax`
  skip NaN, and the changes use `nan_to_num`), and the B1 standardiser maps a non-finite result to
  0. Production follows the same rule: aggregate over the finite history values and map a
  non-finite aggregate to 0. The gap is not filled from another run, so production matches research
  in this case too. The asset reports the missing history run as a degradation, naming the run.
- **A lag time whose expected-weather row has no covering run** (every run whose native origin
  is at or before g is missing): the row falls back to the next-freshest run that covers g, by
  smallest lead. If no run covers g, every column of the row is 0, which is the value research's
  float16 rounding of NaN gives. The asset reports the degradation.

#### The main error-handling paths

- **Today's 00Z run is absent at 15:00 UTC**: the asset reports a degradation to Sentry tagged
  `degraded_asset=live_primary_substation_forecasts`, naming the run, and finishes without writing.
  It does not forecast from yesterday's run, because every lead would be 24 hours longer than the
  network trained on. NGED keeps the fallback forecast from `live_forecasts`, which runs at every
  slot.
- **Today's run has fewer than 51 members in a cell**: the summary is the mean and standard
  deviation over the members present. `nwp_run_is_complete` already warns about the run and names
  the absent members, so the asset adds no report of its own.
- **The bundle is missing or has an unknown format version**: `load_bundle` raises. That is our own
  deployment fault, not the outside world, so raising is right under the inherent-stability page,
  and the Sentry failure hook reports it. The SHA-256 of every file is checked once, at image build
  time, because the image cannot change after it is built.
- **The output fails `PowerForecast.validate`** (a duplicated key, a NaN quantile, a crossing):
  raises, because that is our bug.

### What changes, file by file

- **Research prerequisite (in `nged-ml-research`):** experiment 705's exporter also writes each
  seed's weights as `seed<N>.npz` and a `model_release` field in `manifest.json`, and the bundle
  format becomes version `"2"`. The `.pt` files can stay for research's own `reference_check.py`.
- **No `torch` dependency.** The forward pass is a few linear layers, GELU, embedding lookups,
  `log_softmax`, `erf`, and bisection, all of which numpy and scipy do.
- **Root `pyproject.toml` and `uv.lock`**: `substation_demand_forecaster` as a dependency of the
  root project with a `[tool.uv.sources]` workspace entry, because the Dockerfile's `uv sync
  --no-dev --no-editable` installs only the root project's dependencies. `holidays` and `pvlib`
  become runtime dependencies of `substation_demand_forecaster`; `pvlib` stays in the dev group too,
  for `test_radiation.py`.
- **`packages/substation_demand_forecaster/`** (new workspace member):
    - `bundle.py`: `Bundle` (frozen dataclass), `load_bundle(path)`, and `BUNDLE_FORMAT_VERSION`.
    - `ens_summary.py`: `summarise_runs(nwp, primaries)`.
    - `features.py`: `analysis_proxy`, `calendar_columns`, `sun_columns`, `b1_columns`,
      `lag_columns`, `knot_position`, `build_expected_rows`, and `build_input_matrix`.
    - `network.py`: the functional `mlp` and `forward` ported from `reference_check.py`,
      `mixture_quantiles`, and `forecast`.
    - `README.md`: what the package does and where bundles come from.
- **`packages/contracts/src/contracts/settings.py`**: `primary_substation_bundle_path`, defaulting
  to `<local_artifacts_path>/primary_substation_bundle`.
- **`src/nged_substation_forecast/defs/primary_substation_forecast_assets.py`** (new): the asset, on
  `ecmwf_ens_partitions`.
- **`src/nged_substation_forecast/defs/schedules.py`**: `live_primary_substation_forecasts_job` and
  its 12:00 UTC schedule.
- **`src/nged_substation_forecast/definitions.py`**: the new module in `load_assets_from_modules`,
  and the schedule in `schedules=[...]`.
- **`Dockerfile` and `scripts/deploy/build_and_verify_image.sh`**: a second named build context,
  `primary_substation_bundle`, copied to the bundle path. The verify step runs the SHA-256 check of
  every file against `manifest.json` and prints the bundle's research commit.

### Implementation order

1. **The `substation_demand_forecaster` package without the asset**, with the golden tests. This is
   the part where train/serve skew can hide, so it gets reviewed on its own. It does not wait for
   the contract change.
2. **The asset, schedule, definitions, settings, and image changes**, once `PowerForecast` has its
   quantile columns.
3. **Docs**: the runbook steps and the release-log row.

## Design-philosophy check

- **Production path, so it degrades.** A missing ENS run, missing members, missing history runs,
  and missing lags each degrade, and each that the asset reports reaches Sentry with a tag naming
  the asset and the run (rule 4, and "make the telemetry name the fault"). Only our own faults
  raise: a corrupt bundle and a contract violation.
- **The degradation is not recorded on the row.** `PowerForecast` has no degradation column, so the
  record is the Sentry event, as for `live_forecasts` today. The in-band signal of widening bands
  is not implemented here; the network was trained with lag masking, so its bands already widen
  when a lag is missing.
- **Principle kept: "keep the serving path close to load a model, call `predict`".** The serving
  path is "load a bundle, build features, call `forecast`".
- **Principle traded: the documented `BaseForecaster` extension point.** The plan does not subclass
  `BaseForecaster`. `AllFeatures` is keyed by `ensemble_member` and carries no pooled 51-member
  summary, and issue 1 of `nged-ml-research` plans to remove the `FeatureEngineer` slot and the
  MLflow save and load in `BaseForecaster` in January. A plain function avoids building on code
  that is about to go.
- **Principle kept: the training loop owns training.** Production holds no training code. The
  monthly refit is research's job until automatic retraining lands in 2027.

## Tests

- **Golden tests against a bundle (`tests/test_substation_demand_forecaster_golden.py`), skipped
  unless `PRIMARY_SUBSTATION_BUNDLE_PATH` names a bundle.** They run on the workstation against both
  bundles (the production `drop/2026-10` and the test `replay/2026-04`), because the fixtures hold
  NGED telemetry that is not committed. For each fixture origin they compare, column by column:
    - production's ENS summary against `ens_summary.parquet`;
    - every `raw__` column of `build_input_matrix` against `input_matrix.parquet`, to 1e-5. The
      bundle's `cams_columns_source` says how the two CAMS columns were filled. For the `drop`
      bundle production's zeros are compared; for the `replay` bundle, which carries real CAMS
      values, the test feeds the fixture's own CAMS columns into the matrix, so the rest of the
      matrix and the outputs are still compared;
    - the expected-weather rows against `lag_and_expected_rows.parquet`;
    - the seed-mean quantiles at the 11 research levels (the fixtures' levels, passed as the
      `levels` argument) against `outputs_native.parquet`, to 2e-5
      of capacity;
    - the contract-origin megawatts, before the MVA clamp, against `outputs_contract.parquet` on
      its 671 rows after 12:00 UTC. The fixtures are unclamped, so the comparison comes before the
      clamp.

  Each comparison would fail on `main` today because none of the functions exists, and the ENS
  comparison also fails if the member-then-average order is swapped.
- **The golden tests read production's own Delta tables, not the fixture inputs,** for the ENS and
  the power, so they test the real read path. The fixture's `ens_native.parquet` is used only to
  localise a mismatch.
- **Committed unit tests on synthetic data (CI):**
    - `summarise_runs`: two members whose u and v components point in opposite directions give a
      mean wind speed equal to their speed, not zero (fails if u and v are averaged before the
      speed is taken).
    - `b1_columns`: a synthetic cold spell whose last warm day is 11 days before the origin gives a
      cold-snap count of 10 (the cap), and 9.5 when the last warm day is 9.5 days back (fails with
      a 9-day history).
    - `b1_columns`: with one day of history missing, every B1 value is finite and the standardised
      value of an all-NaN aggregate is 0.
    - `lag_columns`: for targets 20 and 30 hours after the native origin, the day-lag offsets are 96
      and 144 half-hours back from the target, as the formula gives; a missing reading gives value
      0, flag 1, and the largest age code.
    - `calendar_columns`: the local hour on the two clock-change days, and the bank-holiday columns
      on Good Friday 2026.
    - `sun_columns`: against a published SPA value at one timestamp and site.
    - `mixture_quantiles`: for a one-component mixture, the quantiles equal the normal quantiles to
      1e-5.
    - `build_expected_rows`: with the run that should cover g missing, the row comes from the
      next-freshest covering run; with none, the row is all zeros.
    - The asset, with a stub bundle of random weights and a tiny NWP and power table: writes 16 ×
      671 rows, every row strictly after the init time; a second run of the same partition leaves
      the row count unchanged; with today's run absent, it writes nothing and calls
      `report_asset_degradation`; MVA rows have no negative quantile.

## Docs to update

- `docs/live_service/operations.md`: how to install a new monthly bundle, and how a backfill's lags
  can differ from the live run's.
- `docs/live_service/aws.md`: the second build context.
- `docs/architecture/production-deployment.md`: the second live asset and why it runs once a day.
- `docs/live_service/model-releases.md`: the release row when the first bundle ships, naming
  experiments 677, 704, and 705.
- `CLAUDE.md`: the package table and the asset list.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
PRIMARY_SUBSTATION_BUNDLE_PATH=<bundle> uv run pytest tests/test_substation_demand_forecaster_golden.py  # workstation
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
scripts/deploy/build_and_verify_image.sh   # bundle checksums and copy
```

## Risks and open questions

- **Not subclassing `BaseForecaster`.** Recommendation: a plain function, for the reason under
  "Design-philosophy check". The cost is that `promoted_model` and `load_forecaster_from_dir` do
  not apply; the bundle is copied by hand from the workstation, as the runbook will say.
- **`pvlib` in the image, or a port of SPA.** Research's sun columns come from pvlib's SPA, and
  ml_core's NOAA geometry will not match to 1e-5. Recommendation: make `pvlib` a runtime dependency
  of `substation_demand_forecaster` and call its numpy SPA functions, so no pandas object appears in
  our code. The cost is pandas arriving in the image as pvlib's own dependency.
- **The first production bundle is refitted on 2026-10-01, so its offset is held flat past
  October.** The next refit, with a 2026-12-01 cut-off, can first run on about 2026-12-06, because
  ERA5 arrives about 5 days late (experiment 705, finding 3).
- **Holiday calendar end.** Research's `HOLIDAY_YEARS` ends in 2028; production computes the
  calendar for the years it needs, so nothing breaks in 2029.
- **No Sentry check-in monitor for the daily schedule.** What is lost is an alarm for the one
  schedule being switched off while the daemon runs. `send_forecast_checkin` takes a
  `monitor_slug` argument, so a second monitor is cheap to add later if wanted.

### Simplicity review: what changed

Ten findings were applied. The forward pass moved from `torch` to numpy, ported from the functional
reference. The asset check, the member threshold, the `availability_mode` config, and the new
Sentry monitor were cut. `select_analysis_proxy` stays unchanged, the asset reuses
`ecmwf_ens_partitions`, and the checksums moved to image build time. The solar-geometry finding is
superseded by the correctness review's finding that research uses pvlib's SPA.

### Correctness review: what changed

All 18 findings were checked against the code and accepted. The history scan grew from 9 to 13
days for the cold-snap column. The golden tests now feed the `replay` bundle's own CAMS columns and
compare before the MVA clamp. `forecast` takes the quantile levels as an argument, and the output
now uses the 13 delivery levels, which the bundle's config (`quantile_levels`) does not fix. The sun
columns now come from pvlib's SPA. The
`definitions.py` and root `pyproject.toml` steps were added, and the `_upsample_nwp_to_half_hourly`
column and `strict_radiation_schedule` traps are now handled. The B1 fallback now follows research's
finite-value aggregation, the expected-weather fallback is specified, and the analysis-proxy unit
test was replaced. The retry window now ends at 15:00 UTC. The version string, the release number's
source, and the backfill retry rule were fixed.
