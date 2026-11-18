# Plan: serve the step-5 demand network for the 16 Primaries (#1706)

**The problem.** The live service forecasts every series with the naive forecaster, while research
has a neural network for the 16 Primaries that beats the manual heuristic and climatology at every
lead band. The network is step 5 of research experiment 677 at monthly refits. Training stays in
`nged-ml-research`, so production needs inference only, from a plain directory of weights and
constants that research experiment 705 exports. The goal is a daily forecast for the Primaries in
the live service before the Christmas 2026 break.

**The planned solution.** A new workspace package, `demand_network`, rebuilds the network's 60
input columns from the `nwp` and `cleaned_power_time_series` Delta tables, runs a forward pass
re-implemented in numpy for each of the three seeds, and reads 51 quantiles out of each seed's Gaussian
mixture. A new daily Dagster asset, `live_primaries_network_forecasts`, runs it at the 12:00 UTC
origin on the day's 00Z ECMWF ENS run and writes the result to the existing `power_forecasts` Delta
table, encoding quantile level k as `ensemble_member` k, as `ClimatologyForecaster` already does.
The bundle is copied into the image beside the existing champion model. The 6-hourly
`live_forecasts` asset keeps running unchanged, as the fallback.

## Verdict, size, and departures

**Verdict: worth implementing, as the issue describes.** The maintainer chose the step-5 network
over the full Candidate on 2026-11-18. Experiment 677 measured a normalised MAE of the median of
7.39 and a CRPS of 4.19 points of capacity for step 5 at monthly refits, against 7.48 and 4.26 for
the Candidate. Step 5 also needs no daily K1 run, no stored lag-free table, and no neighbour list.

**Size: complex.** The five triggers:

- **What gets stored:** fires. New rows in `power_forecasts` under a new `experiment_name`, and a
  new directory in the image.
- **The production serving path:** fires. A new live asset.
- **A degradation rule:** fires. The asset needs rules for a missing ENS run, missing members, and
  missing lags.
- **More than one defensible design:** fires. The output encoding and whether to subclass
  `BaseForecaster` both have alternatives (see "Risks and open questions").
- **Callers I could not name without searching:** does not fire. `select_analysis_proxy` is called
  unchanged, and every other function is new.

**Reviews: both plan reviews (simplicity, then correctness), and all three diff reviews.**

**Departures from the issue body.** The issue does not say how quantiles are stored. This plan
stores them as 51 ensemble members in the existing `PowerForecast` schema rather than adding the
`p1` to `p99` columns of the planned Representation 2, because no contract changes, and because
every existing reader (dashboard, metrics, live health check) already handles member-encoded
quantiles from `ClimatologyForecaster`.

## Solution

### Walk through

#### The happy path

1. **A schedule fires `live_primaries_network_forecasts` at 12:00 UTC each day.** The asset
   reuses `ecmwf_ens_partitions` (`defs/assets.py`), so its partition maps one-to-one to the day's
   ENS run; the partition's date gives `power_fcst_init_time` = that date at 12:00 UTC
   and the native origin = 09:00 UTC, the time the network was trained to treat the 00Z run as
   available. The 12:00 slot is later than `ecmwf_ens_schedule` (10:30 UTC), so on a normal day the
   run is already on disk; the two schedules couple through data at rest, never through run status
   (design principle 14).
2. **The asset loads the bundle** with `demand_network.bundle.load_bundle(path)` from the new
   `Settings.primaries_network_bundle_path`. `load_bundle` checks `bundle_format_version == 2`,
   reads each seed's weights from `components/step5/weights/seed<N>.npz` with `numpy.load`, and
   returns
   a frozen `Bundle` dataclass holding the config, the standardisers, the column order, the offset
   knots, and the 16 Primaries from `primaries.json` (identifier, H3 cell, latitude, longitude,
   `effective_capacity_mw`, `capacity_mw`, embedding index). The Primaries and their capacities come
   from the bundle, never from today's `TimeSeriesMetadata` table, for the same reason
   `live_forecasts` uses its frozen metadata copy.
3. **The asset checks that today's 00Z run is present** with the existing
   `_available_nwp_init_times`. If it is not, the asset raises `RetryRequested` (every 30 minutes,
   up to 5 times, so until 14:30 UTC), which matches the last retry of `ecmwf_ens` (8 retries from
   10:30 UTC).
4. **The asset reads the NWP it needs in one scan** of the `nwp` Delta table, pruned to the 16 H3
   cells and to runs initialised from the native origin minus 9 days to today's 00Z run. The 9 days
   cover the expected-weather rows, which reach back to 192 hours before the native origin, and the
   trailing windows of the thermal-lag columns.
5. **`demand_network.ens_summary.summarise_runs(nwp, cells)` builds the ENS summary per run,** by
   calling the existing `ml_core.features._nwp._upsample_nwp_to_half_hourly` on each member
   separately (with the Primaries' coordinates as `solar_coordinates`), deriving `wind_speed_10m =
   sqrt(u² + v²)` per member, and only then taking the member mean and the population standard
   deviation (`ddof=0`). Experiment 705 found that averaging first is wrong by up to 4.24 m/s in
   wind speed. Precipitation gets `log1p` of the value clipped at zero after averaging, as the
   bundle's `ens_summary_recipe` states.
6. **`demand_network.features.build_input_matrix(...)` builds the 60 columns** for each (Primary,
   valid half-hour) from the native origin plus 0.5 hours to plus 339 hours, in the order of
   `components/step5/columns.json`:
    - the 14 ENS means and standard deviations of today's run;
    - the 21 calendar and sun columns, computed from the valid time (solar zenith cosine, azimuth
      sine and cosine, local-hour sines and cosines at 24 and 12 hours in Europe/London, one-hot day
      of week, the COVID flag, three Christmas columns, and three bank-holiday columns from the
      `holidays` package's England calendar, as research experiment 87 does);
    - the 2 lead columns (lead over 336 hours, and its square root);
    - the 14 thermal-lag and solar-gain ("B1") columns, from the analysis proxy before the native
      origin and today's run after it, with the two previous-day CAMS columns set to zero;
    - the 8 lag columns: day and week lag value, missing flag, and age code, plus the two grid
      indices, with the lag times from the rule in the bundle's config (`target - 48 * ceil((target
      - (origin - 48)) / 48)` in half-hour steps, and the same with 336), reading cleaned power from
      `nged_data.storage.scan_cleaned_power` divided by `effective_capacity_mw`;
    - the knot position (months since 2019-09 plus the fraction of the month, clamped to the knot
      range).

   The analysis proxy comes from `select_analysis_proxy` called on the ENS summary means with the
   two new arguments, `member=None` (take the frame as already reduced) and `min_lead=9 h` with
   `max_lead=33.5 h`, which experiment 705 showed reproduces the research history exactly.
   `select_analysis_proxy` itself is not changed: `demand_network.features.analysis_proxy` filters
   the summary means to ENS leads above 9 and below 33.5 hours, labels the frame
   `ensemble_member = NWP_ANALYSIS_MEMBER`, and calls the function as it stands. The
   expected-weather rows for each lag time come from `build_expected_rows(...)`: for each half-hour
   g in (native origin − 192 h, native origin − 24 h], the ENS summary row of the freshest run whose
   native 09:00 origin is at or before g, its 7 weather means and 21 recomputed date-time columns.
   Standardisation follows the rules in `standardisers.json` (float32 arithmetic, clip at ±5, NaN
   to 0 for the 37 lag-free columns; no clip for the B1 columns; float16 rounding of the
   expected-weather rows).
7. **`demand_network.network.forecast(bundle, matrix, expected_rows)`** runs each seed's forward
   pass in numpy, with `scipy.special.erf` for the exact GELU and the mixture's cumulative
   distribution (the weather, date-time, and shared lag encoders, then the mixer and the per-Primary offset
   interpolated between knots), reads 51 quantiles at levels (k + 0.5) / 51 from each seed's
   mixture by 40 bisection steps over ±40 standard deviations, converts each seed's quantiles to a
   fraction of capacity with the per-Primary target mean and standard deviation, averages the three
   seeds level by level, and sorts within each row. The forward pass is ported from the bundle's
   functional `forward` and `mlp` of `reference_check.py`, applying the stored weights by their
   `state_dict` names, so the port diffs line by line against the reference and imports nothing
   from research.
8. **The asset relabels the origin to 12:00 UTC, multiplies by `capacity_mw`, and keeps valid
   times strictly after 12:00 UTC and up to 12:00 UTC 14 days later** (671 half-hours per Primary,
   because `PowerForecast.valid_time` must be strictly after `power_fcst_init_time` and research's
   lead-0 row is dropped). It clamps every quantile of the eight MVA Primaries at zero, as the
   backtest files do. It builds a `PowerForecast` frame with `ensemble_member` k for level k,
   `power_fcst_model_name = "primaries_demand_network"`, `experiment_name =
   "primaries_demand_network"`, `power_fcst_model_version` from the bundle's release number (0
   until a release is cut), `nwp_init_time` = today's 00Z run, and `fold_id = "live"`.
9. **The asset writes with `write_power_forecasts`**, replacing exactly the partition
   (`experiment_name`, `"live"`) and this `power_fcst_init_time`, so a rerun never duplicates rows,
   No Sentry check-in monitor is added: a failed run reaches Sentry through
   `sentry_capture_failure`, a missing run through the degradation report, and a dead daemon
   through the existing `live-forecasts` monitor.

#### The main branches off the happy path

- **A backfill of a past partition** reads the same NWP runs as the live run did, because the
  input rule is defined by the native origin and not by what is on disk now. The asset has no
  `availability_mode` config.
- **A Primary whose day or week lag is missing** (no cleaned reading at the lag time, or a stale
  telemetry feed) gets the missing encoding the network was trained with (value 0, flag 1, the
  largest age code). The network trained with 30% of origin days lag-masked, so this degrades the
  forecast towards the lag-free forecast rather than failing.
- **An analysis-proxy half-hour with no first-day run** (a missing daily run in the past 9 days).
  The first-day windows of consecutive daily runs touch with only 0.5 hours of overlap, so the
  windowed proxy leaves a gap of a whole day when one run is missing. The asset coalesces the
  windowed proxy with a second call of `select_analysis_proxy` without the lead window, which fills
  the gap from the freshest older run that covers it, and reports the degradation. Any half-hour
  still null after that (several runs missing) gets the standardised value 0 in its B1 columns,
  the column's training mean, rather than reaching the network as NaN. Without this branch, one
  missed ENS day would fail the asset for the 9 days the B1 columns reach back. Research leaves
  the value missing, so this branch differs from research by design; it is the only branch that
  does.

#### The main error-handling paths

- **Today's 00Z run is absent at 14:00 UTC**: the asset reports a degradation to Sentry tagged
  `degraded_asset=live_primaries_network_forecasts`, naming the run, and finishes without writing.
  It does not forecast from yesterday's run, because every lead would be 24 hours longer than the
  network trained on. NGED keeps the fallback forecast from `live_forecasts`, which runs at every
  slot.
- **Today's run has fewer than 51 members in a cell**: the summary is the mean and standard
  deviation over the members present. `nwp_run_is_complete` already warns about the run and names
  the absent members, so the asset adds no report of its own.
- **The bundle is missing or has an unknown format version**: `load_bundle` raises. The SHA-256 of
  every file is checked once, at image build time, because the image cannot change after it is
  built. That is our own deployment fault, not the outside world, so raising is right under the
  inherent-stability page; the run fails loudly and the Sentry failure hook reports it.
- **The output fails `PowerForecast.validate`** (a duplicated key, a NaN quantile): raises, because
  that is our bug.

### What changes, file by file

- **No `torch` dependency.** The forward pass is a few linear layers, GELU, embedding lookups,
  `log_softmax`, `erf`, and bisection, all of which numpy and scipy do. `holidays` is added as a
  dependency of the new package; `numpy` and `scipy` are already locked.
- **Research prerequisite (in `nged-ml-research`):** experiment 705's exporter also writes each
  seed's weights as `seed<N>.npz`, and the bundle format becomes version 2. The `.pt` files can
  stay for research's own `reference_check.py`.
- **`packages/demand_network/`** (new workspace member):
    - `bundle.py`: `Bundle` (frozen dataclass), `load_bundle(path)`, `BUNDLE_FORMAT_VERSION: Final =
      1`.
    - `ens_summary.py`: `summarise_runs(nwp, coordinates)`.
    - `features.py`: `calendar_columns`, `b1_columns`, `lag_columns`, `knot_position`,
      `build_expected_rows`, `build_input_matrix`.
    - `network.py`: the functional `mlp` and `forward` ported from `reference_check.py`,
      `mixture_quantiles`, and `forecast`.
    - `README.md`: what the package does and where bundles come from.
- **`packages/contracts/src/contracts/settings.py`**: `primaries_network_bundle_path`, defaulting
  to `<local_artifacts_path>/primaries_network_bundle`.
- **`src/nged_substation_forecast/defs/primaries_network_assets.py`** (new): the asset, on
  `ecmwf_ens_partitions`.
- **`src/nged_substation_forecast/defs/schedules.py`**: `live_primaries_network_forecasts_job` and
  its 12:00 UTC schedule.
- **`Dockerfile` and `scripts/deploy/build_and_verify_image.sh`**: a second named build context,
  `primaries_network_bundle`, copied to the bundle path. The verify step runs the SHA-256 check of
  every file against `manifest.json` and prints the bundle's research commit.

### Implementation order (three pull requests on top of this plan)

1. **The `demand_network` package without the asset**, with the golden tests. This is the part where train/serve skew can hide, so it gets reviewed on its
   own.
2. **The asset, schedule, settings, and image changes.**
3. **Docs**: the runbook steps and the release-log row.

## Design-philosophy check

- **Production path, so it degrades.** A missing ENS run, missing members, and missing lags each
  degrade and report to Sentry with a tag that names the asset and the run (rule 4 and "make the
  telemetry name the fault"). Only our own faults raise: a corrupt bundle and a contract violation.
- **The degradation is not recorded on the row.** `PowerForecast` has no degradation column, so the
  record is the Sentry event, as for `live_forecasts` today. The in-band signal of widening bands
  is not implemented here; the network was trained with lag masking, so its bands already widen
  when a lag is missing.
- **Principle traded: "keep the serving path close to load a model, call `predict`".** The serving
  path is "load a bundle, build features, call `forecast`", which keeps to it. The plan does not
  subclass `BaseForecaster`, which trades away the documented extension point. What that buys:
  `AllFeatures` is keyed by `ensemble_member` and carries no pooled 51-member summary, and issue 1
  of `nged-ml-research` plans to remove the `FeatureEngineer` slot and the MLflow save and load in
  `BaseForecaster` in January. A plain function avoids building on code that is about to go.
- **Principle kept: the training loop owns training.** Production holds no training code; the
  monthly refit is research's job until automatic retraining lands in 2027.

## Tests

- **Golden tests against a bundle (`tests/test_demand_network_golden.py`), skipped unless
  `PRIMARIES_NETWORK_BUNDLE_PATH` names a bundle.** They run on the workstation against both
  bundles (the production `drop/2026-10` and the test `replay/2026-04`), because the fixtures hold
  NGED telemetry that is not committed. For each fixture origin they compare, column by column,
  production's `ens_summary` against `ens_summary.parquet`, every `raw__` column of
  `build_input_matrix` against `input_matrix.parquet` (to 1e-5), the expected-weather rows against
  `lag_and_expected_rows.parquet`, and the seed-mean quantiles at the 11 research levels against
  `outputs_native.parquet` (to 2e-5 of capacity). The contract rows are compared with
  `outputs_contract.parquet` on its 671 rows after 12:00 UTC, because the research file's first row
  sits at 12:00 UTC itself and `PowerForecast` drops it. Each would fail on `main` today because none of
  the functions exists; each also fails if the member order of step 5 is swapped back.
- **The golden tests read production's own Delta tables, not the fixture inputs,** for the ENS and
  the power, so they test the real read path. The fixture's `ens_native.parquet` is used only to
  localise a mismatch.
- **Committed unit tests on synthetic data (CI):**
    - `analysis_proxy`: on three daily runs with the middle run missing, the half-hours of the
      missing run's first day come from the older run's later leads and are not null.
    - `summarise_runs`: two members whose 3-hourly radiation peaks differ give a different standard
      deviation from the average-then-interpolate order (fails if the order is swapped).
    - `lag_columns`: a target 30 hours after the native origin reads the lag 48 half-hours earlier
      than a target 20 hours after it, at the boundary the formula sets; a missing reading gives
      value 0, flag 1, and the largest age code.
    - `calendar_columns`: the local hour on the two clock-change days, and the bank-holiday columns
      on Good Friday 2026.
    - `mixture_quantiles`: for a one-component mixture, the quantiles equal the normal quantiles to
      1e-5.
    - The asset, with a stub bundle of random weights and a tiny NWP and power table: writes 16 ×
      671 × 51 rows, every row strictly after the init time; a second run of the same partition
      leaves the row count unchanged; with today's run absent, it writes nothing and calls
      `report_asset_degradation`; MVA rows have no negative quantile.

## Docs to update

- `docs/live_service/operations.md`: how to install a new monthly bundle and backfill a day.
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
PRIMARIES_NETWORK_BUNDLE_PATH=<bundle> uv run pytest tests/test_demand_network_golden.py  # workstation
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
scripts/deploy/build_and_verify_image.sh   # image size and bundle copy
```

## Risks and open questions

- **Quantiles as members, or Representation 2 columns?** Recommendation: members now, because no
  contract changes and every reader already copes; derive the `p1` to `p99` delivery columns later
  from the 51 levels, when the delivery table is built.
- **Not subclassing `BaseForecaster`.** Recommendation: a plain function, for the reason under
  "Design-philosophy check". The cost is that `promoted_model` and `load_forecaster_from_dir` do
  not apply; the bundle is copied by hand from the workstation, as the runbook will say.
- **The threshold of 40 members.** A guess, chosen so a run missing a few members still forecasts.
  Research raises below 51.
- **Holiday calendar end.** Research's `HOLIDAY_YEARS` ends in 2028; production computes the
  calendar for the years it needs, so nothing breaks in 2029.
- **The first production bundle is refitted on 2026-10-01, so its offset is held flat past
  October.** The next refit, with a 2026-12-01 cut-off, can first run on about 2026-12-06 because
  ERA5 arrives about 5 days late (experiment 705, finding 3).
- **numpy instead of `torch`.** Recommendation: numpy, which keeps the image the size it is and
  adds no dependency; the cost is the small research change to write `.npz` weights. If the later
  Candidate or prevailing head needs operations numpy lacks, `torch` can be added then.
- **No Sentry check-in monitor for the daily schedule.** What is lost is an alarm for the one
  schedule being switched off while the daemon runs. Recommendation: add it later if wanted,
  because `send_forecast_checkin` hard-codes the `live_forecasts` monitor today.
- **Solar geometry.** If research's zenith and azimuth columns match ml_core's NOAA geometry in
  `_half_hour_clear_sky`, move that geometry into its own function and reuse it; the golden tests
  decide.

### Simplicity review: findings rejected

None rejected outright. Ten findings were applied as above, and the solar-geometry finding is left
for the golden tests to decide.
