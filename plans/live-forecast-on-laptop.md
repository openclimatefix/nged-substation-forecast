# Live forecast running 6-hourly on the laptop (epic #137)

> Temporary mechanical checklist for the in-flight PR (repo convention: `plans/` holds at most
> one file, deleted when the work merges — paste this, or a summary, into the PR body first).
> Nothing may link here from code or `docs/`.

## Context

Epic [#137](https://github.com/openclimatefix/nged-substation-forecast/issues/137) (v0.1:
deploy naïve forecast on AWS). Goal of this plan: **the forecast runs automatically every
6 hours on Jack's laptop**. That covers:

- **#221** — the `live_forecasts` Dagster asset (production inference: plain disk model load,
  no MLflow at runtime, single-run feature engineering across all ensemble members, idempotent
  Delta write with `fold_id="live"`).
- **#208 (local half)** — 6-hourly automation via `dg dev` + persistent `DAGSTER_HOME`, plus a
  replay-mode backfill test.

Deferred (do NOT do here): **#246 (scale `power_fcst` to [−1, +1]) — explicitly deferred by
Jack; forecasts stay in MW/MVA for now**, #121/#50 (S3 paths), #197 (MLflow param bug — not on
the live path, which never touches MLflow), #222 (Docker), #206 (AWS), #63 (Sentry), #209
(version bump).

Authoritative design: `docs/roadmap/live-service.md` — sections "The live_forecasts asset",
"Production model artifacts", and "Implementation details → live_forecasts". Decisions already
made with Jack (2026-07-04):

1. Model promotion is a **manually-triggered Dagster asset** `production_model` (config
   `mlflow_run_id`) — not the `scripts/fetch_model.py` sketch in the design doc; keep the
   download logic in a pure helper so a script wrapper stays trivial for #222 later.
2. Local automation = `uv run dg dev` with `DAGSTER_HOME` set per the README scaffolding
   (README.md:32-52), not systemd.

---

## The PR — #221: `live_forecasts` + `production_model` + schedules

### ml_core changes

1. **Shared constant** `NWP_PUBLICATION_DELAY_HOURS: Final[int] = 6` in
   `packages/ml_core/src/ml_core/features/_nwp.py` (MkDocs-style string-literal doc, per
   CLAUDE.md); use it as the default in `_engineer_features`
   (`packages/ml_core/src/ml_core/features/tabular_feature_engineer.py:93`) and everywhere
   below.
2. **Extend `FeatureEngineer.engineer()`**
   (`packages/ml_core/src/ml_core/features/feature_engineer.py:21-30`) with single-run
   params, defaults preserving bulk mode (CV callers untouched):
   `power_fcst_init_time: datetime | None = None`, `nwp_init_time: datetime | None = None`,
   `nwp_publication_delay_hours: int = NWP_PUBLICATION_DELAY_HOURS`.
   `TabularFeatureEngineer.engineer` (tabular_feature_engineer.py:68-83) passes them straight
   through to `_engineer_features`, which already implements single-run mode — the public
   interface just doesn't expose it today. Docstring: summarise bulk vs single-run modes,
   pointing at `_engineer_features` for detail.
3. **`model_class` in meta.json**: `XGBoostForecaster.save`
   (`packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py:~196-208`) adds
   `"model_class": f"{type(self).__module__}.{type(self).__qualname__}"` to the meta.json
   dict (`load()` ignores unknown keys — no change needed there). Document the contract in
   the `BaseForecaster.save` docstring
   (`packages/ml_core/src/ml_core/base_forecaster.py:~113-123`): a saved directory must
   contain `meta.json` with a `model_class` field so production can reconstruct the concrete
   class (issue #221 specifies this).
4. **New module `packages/ml_core/src/ml_core/_production_helpers.py`** (pure, unit-tested,
   injected clock — no wall-clock reads anywhere):
   - `AvailabilityModeType = Literal["live", "replay"]` (follow the `Type`-suffix convention
     from CLAUDE.md).
   - `select_nwp_init_time(available_init_times: Sequence[datetime], *, t0: datetime,
     availability_mode: AvailabilityModeType,
     nwp_publication_delay_hours: int = NWP_PUBLICATION_DELAY_HOURS) -> datetime` —
     cutoff = `t0` for `"live"` (no modelled delay: the Delta table only contains genuinely
     published runs), `t0 − timedelta(hours=delay)` for `"replay"` (reconstructs what was
     available at a historical t0). Return the freshest init ≤ cutoff; raise `ValueError`
     listing cutoff + available runs when none qualify.
   - `build_live_power_frame(observed_power: pt.LazyFrame[PowerTimeSeries],
     time_series_ids: list[int], *, t0: datetime, history: timedelta, horizon: timedelta)
     -> pt.LazyFrame[PowerTimeSeries]` — dense half-hourly `(time_series_id, time)` spine
     over `(t0 − history, t0 + horizon]` left-joined with observations (future rows have
     null power). **Needed because `_join_nwp_single_run`
     (`packages/ml_core/src/ml_core/features/_nwp.py:40-68`) is power-centric** — with no
     future power rows a live run would emit zero forecast rows. Also harmless for replay
     (future observations exist; `_nullify_leaky_lags` already prevents lag leakage).
   - `load_forecaster_from_dir(path: Path) -> BaseForecaster` — read `meta.json`, resolve
     `meta["model_class"]` via `hydra.utils.get_class` (same mechanism as
     `load_experiment_forecaster`, `packages/ml_core/src/ml_core/_mlflow_runs.py:52`),
     return `cls.load(path)`. Friendly errors: missing dir/meta.json → "materialise the
     production_model asset first"; missing `model_class` key → "re-promote with a code
     version that stamps model_class".
   - `fetch_model_artifacts(run_id: str, dest: Path) -> None` —
     `mlflow.artifacts.download_artifacts(run_id=..., artifact_path="model", dst_path=tmp)`
     (mirror `base_forecaster.py:161-165`) into a temp dir, then atomically replace `dest`
     (rmtree + move) so a half-failed download never leaves a corrupt model. Also write a
     `promotion.json` (`{"mlflow_run_id": ..., "promoted_at": ...}`) into `dest` for
     provenance (`load()` globs `*.ubj`, so an extra JSON file is unaffected). Caller sets
     the tracking URI.

### contracts

5. **`Settings.production_model_path: Path`** (default
   `PROJECT_ROOT / "data" / "production_model"`) in
   `packages/contracts/src/contracts/settings.py`, after `model_cache_base_path`
   (settings.py:125-134). Description: written by the `production_model` asset, read by
   `live_forecasts` via a plain `BaseForecaster` disk load — no MLflow at inference time;
   later `COPY`'d into the container image at build time.

### New file `src/nged_substation_forecast/defs/production_assets.py`

(New file per the design doc — `cv_assets.py` is already ~900 lines.)

6. Module constants (with string-literal docs):
   - `LIVE_FORECAST_HORIZON: Final[timedelta] = timedelta(days=14)`
   - `LIVE_POWER_HISTORY: Final[timedelta] = timedelta(days=15)` (≥ longest power lag,
     336 h, + margin)
   - `live_forecast_partitions = TimeWindowPartitionsDefinition(
     cron_schedule="0 0,6,12,18 * * *", start=<a few days before ship date>,
     fmt="%Y-%m-%d-%H:%M", timezone="UTC")` — no deep empty backlog.
7. **`production_model` asset** — unpartitioned, no deps, manually triggered from the UI
   launchpad. `ProductionModelConfig(Config)` with `mlflow_run_id: str` (the champion fold
   run id from the MLflow leaderboard). Body: `settings = Settings()`;
   `mlflow.set_tracking_uri(settings.mlflow_tracking_uri)`;
   `fetch_model_artifacts(config.mlflow_run_id, settings.production_model_path)`; read back
   `meta.json`; `context.add_output_metadata`: `mlflow_run_id`, `model_class`,
   `experiment_name` (from `model_params`), `n_trained_time_series`, the path. (Promotion =
   a Dagster materialisation → audit trail + lineage for free.)
8. **`live_forecasts` asset**:

   ```python
   @asset(
       partitions_def=live_forecast_partitions,
       deps=[
           AssetDep("ecmwf_ens",
                    partition_mapping=TimeWindowPartitionMapping(start_offset=-16, end_offset=0)),
           "power_time_series_and_metadata",
           "production_model",
       ],
   )
   def live_forecasts(context: AssetExecutionContext, config: LiveForecastsConfig) -> None:
   ```

   `LiveForecastsConfig(Config)` with
   `availability_mode: AvailabilityModeType = "live"`. Thin body over the tested helpers:

   1. `t0 = context.partition_time_window.end` — partition key is the window *start*; t0 is
      the window *end* (the scheduled tick). **Document prominently in the asset docstring**
      ("partition 2026-07-04-00:00 produces the forecast initialised 06:00") or backfills
      will confuse.
   2. `forecaster = load_forecaster_from_dir(settings.production_model_path)`; raise if
      `trained_time_series_ids` is empty. **No MLflow import/use anywhere in this asset.**
   3. `available = _available_nwp_init_times(settings)` — private IO helper in this module
      using `DeltaTable(str(settings.nwp_data_path)).partitions()` (partition metadata only,
      no data scan), parsing `init_time` partition values to tz-aware UTC datetimes.
   4. `nwp_init = select_nwp_init_time(available, t0=t0,
      availability_mode=config.availability_mode)`.
   5. Reuse `_load_engineering_inputs` (import from `defs.cv_assets`,
      `src/nged_substation_forecast/defs/cv_assets.py:231`) with
      `window_start = t0 − LIVE_POWER_HISTORY`, `window_end = t0 + LIVE_FORECAST_HORIZON`,
      `init_time_start=init_time_end=nwp_init`, `ensemble_members=None` (all ~51 members).
      It already prunes the NWP scan to one run × relevant H3 cells.
   6. `power_full = build_live_power_frame(power_ts, forecaster.trained_time_series_ids,
      t0=t0, history=LIVE_POWER_HISTORY, horizon=LIVE_FORECAST_HORIZON)`.
   7. `features = forecaster.feature_engineer.engineer(
      selected_features=forecaster.model_params.selected_features,
      power_time_series=power_full, time_series_metadata=metadata_df, nwp=nwp_lf,
      power_fcst_init_time=t0, nwp_init_time=nwp_init)`.
   8. Filter to genuine forecast rows: `valid_time > t0` and `ensemble_member.is_not_null()`
      (history rows and beyond-run-coverage join misses); re-wrap per the CLAUDE.md Patito
      filter gotcha (`pt.LazyFrame.from_existing(...).set_model(AllFeatures)`).
   9. `forecasts = forecaster.predict(features)` — `fold_id="live"` is the default
      (`base_forecaster.py:184`); raise `ValueError` if `forecasts.height == 0`.
   10. Idempotent write:

       ```python
       write_deltalake(
           table_or_uri=settings.power_forecasts_data_path,
           data=forecasts.to_arrow(),
           mode="overwrite",
           predicate=(
               f"experiment_name = '{forecaster.model_params.experiment_name}' "
               f"AND fold_id = 'live' "
               f"AND power_fcst_init_time = '<t0 literal>'"
           ),
           partition_by=["experiment_name", "fold_id"],
       )
       ```

       The `power_fcst_init_time` term is **essential** — without it a re-run of one 6-hour
       slot wipes every live row in the partition. It is a non-partition column, which
       delta-rs supports in replaceWhere predicates. The timestamp-literal syntax is the main
       unknown — pin it empirically via the idempotency + accumulation tests (candidates:
       `t0.isoformat()` string, `t0.strftime('%Y-%m-%d %H:%M:%S')`, DataFusion
       `TIMESTAMP '...'`).
   11. `context.add_output_metadata`: t0, availability_mode, chosen `nwp_init_time`, n_rows,
       n_time_series, n_ensemble_members, experiment_name.

### Schedules & wiring

9. `src/nged_substation_forecast/defs/assets.py`: extract the inline
   `DailyPartitionsDefinition(start_date="2024-04-01", timezone="UTC", end_offset=1)`
   (assets.py:143) to a module-level `ecmwf_ens_partitions` so the new job shares the same
   object.
10. `src/nged_substation_forecast/defs/schedules.py` (pattern: the existing hourly
    `power_time_series_and_metadata_schedule` in the same file):
    - **`ecmwf_ens` currently has no schedule at all** — hands-off operation needs one:

      ```python
      ecmwf_ens_job = define_asset_job(
          "ecmwf_ens_job", selection=AssetSelection.assets("ecmwf_ens"),
          partitions_def=ecmwf_ens_partitions)

      @schedule(job=ecmwf_ens_job, cron_schedule="30 7 * * *", execution_timezone="UTC")
      def ecmwf_ens_schedule(context: ScheduleEvaluationContext) -> RunRequest:
          return RunRequest(partition_key=context.scheduled_execution_time.strftime("%Y-%m-%d"))
      ```

      07:30 UTC ≈ 00Z init + publication/ingest delay; today's key exists thanks to
      `end_offset=1`. Tune against Dynamical's actual landing time later — a too-early run
      fails cleanly and is re-materialisable, and live mode always takes the freshest run
      actually present.
    - 6-hourly live schedule:

      ```python
      live_forecasts_job = define_asset_job(
          "live_forecasts_job", selection=AssetSelection.assets("live_forecasts"),
          partitions_def=live_forecast_partitions)
      live_forecasts_schedule = build_schedule_from_partitioned_job(live_forecasts_job)
      ```

      Ticks at 00/06/12/18 UTC, materialising the just-completed window with default config
      → `availability_mode="live"` (design: the schedule is always live; replays are
      manual).
11. `src/nged_substation_forecast/definitions.py`: add `production_assets` to
    `load_assets_from_modules([assets, cv_assets, production_assets])`; add
    `ecmwf_ens_schedule` and `live_forecasts_schedule` to the `schedules=[...]` list (jobs
    referenced by schedules are included automatically — matches the existing pattern).

### Tests

- `packages/ml_core/tests/test_production_helpers.py` (new, unit):
  - `select_nwp_init_time`: live picks freshest ≤ t0; replay picks freshest ≤ t0−6h;
    **live vs replay diverge** when a run exists in `(t0−6h, t0]`; raises when nothing
    qualifies. All with injected datetimes.
  - `build_live_power_frame`: half-hourly grid bounds `(t0−history, t0+horizon]`, observed
    values joined, nulls beyond observations, all ids present.
  - `load_forecaster_from_dir`: `XGBoostForecaster(config).save(tmp)` → loader returns an
    `XGBoostForecaster`; meta.json without `model_class` raises the helpful error.
- Engineer passthrough (unit): `TabularFeatureEngineer.engineer(single-run params)` output
  equals direct `_engineer_features` single-run output on the
  `packages/ml_core/tests/test_cross_mode_equivalence.py` fixtures.
- `tests/test_production_model.py` (new, integration): file-based MLflow env — pattern the
  env fixture in `tests/test_cv_power_forecasts.py` (monkeypatch.setenv for all Settings
  paths + `MLFLOW_ALLOW_FILE_STORE=true`, see the memory note); `save_to_mlflow` a tiny
  model → materialise `production_model` with `mlflow_run_id` config → directory populated
  (meta.json + promotion.json), output metadata correct; re-promotion with a second run id
  replaces the model.
- `tests/test_live_forecasts.py` (new, integration; fixtures patterned on
  `tests/test_cv_power_forecasts.py`: fake power/NWP/metadata Delta builders, plus
  `PRODUCTION_MODEL_PATH` env var and a tiny model trained inline and saved to it. Data: two
  NWP runs — D−1 00Z and D 00Z — both covering valid times just after `t0 = D 00:00`
  (partition key `D−1 18:00`), 3 ensemble members, one trained + one untrained series).
  The design doc's required list:
  1. live vs replay select different `nwp_init_time` for the same partition (assert on the
     Delta rows).
  2. Only `trained_time_series_ids` are forecast (untrained series absent).
  3. All ensemble members present in the output.
  4. Idempotency: materialise the same partition twice → row count unchanged (this also
     pins the replaceWhere timestamp syntax).
  5. Accumulation: a second partition's rows coexist with the first (guards against the
     predicate wiping the whole live partition).

### Ship-time triage (same PR or immediate follow-up)

- `docs/roadmap/live-service.md`: mark "The live_forecasts asset" section ✅; delete the
  "`live_forecasts` — implementation notes, tests, verification" subsection (paste into the
  PR body); record the `production_model`-asset promotion decision in "Production model
  artifacts" (the asset supersedes the "researcher runs a script" step for local operation;
  a script wrapper remains trivial for the Docker build).
- Close #221; comment on #208 that the local half is running (close #208 only after the
  multi-day dress rehearsal).
- Run the markdown lint (command below).
- PR mechanics (CLAUDE.md): add labels + `gh pr edit --add-assignee JackKelly`.
- Delete this `plans/` file when the work merges.

---

## Operational steps (no PR — run after merge)

1. Materialise `power_time_series_and_metadata` and any missing recent `ecmwf_ens` daily
   partitions (UI backfill).
2. **Promote a champion**: pick the best existing fold run id from the MLflow leaderboard
   (`uv run mlflow ui --gunicorn-opts "--workers 1"` — the py3.14 gunicorn workaround);
   materialise `production_model` with that `mlflow_run_id`. No retrain needed — forecasts
   stay in MW/MVA. (If no suitable trained model exists, run one CV experiment via
   `register_experiment_job` first. Note: a model saved *before* this PR lacks `model_class`
   in meta.json — re-save/re-train if needed, or promote a freshly trained fold.)
3. **DAGSTER_HOME** per README.md:32-52: `mkdir -p ~/dagster_home`, write the sample
   `dagster.yaml` (sqlite storage `base_dir: "dagster_history"` — relative to the repo cwd
   where `dg dev` runs; keep the ECMWF concurrency-pool block); `export
   DAGSTER_HOME=~/dagster_home` in `.bashrc`.
4. `uv run dg dev` from the repo root in tmux; enable all three schedules in the UI
   (`power_time_series_and_metadata_schedule`, `ecmwf_ens_schedule`,
   `live_forecasts_schedule`).
5. **First live run**: materialise the latest `live_forecasts` partition manually (default
   config = live) rather than waiting for the tick; confirm `fold_id="live"` rows:
   `pl.scan_delta("data/power_forecasts").filter(pl.col("fold_id") == "live").collect()`.
   Plot via `plot_power_forecast_job` (`fold_id: "live"`, the model's `experiment_name`, the
   run's `power_fcst_init_time`).
6. **Replay test (#208)**: launch an already-passed partition from the launchpad with
   `availability_mode: "replay"`; confirm the chosen `nwp_init_time` respects the 6 h delay
   and re-runs don't duplicate rows.
7. Leave running for several days; after a sleep/missed slot, backfill via per-partition
   launchpad runs with replay mode (the backfill dialog may not expose run config).

## Verification

```bash
uv run pytest                                   # full suite before the PR
uv run ruff check . && uv run ruff format . && uv run ty check
uv run pymarkdown scan -r docs README.md CLAUDE.md metadata/README.md packages/*/README.md
uv run dg dev   # graph loads; live_forecasts / production_model / 3 schedules visible
```

End-to-end: operational steps 5–6 above (a real live materialisation, a replay, a plot).

## Risks / notes

1. **delta-rs replaceWhere on a timestamp column** — literal syntax unconfirmed; the
   idempotency test pins it early. Highest-uncertainty item in the PR.
2. **t0 = partition window end** (key ≠ init time by 6 h) — document prominently.
3. **Weather-lag features would go null at live time** (only one NWP run is loaded) — none
   in the current champion config; flag in the asset docstring so a future feature change
   trips over it consciously.
4. The 06:00 live tick may beat the 06:00 power ingest — near-t0 lags are null anyway
   (NGED ~5 h latency) and XGBoost handles nulls natively. Acceptable.
5. `live_forecasts` output stays in **MW/MVA** until #246 lands (deferred); the
   `PowerForecast.power_fcst` docstring already says so — no doc change needed for this.
