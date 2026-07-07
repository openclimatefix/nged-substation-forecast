# Running live forecasts end-to-end

How to get a champion model producing live, 6-hourly forecasts, and how to backfill a slot that
was missed.

There are two layers. **Promotion** (step 1–2) is manual and occasional — you do it once per
champion model, whenever a better candidate clears the [ML Experimentation](../ml_experimentation/index.md)
leaderboard. **Inference** (step 3) is automatic — once a model is promoted, the 6-hourly schedule
keeps producing forecasts from it with no further action, as long as Dagster's daemon is running.

Production inference has **zero dependency on MLflow at runtime**: the promoted model is a plain
directory on disk, loaded with a plain `save`/`load` round trip — no run id, tracking-server call,
or cache lookup on the hot path. See
[roadmap: Production model artifacts](../roadmap/live-service.md#production-model-artifacts) for
why this was chosen over reusing the CV pipeline's MLflow-cache mechanism.

---

## Prerequisites — a persistent Dagster instance

The 6-hourly schedule only fires while Dagster's daemon is running continuously, so `uv run dg
dev` needs to keep running (rather than being started and stopped around each manual step) and
needs a persistent `DAGSTER_HOME` so its schedule state survives a restart:

1. `mkdir ~/dagster_home/` and put the `dagster.yaml` shown in the repository README's
   [Setup](https://github.com/openclimatefix/nged-substation-forecast#setup) section into it.
2. `export DAGSTER_HOME=~/dagster_home` (add to `.bashrc` so it persists across terminals).
3. `uv run dg dev`, and leave it running. Open `http://localhost:3000` to use the UI.

## Step 1 — Pick a champion model

**Trigger:** Materialise `promotable_model_runs` (unpartitioned — no config needed) from the
Dagster UI, then open its output metadata.

This asset lists every MLflow fold run tagged `cv_role=fold` — i.e. every model any experiment has
trained — as a metadata table: `run_id`, `experiment_name`, `fold_id`, `started_at`. It writes
nothing to disk and has no dependents; materialise it any time you want to refresh the list before
picking a champion. The pick itself is still by eye: metrics vary per experiment, so there is no
single sort key that could automate "which run is best" — cross-check candidates against the
leaderboard in the MLflow UI (`uv run mlflow ui --gunicorn-opts "--workers 1"` →
`http://localhost:5000`; see
[ML Experimentation: Viewing results in the MLflow UI](../ml_experimentation/dagster-workflow.md#viewing-results-in-the-mlflow-ui)).

Copy the `run_id` of the fold you want to promote.

## Step 2 — Materialise `promoted_model`

**Trigger:** Dagster UI → Assets → `promoted_model` → "Materialize". Fill in
`PromotedModelConfig.mlflow_run_id` with the run id from step 1.

**What the asset does:**

1. Downloads that run's saved model artifacts from MLflow
   (`ml_core._production_helpers.fetch_model_artifacts`), stamps a `promotion.json`
   (`mlflow_run_id`, `promoted_at`), and atomically replaces the directory at
   `Settings.production_model_path` (`data/production_model/` by default) with the new artifacts.
2. Reads back the new `meta.json` and reports `model_class`, `experiment_name`, and
   `n_trained_time_series` as output metadata, so you can confirm the right model landed.

Promotion is a Dagster materialisation rather than a bare script, so every promotion is recorded
in Dagster's run history — an audit trail for free. Re-promoting with a different `mlflow_run_id`
**replaces** the directory outright; artifacts from the previous champion are not merged in.

## Step 3 — Let the schedule run, or materialise `live_forecasts` by hand

Once a model is promoted, `live_forecasts` produces a new forecast automatically every 6 hours —
at 00:00, 06:00, 12:00, and 18:00 UTC — via `live_forecasts_schedule`. `power_time_series_and_metadata`
(a separate, hourly-scheduled job `live_forecasts` depends on but isn't ordered against) is
itself scheduled 5 minutes *before* each hour so that hour's pull has landed by the time
`live_forecasts` ticks — a cheap mitigation, not a guarantee; see
`power_time_series_and_metadata_schedule`'s docstring (`defs/schedules.py`) for the more rigorous
fix still to explore. This needs the Dagster daemon running (part of `dg dev`; see
[Prerequisites](#prerequisites-a-persistent-dagster-instance) above) to fire on time.

To materialise one 6-hourly slot yourself — e.g. right after promoting a model, so you don't have
to wait for the next tick, or to inspect a specific partition — go to Dagster UI → Assets →
`live_forecasts` → select the partition → fill in `LiveForecastsConfig.availability_mode` →
"Materialize".

**Partition semantics — read this before picking a partition.** A partition key names the *start*
of its 6-hour window; the forecast's `power_fcst_init_time` (init time) is that window's *end*,
six hours later. For example, partition key `"2026-07-04-00:00"` covers the window from
2026-07-04 00:00 UTC (the key itself) to 2026-07-04 06:00 UTC (the next tick), so *that*
partition's `power_fcst_init_time` is 2026-07-04 06:00 UTC — not at the midnight the key names.
(The `live_forecasts` asset's own docstring has the full explanation.)

`availability_mode` controls which NWP run is used:

| `availability_mode` | When to use | Behaviour |
|---|---|---|
| `"live"` (default; what the schedule always uses) | Materialising the current slot, right after it ticks | Joins the **freshest NWP run actually present** with `nwp_init_time <= power_fcst_init_time` — no modelled delay, since reality already constrains the table to genuinely published runs |
| `"replay"` | Backfilling a missed or historical slot | Joins the freshest run with `nwp_init_time <= power_fcst_init_time − nwp_publication_delay_hours`, reconstructing what was genuinely *available* at that historical `power_fcst_init_time` (without the delay, a replay would leak NWP runs that only landed afterwards) |

**What the asset does:**

1. Loads the production model from `Settings.production_model_path` via a plain disk `load` —
   the concrete forecaster class is reconstructed from `meta.json`'s `model_class` field. Raises
   if the model has no trained time series (re-promote first).
2. Resolves which NWP `init_time` to join against via `select_nwp_init_time` and
   `availability_mode` (table above).
3. Builds the power spine (`build_live_power_frame`), covering 15 days of history (long enough
   for the longest power-lag feature any production model uses) and the 14-day forecast horizon.
4. Engineers features in **single-run mode** (an explicit `power_fcst_init_time`) across **all
   NWP ensemble members**, then drops join artefacts: history rows
   (`valid_time <= power_fcst_init_time`) and any row the ensemble join missed.
5. Forecasts exactly `forecaster.trained_time_series_ids` — never today's eligibility set (the
   train==predict population invariant) — and writes to `power_forecasts` with `fold_id="live"`.
6. **Idempotent write:** overwrites only this partition's rows (matching `experiment_name`,
   `fold_id="live"`, and this `power_fcst_init_time`), so re-running or replaying a slot never
   duplicates rows or disturbs any other partition.
7. Logs **nothing** to MLflow — a 6-hourly production run is not an experiment. Live performance
   will be tracked by production monitoring once it exists (not yet implemented — see
   [roadmap: Production monitoring](../roadmap/live-service.md#production-monitoring)).

## Inspecting a live forecast

Use `plot_power_forecast_job` — the same job
[ML Experimentation](../ml_experimentation/dagster-workflow.md#inspecting-a-forecast-plot_power_forecast_job)
uses to inspect backtest forecasts — with `fold_id="live"`:

| Field | Example | Notes |
|---|---|---|
| `experiment_name` | `"xgboost_v1"` | The promoted model's experiment name (see `promoted_model`'s output metadata from step 2) |
| `fold_id` | `"live"` | Always `"live"` for a production forecast |
| `power_fcst_init_time` | `"2026-07-04T06:00:00+00:00"` | The partition's forecast init time — see the partition-semantics note in step 3 |
| `time_series_ids` | `[1, 2, 3, 4]` | Between 1 and 4 ids; each drawn on its own panel |

## Backfilling a missed slot

If a scheduled tick was missed — the daemon was down, or a run failed — materialise that
partition from the Dagster UI with `LiveForecastsConfig.availability_mode="replay"` (see the table
in [step 3](#step-3-let-the-schedule-run-or-materialise-live_forecasts-by-hand)). This reconstructs
what NWP data was genuinely available at that historical `power_fcst_init_time`, rather than
accidentally using data that only arrived afterwards.

This is also the shape of the "local dress rehearsal" for
[#208](https://github.com/openclimatefix/nged-substation-forecast/issues/208): run `dg dev`
continuously for several days, confirm a forecast lands every 6 hours, then deliberately kill a
run mid-flight and backfill the missed partition in replay mode — confirming no duplicate rows
land in `power_forecasts` either way.
