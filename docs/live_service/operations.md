# Operating the live service

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

## Prerequisites — a running Dagster instance

Everything on this page is driven from the Dagster UI, and is the same whichever environment
the stack runs in — bring one up first:

- **Locally on a laptop** — [Running the whole stack locally](local.md); the UI is at
  `http://localhost:3000` and runs execute on your machine.
- **Deployed on AWS** — [Setting up the live service on AWS](aws.md); the UI is at
  `http://nged-forecast-ctrl:3000` over Tailscale, and each run executes on an ephemeral
  Fargate task. To get your laptop onto the tailnet and reach that UI, see
  [Connecting to the AWS control plane](connecting.md).

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

Promotion (this step and the next) always happens **on your laptop**, whichever environment
serves the forecasts: the candidate models live in the laptop's local MLflow file store, and
the promoted artifacts land in a local directory (`data/production_model/`) that the AWS
deployment bakes into its container image at build time.

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

On the local stack, the next scheduled run picks the new champion up immediately (the asset
loads it from disk). In the AWS deployment there is one more leg: rebuild and push the image,
then point the task definition at the new tag — see
[Setting up the live service on AWS: Redeploying a new champion model](aws.md#redeploying-a-new-champion-model).

## Step 3 — Let the schedule run, or materialise `live_forecasts` by hand

Once a model is promoted, `live_forecasts` produces a new forecast automatically every 6 hours —
at 00:00, 06:00, 12:00, and 18:00 UTC — via `live_forecasts_schedule`. `power_time_series_and_metadata`
(a separate, hourly-scheduled job `live_forecasts` depends on but isn't ordered against) is
itself scheduled 5 minutes *before* each hour so that hour's pull has landed by the time
`live_forecasts` ticks — a cheap mitigation, not a guarantee; see
`power_time_series_and_metadata_schedule`'s docstring (`defs/schedules.py`) for the more rigorous
fix still to explore. This needs the Dagster daemon running (see
[Prerequisites](#prerequisites-a-running-dagster-instance) above) to fire on time.

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

`plot_power_forecast` is a **local-development convenience**: it writes Altair HTML to
`LOCAL_ARTIFACTS_PATH`, which on an ephemeral Fargate task is a disk that is discarded when the
task exits — the file would never be seen. The durable, S3-native (S3 is AWS's object store) way
to look at a deployed service's forecasts is the **dashboard**
(`packages/dashboard/main.py`), which reads whichever
tables it needs directly via their `Settings` paths and renders on demand — point it at the same
`Settings` and it works identically whether each path is local or `s3://` (see
[Configuration reference](setup.md#at-a-glance-which-settings-for-which-environment) for the
read-only laptop credentials).

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
