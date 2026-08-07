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
(a separate, hourly-scheduled job `live_forecasts` depends on but is deliberately not ordered
against) is itself scheduled 5 minutes *before* each hour so that hour's pull has landed by the
time `live_forecasts` ticks. That offset is an optimisation for freshness, not a precondition: if
the pull fails or runs long, the forecast still goes out on time against whatever is already on
disk, and records how stale that input was — see
[Inherent Stability, rule 11](../design-philosophy/inherent-stability.md#the-rules). This needs the
Dagster daemon running (see
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

## Rolling back to the previous champion

**Trigger:** a promoted model turns out to be worse in production than the one it replaced.

Rollback is promotion run backwards: re-materialise `promoted_model` (step 2) with the *previous*
champion's `mlflow_run_id`. Because promotion **replaces** the artifact directory outright rather
than merging into it, there is nothing to undo — the previous model returns exactly as it was, and
the next scheduled tick picks it up.

The one thing to get right is finding the run id you are rolling back *to*. The current
`promotion.json` records only the champion serving right now, and it is overwritten on every
promotion, so the history lives in **Dagster's run history**: open Assets → `promoted_model` →
"Runs", and read `PromotedModelConfig.mlflow_run_id` off the previous successful materialisation.
Keeping a note of the outgoing run id *before* you promote is cheaper than looking it up under
pressure afterwards.

On AWS there is the same extra leg as for promotion: rebuild and push the image, then point the
task definition at the tag — see
[Redeploying a new champion model](aws.md#redeploying-a-new-champion-model). Rolling back to a tag
that was previously deployed avoids the rebuild.

This path works today but is not yet one command, which is what
[T3.2](../design-philosophy/engineering-hypotheses.md#h3-one-click-promotion-and-one-click-rollback) measures.

## Degraded input data — NWP feed down, or telemetry stalled

**Trigger:** an asset check reports a warning, or `live_forecasts` fails.

The service is designed to keep answering as its inputs degrade rather than to stop — the
reasoning, and the ladder of degradation states, are in
[Inherent Stability](../design-philosophy/inherent-stability.md). None of the situations below is a
same-day emergency; all are next-business-day fixes.

**Reading the freshness check.** `power_data_is_fresh` runs against
`power_time_series_and_metadata`, is **non-blocking** and **WARN**-severity, and reports on
*on-disk data recency* rather than on whether the asset materialised. It flags any time series
whose most recent reading is more than 24 hours old, and its metadata carries a table of the late
series with `last_seen` and `hours_late`. A warning therefore never stops forecasts being
produced; it tells you which feed to chase. A handful of persistently-late series is usually a
decommissioned or renamed substation rather than an outage — check the roster before escalating.

**Reading the NWP check.** `nwp_has_no_unexpected_nulls` runs inside the `ecmwf_ens` asset, from
the frame already in memory, and is likewise non-blocking WARN. A small scattered fraction of
nulls in the three de-accumulated variables is *expected* and is not a fault — see
[Known ECMWF ENS Data-Quality Issues](../architecture/ecmwf-ens-known-issues.md). A whole slice of
nulls is rejected at ingest by `Nwp.validate`, which means the day's run simply does not land: the
symptom you will actually see is a **missed run**, not corrupt data.

**When a daily NWP run is missing.** We ingest one ECMWF run per day (the 00Z run, downloaded at
08:30 UTC), so healthy NWP is between 12 and 30 hours old depending on which 6-hourly slot is
forecasting. Raw age is not a fault signal; a missed *run* is. If one or two runs are missed,
`live_forecasts` continues normally against the freshest run on disk — it selects "the freshest run
present as of the forecast init time", so no intervention is needed beyond fixing the download.
Materialise the missed `ecmwf_ens` partitions once the feed recovers.

**When `live_forecasts` fails outright.** Today, if no NWP run on disk is recent enough to cover
the forecast horizon (roughly 15 days of staleness), the asset raises rather than producing a
degraded forecast, and NGED receive nothing for that slot. That is a known divergence from the
design principle, tracked for v0.5; until it is fixed, treat it as an alert to restore the NWP feed
and then backfill the missed slots in replay mode (see
[Backfilling a missed slot](#backfilling-a-missed-slot)).

**When the model fails to load.** A raise complaining that the promoted model has no trained time
series is *not* a data outage — it is a promotion bug, and it is meant to fail loudly. Re-promote
(step 2), or roll back
([above](#rolling-back-to-the-previous-champion)).

**Log the intervention.** Every entry above that needed a human is a data point for
[T1.1](../design-philosophy/engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself): append
a row to the [intervention log](intervention-log.md) with the date, the trigger, the cause, the
minutes spent, and whether this runbook covered it. A gap in this page is itself the finding.

## Inspecting a live forecast

Use the `view_forecasts` dashboard app — the same app
[ML Experimentation](../ml_experimentation/dagster-workflow.md#inspecting-a-forecast-the-view_forecasts-dashboard)
uses to inspect backtest forecasts — on your laptop:

```bash
uv run marimo edit packages/dashboard/view_forecasts.py
```

Switch the **Data source** radio to `s3` (needs the git-ignored `packages/dashboard/.env.s3`
holding the read-only laptop credentials — see the
[Configuration reference](setup.md#at-a-glance-which-settings-for-which-environment)), keep the
**Fold** dropdown on `live`, then pick a time series and a forecast date/run. The plot shows every
forecast ensemble member against the observed power, from 24 hours before the forecast init time
to 14 days after it.

The dashboard reads whichever tables it needs directly via their `Settings` paths and renders on
demand, so it works identically whether each path is local or `s3://` — nothing runs on AWS, and
nothing is written anywhere.

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
