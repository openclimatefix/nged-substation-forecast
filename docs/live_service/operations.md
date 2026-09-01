# Operating the live service

How to get a champion model producing live, 6-hourly forecasts, and how to backfill a slot that
was missed.

There are two layers. **Promotion** (step 1–2) is manual and occasional — you do it once per
champion model, whenever a better candidate clears the [ML Experimentation](../ml_experimentation/index.md)
leaderboard. **Inference** (step 3) is automatic — once a model is promoted, the 6-hourly schedule
keeps producing forecasts from it with no further action, as long as Dagster's daemon is running.

Production inference has **zero dependency on MLflow at runtime**: the promoted model is a plain
directory on disk, loaded with a plain `save`/`load` round trip — no run id or tracking-server call
on the hot path. See
[roadmap: Production model artifacts](../roadmap/live-service.md#production-model-artifacts) for
why this was chosen over fetching the champion model from MLflow dynamically at runtime.

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

Either UI shows every asset, most of which are nothing to do with running the service. Paste
`tag:layer=production` into the asset-selection box to cut the view down to the four that produce
the forecasts: `power_time_series_and_metadata`, `h3_grid_weights`, `ecmwf_ens`, and
`live_forecasts`. The promotion assets used in steps 1–2 below are deliberately *not* in that
selection: they need MLflow, which the deployment does not reach, so you run them yourself —
today from your laptop, as step 2 says — whichever environment serves the forecasts.
`tag:layer=research` is the rest.

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

A run trained before a feature was renamed in the code cannot be served, and nothing in the table
marks it. Step 2 refuses it rather than letting you discover that at the next 6-hourly tick.

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
   (`ml_core.production_helpers.fetch_model_artifacts`) into a temporary directory.
2. Checks the downloaded model's saved config against the running code, and **fails the
   materialisation if this code could not load that model** — because a feature name no longer
   parses (the offending feature is named), or because `model_params` carry a hyper-parameter this
   code no longer declares. The check runs before anything is written, so a refused promotion
   leaves the previous champion in place and still serving. Re-train the model against the current
   code and promote that run; never hand-edit `meta.json`, because the boosters on disk were
   trained under the config it records.
3. Stamps a `promotion.json` (`mlflow_run_id`, `promoted_at`) and atomically replaces the directory
   at `Settings.production_model_path` (`data/production_model/` by default) with the new artifacts.
4. Reads back the new `meta.json` and reports `model_class`, `experiment_name`, and
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
the pull fails or runs long, the forecast still goes out on time against whatever telemetry is
already on disk — see
[Inherent Stability → The rules](../design-philosophy/inherent-stability.md#the-rules). This needs the
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
   if the model has no trained time series, or if the running code cannot rebuild its saved config
   — a `selected_features` name it cannot parse, or a `model_params` key it no longer declares.
   Promotion (step 2 above) applies that same check, so it fires here only when the code changed
   after the champion was promoted; either way, re-promote from a run trained against the current
   code rather than hand-editing what is on disk.
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
[T3.2, rollback effort](../design-philosophy/engineering-hypotheses.md#h3-one-click-promotion-and-one-click-rollback)
measures.

## Degraded input data — NWP feed down, or telemetry stalled

**Trigger:** an asset check reports a warning, or `live_forecasts` fails.

The service is designed to keep answering as its inputs degrade rather than to stop — the
reasoning, and the ladder of degradation states, are in
[Inherent Stability](../design-philosophy/inherent-stability.md). None of the situations below is a
same-day emergency; all are next-business-day fixes.

**What a Sentry event's tags tell you first.** `fault_category:run_failed` means a scheduled job
failed outright, so that cycle did not run; an event without it is a degradation the service kept
forecasting through. The full routing is in
[Setting up Sentry telemetry](sentry.md#turn-it-on-in-production). Two things send nothing at all: a
transient failure reading NGED's bucket, which `power_time_series_and_metadata` retries twice
before reporting, and a run you cancelled yourself.

**Reading the freshness check.** `power_data_is_fresh` runs against
`power_time_series_and_metadata`, is **non-blocking** and **WARN**-severity, and reports on
*on-disk data recency* rather than on whether the asset materialised. It flags any time series
whose most recent reading is more than 24 hours old, and its metadata carries a table of the late
series with `last_seen` and `hours_late`. A warning therefore never stops forecasts being
produced; it tells you which feed to chase. A handful of persistently-late series is usually a
decommissioned or renamed substation rather than an outage — check the roster before escalating.

That table is capped at 50 rows
([why](../architecture/production-deployment.md#warn-on-stale-power-data-with-a-dagster-asset-check)).
**Read `n_late`, not the table's length**, to see how big the stall is: `n_late` counts every late
series, and `n_late_listed` tells you how many rows the table holds, so the two agreeing means you
are looking at every late series and the two differing means the list is truncated. The same pair
appears on the live-forecast check as `n_time_series_missing` and `n_time_series_missing_listed`.

Mind the order when it *is* truncated: never-reported series come first, then the most-stale ones,
so a roster with more than 50 never-reported series fills the table and no stale series appears in
it at all. Read `n_stale` and `n_never_reported` — never truncated — before concluding from the
table that nothing has gone stale. All three counts, and `n_series_total` beside them, describe the
series the check is *watching*: the silenced series below are excluded from every one of them.

**Silencing a series we know is dead.** `_KNOWN_DEAD_TIME_SERIES_IDS` in
`src/nged_substation_forecast/defs/checks.py` lists the `time_series_id`s the check ignores, so a
broken monitor cannot hold it yellow for ever
([why](../architecture/production-deployment.md#silence-the-series-we-already-know-are-dead)). Add
an id, with a comment saying why, then commit, rebuild the image and redeploy. Removing an id starts
the warnings again. Either edit is an intervention worth an
[intervention-log](intervention-log.md) row under `upstream-outage`: the data is stuck and a human
had to act.

Two descriptions come from that list. `Ignoring N known-dead time series: 33.` is appended to
every run, green or yellow, so the silencing cannot be quietly forgotten — read `n_silenced` and
`silenced_time_series_ids` for the same thing in the metadata. `Reporting again, so no longer dead:
33.` means a silenced series has sent data within the threshold, which fails the check until you
delete it from the list; the check does not remove it for you, and the yellow lasts only while the
series keeps reporting, so a series that revives for an afternoon and dies again leaves no trace.

`n_silenced` counts the ids you listed, not the ids that were actually withheld, so an id that
matches no series still appears: that is how a mistyped id shows itself rather than vanishing.

One description flags a different failure mode from all the others. `Could not evaluate power-data
freshness: …` is the check reporting that it could not read its own inputs — suspect the object
store, or a `metadata.parquet` left half-written by a killed process — not that the feed has
stalled. The named exception is in the description, the full traceback is in the run's logs, and
the exception is also sent to Sentry tagged `asset_check:power_data_is_fresh`, so this one reaches
you without your watching the Checks view. The check degrades this way on purpose rather than
raising, so the hourly ingest keeps running; nothing is known about staleness while it persists, so
treat it as "unknown", not "healthy".

**Reading a failed roster upsert.** `metadata_upsert_failed` in
`power_time_series_and_metadata`'s run metadata means the `TimeSeriesMetadata` roster upsert raised
and was swallowed so the power write could go ahead, and it also reaches Sentry tagged
`degraded_asset:power_time_series_and_metadata`. The run **succeeds** by design: the roster is
derived data that NGED re-delivers, and the power time series is not, so a roster fault must not
stall the ingest until an operator intervenes. The roster is left unchanged and the next run that
finds new files retries it, but *that run's* metadata change is lost, because the power rows have
landed and `select_new_rows` will not offer those files again. Read the traceback in the run's logs —
an off-contract roster after a schema change and a bug in our own code both land here, and both want
a fix rather than a re-run.

The 6-hourly forecasts are unaffected while this persists, however long it persists: `live_forecasts`
locates each series from the promoted model's own frozen copy of the roster rows it trained against,
never from the roster itself. What a stalled upsert loses is the metadata change, which matters at
the next training run.

**Reading the NWP check.** `nwp_has_no_unexpected_nulls` runs inside the `ecmwf_ens` asset, from
the frame already in memory, and is likewise non-blocking WARN. Nulls in the three de-accumulated
variables are *expected* and are not a fault — see
[Known ECMWF ENS Data-Quality Issues](../architecture/ecmwf-ens-known-issues.md).

**It counts two populations, and the keys say which is which.** The `nwp_grid_point` keys count
the raw 0.25° grid Dynamical.org sent us, before aggregation; the `h3_cell` keys count the cells we
store afterwards. They answer different questions and are not comparable as rates.

- **"Is the feed broken, and since when?"** — read `null_nwp_grid_point_fraction`, and
  `affected_nwp_variables` (or the `per_nwp_variable` table, for each variable's own counts) for
  which variable to name in a mail to Dynamical.org. This is the
  number to take to the provider, because it is free of our H3 resolution and aggregation policy,
  both of which move a cell count without anything upstream having changed. It is published on
  every materialisation as well as on the check, so plot it on the asset timeline: a single run's
  value means little, and the trend across runs is the signal.
- **"How much did the model lose?"** — read `n_null_h3_cells`. The aggregation renormalises each
  cell over the grid points that supplied a value, so it absorbs most upstream scatter and this
  count stays small even when the feed is badly corrupt. Only this side drives the check's
  pass/fail.

A run with a non-zero grid-point fraction and zero null cells is a corrupt run the aggregation
absorbed, not a broken one. The check's
`n_whole_null_h3_slices` metadata is the one worth a second look: those are
`(variable, ensemble_member, valid_time)` slices where the field arrived wholesale empty. A handful
is not a fault and the run is kept regardless, but a count that climbs run after run is worth
raising with Dynamical.org. Only a variable empty in *every* slice is rejected at ingest by
`Nwp.validate`, and even then `ecmwf_ens` retries first — so the symptom of that case is a
**missed run** at the end of a long-running job, not corrupt data.

**This check is the only place a badly-degraded run is reported, and the run is already on disk by
the time you read it.** Everything short of a wholly-empty variable lands, so
`n_whole_null_h3_slices` is not merely informational — it is the sole signal distinguishing a run
that lost two slices from one that lost nearly all of them, and both land looking equally green.
Nothing downstream consumes it: no training filter, no metric, and no Sentry alert. That last one
is an omission rather than a limit — `power_data_is_fresh` warns *and* sends a Sentry event, so a
non-blocking check plainly can; this one simply does not. Correcting a badly-degraded run means
re-materialising its partition by hand, once the upstream data is fixed. So if this count is ever
large rather than a handful, treat it as an incident to act on deliberately — the pipeline will not
act on it for you. Making a large count escalate is tracked in
[issue #501](https://github.com/openclimatefix/nged-substation-forecast/issues/501).

**Reading the instantaneous-variable check.** `nwp_instantaneous_variables_have_no_nulls` counts the
raw grid again, over the nine variables that are never legitimately null, and fails on a single null
grid point. It carries the same `nwp_grid_point` keys, and counts lead-0, which the de-accumulated
check excludes. **A red result here is a mail to Dynamical.org, not a re-run**: the run has landed,
the aggregation absorbed the nulls before they reached a stored cell, and there is nothing to fix on
our side. Quote `affected_nwp_variables` and the `per_nwp_variable` counts. This has never yet fired
on real data — an instantaneous variable's nulls have so far only arrived as whole-step dropouts,
which fail ingest outright and show up as a missed run instead.

**Reading the NWP completeness check.** `nwp_run_is_complete` also runs inside `ecmwf_ens`, also
non-blocking WARN, and asks the other question: did the whole run arrive? Its description names
the missing ensemble members and the missing lead times in hours, and its metadata carries the
observed-versus-expected member, step, cell, and row counts. **The run has already landed when
this warns** — a short run is kept, because partial NWP forecasts better than falling back on
yesterday's run. The action is to chase Dynamical.org, and to re-materialise the partition once
they republish the complete run.

**All three NWP checks share one description that flags a different failure mode from all the
others**, just as `power_data_is_fresh` does above. `Could not assess the ingested NWP run: …` says
the assessment itself failed, not that the run is degraded — so it appears on all three checks at
once, and the shape metadata (`n_ensemble_members` and the rest) and the grid-point metadata
(`null_nwp_grid_point_fraction` and `n_null_nwp_grid_points`) are absent from that materialisation:
there is no report to read them from. Treat the corruption rate as unknown for that run, not zero,
and mind the gap when reading the trend. The run still lands. One Sentry event is sent, tagged
`asset_check:nwp_has_no_unexpected_nulls` whichever assessment raised.

**Re-materialising a partition that has already landed replaces it.** `write_nwp` overwrites the
`(nwp_model_id, init_time)` partition it is handed rather than appending to it, so re-running the
partition after Dynamical republishes a run swaps the short copy for the complete copy. Wait until
they actually have: a re-run against a run that is still incomplete replaces the good rows with the
short rows. The same holds for a partition whose run *failed* — re-running replaces whatever
landed, so there is no need to inspect the table first.

Two things a re-run does cost:

- **Do not re-materialise a partition while another materialisation of that same partition is
  running.** The two writes contend and the loser fails with delta-rs' `CommitFailedError`. That
  costs a run, not the table, and re-running afterwards is safe. Partitions for *different* dates
  do not contend at all, so a backfill alongside the daily schedule is fine.
- **The superseded rows stay on disk, and the default `vacuum` will not clear them.** Delta marks
  the old parquet files as removed rather than deleting them, so replacing a V1 partition leaves
  ~7.24M dead rows — about 137 MiB — behind. `DeltaTable.vacuum()` reports deleting those files and
  does not: it re-encodes the `init_time` partition directory's already percent-encoded name,
  deletes at a path that does not exist, and counts the resulting `NotFound` as a deletion
  ([issue #593](https://github.com/openclimatefix/nged-substation-forecast/issues/593)). Reads are
  unaffected — every reader goes through the transaction log. `vacuum(full=True)` does reclaim the
  space, because it deletes at paths taken from a storage listing rather than from the tombstones.
  Run it only when nothing needs an older version of the table: it removes a superseded file once
  that file is older than the retention window, counting from when the file was *written* rather
  than from when it was replaced, so time travel to the version before a re-materialisation can go
  immediately instead of seven days later.

Every materialisation whose completeness assessment succeeded also publishes `n_ensemble_members`,
`n_valid_times`, `n_h3_cells`, and the `valid_time` range as metadata, so the Dagster UI timeline
shows slow drift in the upstream dataset before it becomes a warning.

**Reading the live-forecast check.** `live_forecasts_are_healthy` runs against `live_forecasts`
after each 6-hourly materialisation, is likewise non-blocking WARN, and reads the forecasts back
*off disk* — so it answers "did this slot really land usable rows?", which the run's own green tick
does not. It goes yellow when the slot wrote no rows at all, when any row carries a null, NaN or
infinite `power_fcst`, when a row targets a `valid_time` at or before its own init time, when the
forecast reaches less than half the 14-day horizon we ask for, when the slot forecast fewer time
series than the promoted model was trained on, or when the NWP feed is behind. Its metadata carries
each of those counts plus `nwp_init_time_on_disk` and `nwp_init_time_expected`, so the description
usually tells you which of the two halves — the write or the weather — is at fault. Note that the
check only reports on slots where the asset *succeeded*; a slot that raised fails the run instead,
and Sentry reports that — see *When `live_forecasts` fails outright* below.

**When a daily NWP run is missing.** We ingest one ECMWF run per day (the 00Z run, downloaded at
08:30 UTC), so healthy NWP is between 12 and 30 hours old depending on which 6-hourly slot is
forecasting. Raw age is not a fault signal; a missed *run* is, and
`live_forecasts_are_healthy`'s `n_missed_nwp_runs` is that count. It measures how far *behind* the
feed is — how many daily runs separate the freshest run on disk from the freshest that ought to
exist — so it is **zero in every healthy slot**, whichever slot it is, and it clears as soon as a
fresh run lands (an older hole left behind in history no longer degrades anything, because the
forecast uses the freshest run). Read it as follows.

- **0** — healthy, whatever the NWP age happens to be. Do not act on the age.
- **1** — the feed is one daily run behind. `live_forecasts` continues normally against the
  freshest run on disk, so this needs no intervention beyond fixing the download; materialise the
  missed `ecmwf_ens` partition once the feed recovers, and the count clears at the next slot.
- **2 or more** — the download has been failing for more than a day. Still not an emergency, but
  chase it the same business day: forecast quality decays as the run ages, and past roughly 15 days
  of staleness the run stops covering the forecast horizon altogether — see *When
  `live_forecasts` fails outright* below.
- **absent from the metadata** — the NWP table holds no run at or before this slot, so the count
  has no finite answer; the description says so. Expect `live_forecasts` itself to be failing too.

One count deserves a caveat: the check demands the day's run by 14:00 UTC, so a download that
fails today first shows up at the 18:00 slot rather than the 12:00 one — deliberately, for the
reasoning in [Production Deployment — Design: the missed daily NWP run
count](../architecture/production-deployment.md#read-the-live-forecast-back-off-disk-with-a-second-asset-check).

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

**Log the intervention.** Every entry above that needed a human is a data point for the
[T1.1 operability test](../design-philosophy/engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself):
append a row to the [intervention log](intervention-log.md) with the date, the trigger, the cause, the
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

Press **Reload data** to pick up runs written since you opened the app: the date and run selectors
move to the newest run, and your chosen time series is kept.

The dashboard reads whichever tables it needs directly via their `Settings` paths and renders on
demand, so it works identically whether each path is local or `s3://` — nothing runs on AWS, and
nothing is written anywhere.

## Backfilling a missed slot

If a scheduled tick was missed — the daemon was down, or a run failed — materialise that
partition from the Dagster UI with `LiveForecastsConfig.availability_mode="replay"` (see the table
in [step 3](#step-3-let-the-schedule-run-or-materialise-live_forecasts-by-hand)). This reconstructs
what NWP data was genuinely available at that historical `power_fcst_init_time`, rather than
accidentally using data that only arrived afterwards.

Backfilling in replay mode is also the shape of the "local dress rehearsal" for
[#208](https://github.com/openclimatefix/nged-substation-forecast/issues/208): run `dg dev`
continuously for several days, confirm a forecast lands every 6 hours. Then deliberately kill a
run mid-flight and backfill the missed partition in replay mode, confirming no duplicate rows
land in `power_forecasts` either way.
