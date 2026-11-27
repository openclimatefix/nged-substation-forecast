# Operating the live service

How to drive the running live service day to day: the schedules, what the asset checks and Sentry
events mean when an input degrades, how to read and update the daily Primary substation forecast,
and how to backfill a missed partition.

The live service ingests the power telemetry of National Grid Electricity Distribution (NGED) and
the ensemble numerical weather prediction (NWP) from the European Centre for Medium-Range Weather
Forecasts (ECMWF), cleans the telemetry, and checks both inputs. Once a day, a neural network then
forecasts the demand of 16 Primary substations for the next 14 days. The production runtime has
**no dependency on MLflow**: the deployment reaches
no tracking server. The [cross-validation assets](../cross-validation/index.md) that do need MLflow
run on a researcher's laptop.

---

## Prerequisites — a running Dagster instance

Everything on this page is driven from the Dagster UI, and is the same whichever environment the
stack runs in — bring one up first:

- **Locally on a laptop** — [Getting started on your
  laptop](../getting-started.md#running-the-live-service-on-your-laptop); the UI is at
  `http://localhost:3000` and runs execute on your machine.
- **Deployed on AWS** — [Setting up the live service on AWS](aws.md); the UI is at
  `http://nged-forecast-ctrl:3000` over Tailscale, and each run executes on an ephemeral Fargate
  task. To get your laptop onto the tailnet and reach that UI, see [Connecting to the AWS control
  plane](connecting.md).

Either UI shows every asset, most of which are nothing to do with running the service. Paste
`tag:layer=production` into the asset-selection box to cut the view down to the five assets the
deployment runs: `power_time_series_and_metadata`, `clean_nged_power_data`, `h3_grid_weights`,
`ecmwf_ens`, and `live_primary_substation_forecasts`. `tag:layer=research` is the rest: the
cross-validation assets, which need MLflow and so run on your laptop whichever environment runs the
ingest.

## The schedules

Three schedules run the whole recurring pipeline. All three need the Dagster daemon running (see
[Prerequisites](#prerequisites-a-running-dagster-instance) above) to fire on time:

- `power_time_series_and_metadata_schedule` fires at :55 past every hour. Its job runs
  `power_time_series_and_metadata`, which pulls NGED's new telemetry files, and then
  `clean_nged_power_data`, which rebuilds the cleaned power table when its inputs have changed.
- `ecmwf_ens_schedule` fires daily at 10:30 UTC and materialises today's `ecmwf_ens` partition, the
  00Z ECMWF ensemble run from Dynamical.org. The asset retries while the run is not yet published.
- `live_primary_substation_forecasts_schedule` fires daily at 12:00 UTC and materialises today's
  `live_primary_substation_forecasts` partition, which forecasts from the 00Z run that `ecmwf_ens`
  has normally written by then. [The daily Primary substation
  forecast](#the-daily-primary-substation-forecast) below covers that asset.

Nothing downstream waits on another job's run status: an asset that reads the ingested tables runs
against whatever is already on disk — see [Inherent Stability → The
rules](../design-philosophy/inherent-stability.md#the-rules).

To materialise an asset yourself, go to Dagster UI → Assets → the asset → select the partition, if
the asset has partitions → "Materialize".

## Degraded input data — NWP feed down, or telemetry stalled

**Trigger:** an asset check reports a warning, or an ingest job fails.

The service is designed to keep answering as its inputs degrade rather than to stop — the reasoning,
and the ladder of degradation states, are in [Inherent
Stability](../design-philosophy/inherent-stability.md). None of the situations below is a same-day
emergency; all are next-business-day fixes.

**What a Sentry event's tags tell you first.** `fault_category:run_failed` means a scheduled job
failed outright, so that cycle did not run. An event without that tag is a degradation the service
kept running through, or a `retrying_asset` warning that an asset failed and will retry. The full
routing is in [Setting up Sentry telemetry](sentry.md#turn-it-on-in-production). Two things send
nothing at all: a transient failure reading NGED's bucket, which `power_time_series_and_metadata`
retries twice before reporting, and a run you cancelled yourself.

**Reading the freshness check.** `power_data_is_fresh` runs against
`power_time_series_and_metadata`, is **non-blocking** and **WARN**-severity, and reports on *on-disk
data recency* rather than on whether the asset materialised. It flags any time series whose most
recent reading is more than 24 hours old, and its metadata carries a table of the late series with
`last_seen` and `hours_late`. A warning therefore never stops the ingest. The warning tells you
which feed to chase. A handful of persistently-late series is usually a decommissioned or renamed
substation rather than an outage — check the metadata table before escalating.

That table is capped at 50 rows
([why](../architecture/production-deployment.md#warn-on-stale-power-data-with-a-dagster-asset-check)).
**Read `n_late`, not the table's length**, to see how big the stall is: `n_late` counts every late
series, and `n_late_listed` tells you how many rows the table holds, so the two agreeing means you
are looking at every late series and the two differing means the list is truncated.

Mind the order when it *is* truncated: never-reported series come first, then the most-stale ones,
so more than 50 never-reported series fill the table and no stale series appear in it at all.
Read `n_stale` and `n_never_reported` — never truncated — before concluding from the table that
nothing has gone stale. All three counts, and `n_series_total` beside them, describe the series the
check is *watching*: the silenced series below are excluded from every one of them.

**Silencing a series we know is out of service.** `_SILENCED_TIME_SERIES_IDS` in
`src/nged_substation_forecast/defs/checks.py` lists the `time_series_id`s the check ignores, so an
out-of-service meter cannot hold it yellow for ever
([why](../architecture/production-deployment.md#silence-a-series-that-has-stopped-reporting)). Add
an id, with a comment saying why, then commit, rebuild the image and redeploy. Removing an id starts
the warnings again. Either edit is an intervention worth an [intervention-log](intervention-log.md)
row under `upstream-outage`: the data is stuck and a human had to act.

Two descriptions come from that list. `Ignoring N silenced time series: 33.` is appended to every
run, green or yellow, so the silencing cannot be quietly forgotten — read `n_silenced` and
`silenced_time_series_ids` for the same thing in the metadata. `Reporting again, so remove 33 from
_SILENCED_TIME_SERIES_IDS in defs/checks.py.` means a silenced series has sent data within the
threshold, which fails the check until you delete it from the list; the check does not remove it for
you, and the yellow lasts only while the series keeps reporting, so a series that reports for an
afternoon and then stops again leaves no trace.

**NGED's fault notes.** The check description also carries one sentence naming each series whose
metadata table row has an `information` note, for example `NGED fault notes: 33: "Analogues not
working."`, whether or not the series is silenced. The note is NGED's own free text. Cleaning does
not read it, because a note has no date. The ingest downloads every file once, including a file with
no readings, and takes each series' metadata from its newest file, so a silenced series' note stays
current.

`n_silenced` counts the ids you listed, not the ids that were actually withheld, so an id that
matches no series still appears: that is how a mistyped id shows itself rather than vanishing.

One description flags a different failure mode from all the others. `Could not evaluate power-data
freshness: …` is the check reporting that it could not read its own inputs — suspect the object
store, or a `metadata.parquet` left half-written by a killed process — not that the feed has
stalled. The named exception is in the description, the full traceback is in the run's logs, and the
exception is also sent to Sentry tagged `asset_check:power_data_is_fresh`, so this one reaches you
without your watching the Checks view. The check degrades this way on purpose rather than raising,
so the hourly ingest keeps running; nothing is known about staleness while it persists, so treat it
as "unknown", not "healthy".

**On a new deployment, the `cleaned_power_time_series` table does not exist until
`clean_nged_power_data` has run.** Every reader of the cleaned table fails at the read until then.
The hourly ingest job runs `clean_nged_power_data` straight after the first ingest, so materialise
`clean_nged_power_data` by hand only when you need the cleaned table before that first hourly run
finishes.

**Reading a failed cleaning run.** `clean_nged_power_data` rebuilds `cleaned_power_time_series` from
the raw power table when NGED has delivered new data, a cleaning rule has changed, or the
`TimeSeriesMetadata` table has changed. Otherwise the asset returns at once with `skipped: True` in
its run metadata. When `clean_nged_power_data` fails, the run fails and reports to Sentry. Every
reader carries on with the last good cleaned table, and the next hourly
run tries again. Read the traceback in the run's logs: a failing cleaning rule and a metadata
parquet file that no longer matches the `TimeSeriesMetadata` contract both need a fix, not a re-run.

**A cleaning job that stops running without failing shows up only in the Checks view.** The
`cleaned_power_keeps_up_with_raw` check on `power_time_series_and_metadata` warns when the cleaned
table was built from a raw table 2 or more commits (about 6 to 12 hours of NGED deliveries) older
than the current raw table, was built from a different raw table, or is absent. The check is the
only signal for that failure, and sends nothing to Sentry. Fix the cause, then materialise
`clean_nged_power_data`. Setting its `force` run config rebuilds the cleaned table even when nothing
has changed.

**Reading a failed vacuum.** `vacuum_failed: True` in `clean_nged_power_data`'s run metadata means
the new cleaned table was written and only deleting the superseded files failed. A vacuum failure
also reaches Sentry tagged `degraded_asset:clean_nged_power_data`.

**Reading a failed metadata table upsert.** `metadata_upsert_failed` in
`power_time_series_and_metadata`'s run metadata means the `TimeSeriesMetadata` table upsert raised
and was swallowed so the power write could go ahead, and it also reaches Sentry tagged
`degraded_asset:power_time_series_and_metadata`. The run **succeeds** by design: the metadata table
is derived data, and the power time series is not, so a metadata table fault must not stall the
ingest until an operator intervenes. The metadata table is left unchanged, and *that run's* metadata
change is lost until each affected series publishes its next file, about 6 hours later. The
downloaded-files list still records the files, because holding it back would make every later hour
download a growing backlog. Read the traceback in the run's logs — an off-contract metadata table
after a schema change and a bug in our own code both land here, and both want a fix rather than a
re-run.

A stalled upsert loses only the change to the `TimeSeriesMetadata` table, which matters at the next
training run.

**Reading the NWP check.** `nwp_has_no_unexpected_nulls` runs inside the `ecmwf_ens` asset, from the
frame already in memory, and is likewise non-blocking WARN. Nulls in the three de-accumulated
variables are *expected* and are not a fault — see [Known ECMWF ENS Data-Quality
Issues](../architecture/ecmwf-ens-known-issues.md).

**It counts two populations, and the keys say which is which.** The `nwp_grid_point` keys count the
raw 0.25° grid Dynamical.org sent us, before aggregation; the `h3_cell` keys count the cells we
store afterwards. They answer different questions and are not comparable as rates.

- **"Is the feed broken, and since when?"** — read `null_nwp_grid_point_fraction`, and
  `affected_nwp_variables` (or the `per_nwp_variable` table, for each variable's own counts) for
  which variable to name in a mail to Dynamical.org. This is the number to take to the provider,
  because it is free of our H3 resolution and aggregation policy, both of which move a cell count
  without anything upstream having changed. It is published on every materialisation as well as on
  the check, so plot it on the asset timeline: a single run's value means little, and the trend
  across runs is the signal.
- **"How much did the model lose?"** — read `n_null_h3_cells`. The aggregation renormalises each
  cell over the grid points that supplied a value, so it absorbs most upstream scatter and this
  count stays small even when the feed is badly corrupt. Only this side drives the check's
  pass/fail.

A run with a non-zero grid-point fraction and zero null cells is a corrupt run the aggregation
absorbed, not a broken one. The check's `n_whole_null_h3_slices` metadata is the one worth a second
look: those are `(variable, ensemble_member, valid_time)` slices where the field arrived wholesale
empty. A handful is not a fault and the run is kept regardless, but a count that climbs run after
run is worth raising with Dynamical.org. Only a variable empty in *every* slice is rejected at
ingest by `Nwp.validate`, and even then `ecmwf_ens` retries first for a recent run — so the symptom
of that case is a **missed run** at the end of a long-running job, not corrupt data.

**This check is the only place a badly-degraded run is reported, and the run is already on disk by
the time you read it.** Everything short of a wholly-empty variable lands, so
`n_whole_null_h3_slices` is not merely informational — it is the sole signal distinguishing a run
that lost two slices from one that lost nearly all of them, and both land looking equally green.
Nothing downstream consumes it: no training filter, no metric, and no Sentry alert. That last one is
an omission rather than a limit — `power_data_is_fresh` warns *and* sends a Sentry event, so a
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
non-blocking WARN, and asks the other question: did the whole run arrive? Its description names the
missing ensemble members and the missing lead times in hours, and its metadata carries the
observed-versus-expected member, step, cell, and row counts. **The run has already landed when this
warns** — a short run is kept, because partial NWP forecasts better than falling back on yesterday's
run. The action is to chase Dynamical.org, and to re-materialise the partition once they republish
the complete run.

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
short rows. The same holds for a partition whose run *failed* — re-running replaces whatever landed,
so there is no need to inspect the table first.

Two things a re-run does cost:

- **Do not re-materialise a partition while another materialisation of that same partition is
  running.** The two writes contend and the loser fails with delta-rs' `CommitFailedError`. That
  costs a run, not the table, and re-running afterwards is safe. Partitions for *different* dates do
  not contend at all, so a backfill alongside the daily schedule is fine.
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

**When a daily NWP run is missing.** We ingest one ECMWF run per day: the 00Z run, downloaded from
10:30 UTC. `ecmwf_ens` sends a `retrying_asset` warning to Sentry on its first failed attempt. If
the run is still missing when the retries run out, about 4 hours later, `ecmwf_ens_job` fails and
the failure reaches Sentry tagged `fault_category:run_failed`. The day's Primary substation forecast
then writes nothing (see [The daily Primary substation
forecast](#the-daily-primary-substation-forecast)). Chase Dynamical.org, then materialise the missed
`ecmwf_ens` partition once the feed recovers, followed by the same day's
`live_primary_substation_forecasts` partition (see [Backfilling a missed
partition](#backfilling-a-missed-partition)).

**Log the intervention.** Every entry above that needed a human is a data point for the [T1.1
operability
test](../design-philosophy/engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself): append a
row to the [intervention log](intervention-log.md) with the date, the trigger, the cause, the
minutes spent, and whether this runbook covered it. A gap in this page is itself the finding.

## Running the NGED ingest — first run, rebuilds, and the downloaded-files list

**Trigger:** a new deployment, a rebuilt table, a Sentry event naming the downloaded-files list, or
a run failing with `NgedFileParseError`.

**The ingest downloads each file in NGED's bucket once.** `power_time_series_and_metadata` keeps a
downloaded-files list, `list_of_downloaded_files.parquet`, beside `metadata.parquet`. The list holds
the path and `LastModified` of every file in the bucket listing that the previous run processed in
full. Each run downloads only the files whose path and `LastModified` the list lacks, appends their
readings, and then writes the whole listing as the new list. A file that NGED rewrites gets a new
`LastModified`, so the ingest downloads it again.

**Run the ingest once by hand on a new deployment.** With no list, the first run downloads every
file in the bucket: about 22,000 files (33 series, 165 days of history). Time one request first to
estimate the run. Expect a few minutes, because the ingest keeps at most 32 requests open at a time.
The first run also parses every historical file that has no readings, so a malformed old file fails
the run (see the last paragraph of this section).

**Set the `NGED_INGEST` pool's limit to 1 before the first run.** Run `dagster instance concurrency
set NGED_INGEST 1`, because Dagster's YAML takes no per-pool limits and its `default_limit` would
otherwise apply. The pool then queues any scheduled run behind the first run, so the first run and a
scheduled run cannot append the same rows.

**Rebuild a table by deleting it.** The ingest counts the list as empty, and downloads every file
again, whenever the list, the `power_time_series` Delta table, or `metadata.parquet` does not exist.
Deleting `metadata.parquet` therefore rebuilds the metadata table from the whole bucket on the next
run. Moving the `power_time_series` table aside rebuilds the power table the same way.

**Delete the list too when you put an older or different `power_time_series` table at its path.**
The list then records files as loaded whose readings the restored table lacks, and the ingest cannot
tell. Deleting `list_of_downloaded_files.parquet` makes the next run download every file.

**Force one file to download again by removing its row from the list.** Read
`list_of_downloaded_files.parquet` with `polars.read_parquet`, filter out the row whose `path` is
that NGED file's key, and write the list back with `write_parquet`. Before appending, the ingest
drops every reading already in the `power_time_series` table, so a repeat download adds nothing
unless NGED changed a reading's time.

**A `DownloadedFilesError` means the list is damaged, and the ingest stops until you delete the
list.** The error message, which Sentry shows too, names the list's path. The ingest stops rather
than reading a damaged list as empty, because an empty list would download the whole bucket. The next
run after you delete the list downloads every file, and the ingest drops the readings already on
disk.

**Delete the list after any change to its columns.** The old list would otherwise fail validation and
stop the ingest with `DownloadedFilesError`.

**A failed list write is reported and does not fail the run.** Sentry then shows an event tagged
`degraded_asset:power_time_series_and_metadata` with the downloaded-files list's path in its message:
the readings landed, and only the list write failed. The next run downloads the same files again, and
the ingest drops their readings.

**A malformed file, or a file that breaks the `TimeSeriesMetadata` contract, fails the run and
records nothing.** The run fails with a `NgedFileParseError` naming the file's path, and is not
retried. The same file is selected every hour, so the ingest stalls until someone fixes the cause. No
readings are lost: after the fix, the next run ingests the whole backlog.

## The daily Primary substation forecast

**Trigger:** a Sentry event tagged `degraded_asset:live_primary_substation_forecasts`, a missed
check-in on the `live-primary-substation-forecasts` cron monitor, or a new monthly bundle to deploy.

**`live_primary_substation_forecasts` writes one 14-day forecast per day for the 16 Primary
substations.** Each run writes 671 half-hourly rows per Primary to the `power_forecasts` table, with
`power_fcst_init_time` at 12:00 UTC on the partition's day and valid times from 12:30 UTC that day
to 11:30 UTC 14 days later. The rows carry `experiment_name="primary_substation_demand_forecaster"`,
`power_fcst_model_name="substation_demand_forecaster"`, and `fold_id="live"`, and
`power_fcst_model_version` holds the deployed bundle's `model_release`. The 13 quantile columns are
in megawatts, or in megavolt-amperes for the eight Primaries metered in MVA, whose quantiles are
clamped at zero. Re-materialising a partition replaces that `power_fcst_init_time`'s rows, so a
re-run never duplicates a forecast.

**The asset reads three inputs, and none of them comes from MLflow:**

- the day's 00Z ECMWF ensemble run and the 13 daily runs before it, from the `nwp` table that
  `ecmwf_ens` writes;
- the Primaries' cleaned readings from the past 9 days, from the `cleaned_power_time_series` table,
  for the lags of one day and one week;
- the step-5 neural network's bundle, a directory that the production image carries at
  `/app/primary_substation_bundle`. The bundle holds each seed's weights, the standardisers,
  the 16 Primaries with their capacities, and the recipe's constants. The Primaries and their
  capacities come from the bundle, not from the `TimeSeriesMetadata` table.

**Read `n_rows` in the run's metadata first.** A run that wrote a forecast shows `n_rows` 10736 (671
rows for each of the 16 Primaries) and `n_time_series` 16. `n_rows` 0 means the day's ENS run never
arrived, or arrived with no rows in any of the Primaries' H3 cells, and the day has no forecast. The
metadata also names the bundle that made the forecast, in `model_release`, `research_commit`, and
`refit_month`, and lists any history run that was missing in `missing_history_runs`.

**Three metadata fields show how much telemetry the lags had.** `day_lag_missing_fraction` and
`week_lag_missing_fraction` are the shares of the forecast's input rows, across all the Primaries,
whose day lag or week lag has no cleaned reading. `n_primaries_without_readings` counts the
Primaries with no cleaned reading at all in the 7 days the lags read, which end 1 day before the
09:00 UTC native origin.

**A run with `n_rows` 0 still shows as a green, materialised partition in the Dagster UI.** A
backfill launched over the "missing" partitions therefore skips that day. To find such days, open
the asset's partitions in the UI and read the `n_rows` of each materialisation, or filter the
asset's materialisation events on `n_rows` 0, then materialise those partitions by name.

**A missed check-in means no forecast landed today.** On a successful run for today's partition, the
asset sends a check-in to the Sentry cron monitor `live-primary-substation-forecasts`. Sentry raises
the missed check-in at 17:00 UTC if none has arrived. The causes include a dead daemon or box, a
failed run, and a day whose ENS run never arrived. Look in the Dagster UI for the day's run: no run
at all points at the daemon or the box, and a run with `n_rows` 0 points at the NWP feed. Setting up
the monitor is in [Setting up Sentry telemetry](sentry.md#the-daily-forecast-check-in).

**Each degradation reaches Sentry tagged `degraded_asset:live_primary_substation_forecasts`, and its
message names the ENS runs or the bundle involved.** Sentry groups the events by kind, so a fault
that repeats every day stays one Sentry issue. None of the six kinds stops a forecast that could
still be made:

| Kind | What happened | What the asset did | What to do |
|---|---|---|---|
| `day_run_missing` | By the last retry, at about 16:00 UTC, the day's 00Z run was not in the `nwp` table, or had no rows in any of the Primaries' H3 cells | Wrote nothing and sent no check-in, so the day has no forecast | Recover the `ecmwf_ens` partition, then materialise the same day's `live_primary_substation_forecasts` partition |
| `history_runs_missing` | One or more of the 13 history runs is absent | Computed the thermal-lag and solar-gain columns from the history that remains, and took the lags' expected-weather rows from the next-freshest run | Materialise the missing `ecmwf_ens` partitions if Dynamical.org still serves them |
| `day_run_incomplete` | The day's run has, in some but not all of the Primaries' H3 cells, no rows; or, in some H3 cell, fewer than 51 members at any lead, a missing lead, a last lead short of 360 hours, or null radiation after lead 0 | Forecast from the values present: a missing lead is interpolated from its neighbours, fewer members give the mean and spread of the members present, and an input still missing takes its training mean | Chase Dynamical.org, then re-materialise both partitions once the complete run is republished |
| `expected_weather_uncovered` | No ENS run covers some of the half-hours the day and week lags read their expected weather from, which happens when several history runs are missing | Read zeros for those expected-weather rows; `n_uncovered_expected_weather_rows` in the run metadata counts them | As for `history_runs_missing` |
| `bundle_stale` | The day's run falls two or more months after the deployed bundle's refit month, or one month after it and after the 7th of the month, so the forecast's valid times sit past the bundle's last offset knot, where the offset is held flat. The first 7 days of a month are exempt, because the month's refit trains on ERA5, which publishes about 5 days late | Forecast with the old bundle | [Install the current month's bundle](#installing-a-new-monthly-bundle) |
| `primaries_without_readings` | One or more of the bundle's Primaries has no cleaned reading in the 7 days the lags read, which end 1 day before the native origin; the message names their time series | Forecast those Primaries with the day and week lags read as missing, so their forecasts carry no recent level | Check `power_data_is_fresh` and the NGED feed for those time series (see [Degraded input data](#degraded-input-data-nwp-feed-down-or-telemetry-stalled)) |

While the day's run is absent or has none of the Primaries' cells, the asset retries every 30
minutes, 8 times, and sends one
`retrying_asset` warning on the first retry. A `degraded_asset` event that carries a real stack
trace, rather than one of the six messages above, means that the code checking the ENS runs
raised. The forecast still ran, and the stack trace shows the bug.

**A single missing lag is not reported, but a Primary with no readings at all is.** When a Primary
has no cleaned reading at its day or week lag, the neural network reads the missing-lag encoding it
was trained with, and the forecast moves towards the forecast it makes without lags. Stale
telemetry already has its own check, `power_data_is_fresh` (see [Degraded input
data](#degraded-input-data-nwp-feed-down-or-telemetry-stalled)), and the two missing-lag fractions
in the run metadata show how far the lags were affected. A Primary with no reading in the whole
7-day lag window is reported as `primaries_without_readings`.

**A failed run is our own fault, so it fails and reaches Sentry tagged `fault_category:run_failed`.**
The asset raises on a bundle that is missing, has an unknown format version, or was trained on
Copernicus Atmosphere Monitoring Service (CAMS) values that the service cannot compute, on a
partition whose 12:00 UTC origin is before the bundle was created (see [Backfilling a missed
partition](#backfilling-a-missed-partition)), and on output that fails the `PowerForecast`
contract. Read the traceback, fix the image, and redeploy.

### Installing a new monthly bundle

**The research repository exports one bundle per monthly refit, and a new bundle reaches the service
only through a new image.** The image tag names both the code and the bundle, as
`<git commit>-r<model_release>-<research commit>-<refit month>`, so a refit with no code change
still gets a new tag and the previous image stays in the Elastic Container Registry (ECR) for a rollback.

1. **Run the golden tests against the new bundle.** The tests compare every stage of the forecast
   with the fixtures the bundle ships, and read the `nwp` and `cleaned_power_time_series` tables
   under `DATA_PATH_INTERNAL`, so point `DATA_PATH_INTERNAL` at a data folder that holds the
   fixtures' dates:

    ```bash
    PRIMARY_SUBSTATION_BUNDLE_PATH=<bundle-dir> \
        uv run pytest tests/test_substation_demand_forecaster_golden.py
    ```

    `<bundle-dir>` is the directory that holds the bundle's `manifest.json`. Without
    `PRIMARY_SUBSTATION_BUNDLE_PATH`, the golden tests read the bundle at the default path,
    `primary_substation_bundle` under the local artifacts folder, and skip when that directory
    holds no fixtures, so a run that reports only skips has tested nothing.

2. **Build, verify, push, and deploy the image** with the bundle directory as the scripts' one
   argument, following [Redeploying new code](aws.md#redeploying-new-code). The build copies the
   bundle into the image, leaving out the fixtures and the PyTorch checkpoints. The build then
   checks every file's SHA-256 hash against the manifest and loads the bundle the way the asset
   does. A bundle with an unknown format version, a recipe the code does not implement, or CAMS
   columns the service cannot compute fails this check, before the image is pushed.

3. **Check the next day's run metadata.** `model_release`, `research_commit`, and `refit_month`
   should name the new bundle.

**A re-run of a past partition uses the bundle in the image deployed at the time of the re-run.**
The re-run therefore replaces the rows of the bundle that made the original forecast with rows from
the new bundle, under the new `power_fcst_model_version`. A re-run of a partition whose 12:00 UTC
origin is before the deployed bundle was created fails instead and writes nothing, because that
bundle did not serve the day.

## Inspecting forecasts

Use the `view_forecasts` dashboard app — the same app that [Running the cross-validation pipeline
end-to-end](../cross-validation/dagster-workflow.md#inspecting-a-forecast-the-view_forecasts-dashboard)
uses to inspect backtest forecasts — on your laptop:

```bash
uv run marimo edit packages/dashboard/view_forecasts.py
```

Switch the **Data source** radio to `s3` (needs the git-ignored `packages/dashboard/.env.s3` holding
the read-only laptop credentials — see the [Configuration
reference](setup.md#at-a-glance-which-settings-for-which-environment)), pick a fold in the **Fold**
dropdown, then pick a time series and a forecast date/run. The **Fold** dropdown selects `live` when
the `power_forecasts` table holds live rows and the first fold otherwise. The `live` fold holds the
daily Primary substation forecasts, one run per day at 12:00 UTC. The plot
shows the forecast median inside three grey quantile bands (p1 to p99, p5 to p95, and p20 to p80)
against the observed power, from 24 hours before the forecast init time to 14 days after it.

Press **Reload data** to pick up runs written since you opened the app: the date and run selectors
move to the newest run, and your chosen time series is kept.

The dashboard reads whichever tables it needs directly via their `Settings` paths and renders on
demand, so it works identically whether each path is local or `s3://` — nothing runs on AWS, and
nothing is written anywhere.

## Backfilling a missed partition

If a scheduled tick was missed — the daemon was down, or a run failed — the two ingest assets and
the forecast recover differently:

- **`power_time_series_and_metadata`** needs nothing. The next hourly run downloads every file the
  downloaded-files list lacks, so it picks up whatever the missed run would have.
- **`ecmwf_ens`** has one partition per day. Materialise each missed day's partition from the
  Dagster UI (Assets → `ecmwf_ens` → select the partition → "Materialize"), or launch a backfill
  over a range of partitions. Re-materialising a partition that has already landed replaces it (see
  [Degraded input data](#degraded-input-data-nwp-feed-down-or-telemetry-stalled)).
- **`live_primary_substation_forecasts`** has the same daily partitions as `ecmwf_ens`.
  Materialise a missed day's partition once that day's `ecmwf_ens` partition has landed, in the
  same way. A backfill sends no Sentry check-in, because only today's partition sends one. A
  backfill whose 00Z run is more than 36 hours old and absent from the `nwp` table does not retry:
  the backfill reports `day_run_missing` at once and writes nothing. A partition whose earlier run
  wrote `n_rows` 0 counts as materialised, so select it by name rather than through a backfill of
  missing partitions.

**A backfill of a day before the deployed bundle was created needs the bundle that was live on
that day.** The asset compares the partition's 12:00 UTC origin with the `created_utc` timestamp in
the bundle's `manifest.json`. It fails an earlier partition with a Dagster `Failure` that names the
partition, the creation time, and the refit month, before it writes any rows or sends a check-in.
The rule covers every day before the refit month, because research exports a bundle only after the
refit month has begun. The rule also covers the first days of the refit month, before the new
bundle was exported, when the previous month's bundle served the live forecast. A forecast from the
newer bundle is not the forecast the service served on that day. To backfill the day, deploy the
image built with the bundle that was live then, which ECR still holds, materialise the partition,
and redeploy the current image.

**A backfilled forecast sees the same weather as the live run, and its lags can differ.** A
backfill reads only the ENS runs initialised up to its own day's 00Z run, so its weather inputs are
the inputs the live run had. Its day and week lags read the cleaned power table as it is at the
time of the backfill, which holds readings that NGED delivered late. A Primary whose lag was
missing at 12:00 UTC on the day can therefore have a lag in the backfill, and the backfilled
forecast can differ from the forecast the live run would have made. The backfill also uses the
bundle in the image deployed now (see [Installing a new monthly
bundle](#installing-a-new-monthly-bundle)).
