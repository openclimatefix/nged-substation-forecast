---
name: data-download
description: >-
  How to write a one-off bulk-download script in this repo so a crash partway through costs one
  chunk, not the whole run: checkpoint every chunk to disk as soon as it is fetched, resume by
  skipping whatever is already cached, measure one chunk before committing to the rest, look up a
  provider's real parameter names before submitting a request, and treat a missing not-yet-published
  chunk as expected rather than fatal. Load before writing or resuming any script under
  `studies/*/fetch_*.py` or similar that downloads more than a handful of requests, especially a
  multi-month or multi-year backfill from an external weather or climate archive.
---

# Writing a resumable bulk-download script

**A download script that only writes its output once, at the end of its loop, loses everything to
the one chunk that fails.** `fetch_icon_dream.py` (issue #841) ran 85 months of whole-domain GRIB
downloads and cropping, one month at a time, and only called `write_parquet` after every month had
succeeded. The 85th month 404'd — DWD had not published it yet — the script crashed, and the first
84 months' work vanished with it, because none of it had touched disk.

## Checkpoint every chunk to disk immediately, and resume by skipping what is cached

Write each chunk's result to its own file the moment it is fetched, before moving to the next chunk:

```python
month_cache_dir = output_dir / "_month_cache" / variable
month_cache_dir.mkdir(parents=True, exist_ok=True)
for year_month in year_months:
    month_path = month_cache_dir / f"{year_month}.parquet"
    if month_path.exists():
        print(f"{variable} {year_month}: already cached, skipping")
        continue
    frame = fetch_one_month(...)
    frame.write_parquet(month_path)
combined = pl.concat([pl.read_parquet(p) for p in sorted(month_cache_dir.glob("*.parquet"))])
```

Build the final combined file by reading back whatever is on disk, not from an in-memory list
accumulated during the loop — that way a second run of the same command after a crash finishes the
job rather than redoing it, and the combine step itself is safe to re-run. The per-chunk cache
directory is a working area, not the product: it can be deleted once the combined file is written,
or left in place cheaply since it is small relative to the whole-domain source it was cropped from.

**The chunk size that makes sense to checkpoint at is whatever the source is naturally divided
into** — a month for DWD's monthly GRIB files, a year for a yearly archive, a request-sized time
block for an API with its own chunking limit (see CDS below). Checkpointing at a finer grain than
the source's own division buys nothing; checkpointing at a coarser grain (writing only once per
several months) reintroduces the same loss-on-crash problem at a smaller scale.

## Treat "not yet published" as expected, not as a crash

The newest chunk in a backfill running up to "now" may not exist yet — a monthly archive is often
still assembling this month's file. Catch the specific error the provider returns for "not found"
(an HTTP 404, typically) and skip that one chunk with a message, rather than crashing the whole run
or catching every exception indiscriminately:

```python
try:
    frame = fetch_one_month(...)
except requests.exceptions.HTTPError as error:
    if error.response is not None and error.response.status_code == 404:
        print(f"{variable} {year_month}: not published yet (404), skipping")
        continue
    raise
```

Only the specific "not published" signal is swallowed; every other failure still aborts the run and
surfaces the traceback. Silently catching every exception would hide a real bug behind a stream of
"skipping" messages.

## Measure one chunk before committing to the rest

Before looping over years of a product, fetch and measure a single month (or whatever the natural
chunk is): its size, its variables, its served lead, its time convention. Extrapolate the total from
that measurement, and stop to report if the estimate is far from what was expected, rather than
running the full backfill on an assumption. This is what caught ICON-DREAM-EU's `WS` field running
roughly 30x larger than its solar siblings before any bulk fetch was attempted (10 model levels
stacked into one file, and an instantaneous field that compresses far worse than solar's night-zeroed
one) — the difference between transient whole-domain bytes passing through (large, and typically not
worth worrying about if network transfer is unmetered) and the small cropped file that survives (what
actually matters for disk) is worth measuring explicitly, not assuming.

## Look up the provider's real parameter names before submitting a request

Guessing a variable or dataset name and submitting a request to find out it was wrong wastes a queue
slot on a service that queues jobs seriously (CDS can take minutes to hours to even start a job).
Most catalogue-style APIs publish their valid parameter values somewhere queryable without spending a
request: CDS's own catalogue serves a `form.json` per dataset (follow the `form` link on
`https://cds.climate.copernicus.eu/api/catalogue/v1/collections/<dataset-id>`) listing every valid
`variable`, `height_level`, `product_type`, and so on. Fetch and read that before writing the
request, rather than trial-and-error against the retrieval API itself.

## CDS-specific: one job at a time per account, no server-side area crop on every dataset

The Climate Data Store runs one job per account at a time and queues the rest, so a script submitting
several chunk requests concurrently (a small thread pool) keeps the queue full without buying real
parallelism — see `studies/beam_diffuse_split/fetch_era5.py` for the pattern. Some CDS datasets (CERRA
is one) do not support the `area` request parameter on their native projected grid, so every request
returns the whole domain regardless of how it is asked; crop locally after download and delete the
whole-domain file immediately, the same crop-then-delete shape `fetch_icon_dream.py` uses for DWD's
GRIB.

## Round continuous values to a small significand before writing, same as production NWP storage

`delta_store.nwp` rounds every continuous weather variable to 13 significand bits before writing, so
that zstd finds repetition in what would otherwise be incompressible float noise — see
`delta_store.precision.round_to_significand_bits` for the technique and its bounded relative-error
guarantee. Apply the same call to a study download's continuous columns before `write_parquet`,
after casting to `Float32`:

```python
frame = frame.with_columns(pl.col(unit_column).cast(pl.Float32))
frame = frame.with_columns(round_to_significand_bits(pl.col(unit_column), keep_bits=13))
```

## Run a long background download unbuffered, so its log is actually readable while it runs

`python3 script.py > log 2>&1 &` under `nohup` buffers stdout by default, so `print` statements sit
in a buffer rather than reaching the log file until the buffer fills or the process exits — a log
that looks frozen for many minutes while the process is, in fact, making steady progress. Run with
`python3 -u` (or set `PYTHONUNBUFFERED=1`) for any script launched this way, so `tail -f` on the log
shows real-time progress rather than nothing until the process ends.

## Anonymisation: never let a private box or coordinate reach a log, a filename, or a report

Where a download is cropped to a private area (the NGED trial-area box, generator coordinates), keep
every reusable helper working in derived quantities only — a point count, a cell-index array, a grid
of coordinates sent straight to a request — and never returning or printing the raw bounds. See
`studies/weather_downloads/paths.py`'s `TrialAreaBox` for the pattern: the class's methods return
what a caller needs to build a request, never the `lat_min`/`lon_min` etc. themselves, so handling it
carelessly cannot leak the box into a log line, a filename, or a report back to the user.
