---
name: data-download
description: >-
  How to write a resumable bulk-download script in this repo — for a multi-month or multi-year
  backfill from an external weather or climate archive especially — so a crash partway through costs
  one chunk, not the whole run: checkpoint every chunk to disk as soon as it is fetched, resume by
  skipping whatever is already cached, measure one chunk before committing to the rest, look up a
  provider's real parameter names before submitting a request, and size each chunk to the provider's
  own constraints. Load before writing or resuming any bulk-download script (e.g.
  `studies/*/fetch_*.py`) that makes more than a handful of requests.
---

# Writing a resumable bulk-download script

**A download script that writes its output only once, at the end of its loop, loses every chunk
already fetched when one chunk fails.** A DWD ICON-DREAM-EU backfill script ran 85 months of
whole-domain GRIB downloads and cropping, one month at a time, and called `write_parquet` only after
every month had succeeded. The 85th month 404'd — DWD had not published it yet — the script
crashed, and the first 84 months' work vanished with it, because none of it had touched disk. This
skill turns that incident into a general checklist; see the `study` skill for where a download's
output lives (`data/studies/weather/<PRODUCT>/`) and for the CDS and anonymisation rules it already
owns, which this skill does not repeat.

## Checkpoint every chunk to disk immediately, and resume by skipping what is cached

**Write each chunk's result to its own file the moment it is fetched, atomically, before moving to
the next chunk.** A plain `frame.write_parquet(month_path)` leaves a truncated, unreadable file if
the process is killed mid-write — and the resume check then treats that truncated file as "already
cached" and skips it forever. Write to a `.partial` suffix and rename on success, the same pattern
`studies/beam_diffuse_split/fetch_era5.py` already uses for its own chunks:

```python
import polars as pl

month_cache_dir = output_dir / "_month_cache" / variable
month_cache_dir.mkdir(parents=True, exist_ok=True)
for year_month in year_months:
    month_path = month_cache_dir / f"{year_month}.parquet"
    if month_path.exists():
        print(f"{variable} {year_month}: already cached, skipping")
        continue
    frame = fetch_one_month(...)  # raises on failure; see "missing chunk" below
    partial = month_path.with_suffix(".parquet.partial")
    frame.write_parquet(partial)
    partial.rename(month_path)

# Combine only the months this run asked for, not every file the cache directory happens to
# hold — an old run's leftover chunks from a different date range must not silently join in.
cached_paths = [p for ym in year_months if (p := month_cache_dir / f"{ym}.parquet").exists()]
if not cached_paths:
    raise RuntimeError(f"{variable}: no month succeeded across {year_months[0]}-{year_months[-1]}")
combined = pl.concat([pl.read_parquet(path) for path in cached_paths])
```

Build the final combined file by reading back whatever the *current run's requested range* has
cached, not by globbing the whole cache directory — a directory reused across differently-scoped
runs otherwise mixes in months nobody asked for this time. Re-running the same command after a crash
then finishes the job rather than redoing it, and the combine step itself is safe to re-run. The
per-chunk cache directory is a working area, not the product: delete it once the combined file is
written, or leave it in place, since the cache is small relative to the whole-domain source it was
cropped from.

**The chunk size that makes sense to checkpoint at depends on what limits the provider, not just
what the source happens to be divided into.** A plain HTTPS archive served as one file per month
(DWD's GRIB) checkpoints naturally at a month. A queued API instead prices a request by field count
and queues one job per account at a time — the Climate Data Store's ERA5 endpoint caps a request at
121,000 fields, and queue *wait*, not data volume, sets the wall-clock time, so
`studies/beam_diffuse_split/fetch_era5.py` requests six months per chunk rather than one: 15
half-yearly requests spend almost no time queued where 85 monthly ones spend hours. Checkpointing
still happens per chunk either way; only the size of a chunk changes. A dataset's `costing_api`
catalogue link (see below) returns a request's field count before it is submitted, which is how to
check a chunk stays under a queued API's limit.

## Treat a documented "not yet published" signal as expected, not as a crash — but only for the newest chunk

**The newest chunk in a backfill running up to "now" may not exist yet**, because the source archive
is often still assembling it. Catch the provider's own "missing" signal for that one case and skip
with a message, rather than crashing the whole run:

```python
import requests

try:
    frame = fetch_one_month(...)
except requests.exceptions.HTTPError as error:
    if error.response is not None and error.response.status_code == 404:
        print(f"{variable} {year_month}: not published yet (404), skipping")
        continue
    raise
```

Two things this pattern gets wrong if copied blindly. First, **the missing signal is provider- and
library-specific** — `requests.HTTPError` only fires where the calling code raises for a 4xx/5xx
status; another provider may return 403 on an unauthorised listing, return 200 with an HTML error
page, or route through a client library (`cdsapi`, `httpx`, plain `urllib`) that raises its own
exception type. Confirm the real "missing" signal during the one-chunk measurement below, rather
than assuming a 404. Second, **only skip the signal for a chunk expected to be newer than the
provider's publication lag** — a bare "except 404: skip" over every chunk turns a mistyped URL, or a
real gap mid-archive, into a silent, empty-looking success rather than a failure. Treat the signal on
any chunk earlier than the trailing few as fatal, and record every skipped chunk in the lineage note.

**A script fetching several chunks concurrently should log each failure and continue, rather than
abort.** `fetch_era5.py`'s thread pool catches any exception per chunk, logs it, lets the other
in-flight chunks finish, and exits non-zero if any chunk failed — appropriate where a pool is already
resilient to one slow or failed request among several. A single sequential loop, where resuming costs
almost nothing thanks to the cache above, can instead abort on an unexpected error and surface the
traceback immediately, then resume from the cache once the cause is fixed.

## Measure one chunk before committing to the rest

**Fetch and measure a single chunk before looping over years of a product**: its size, its
variables, its served lead, and its time convention. Extrapolate the total from that measurement,
and stop to report if the estimate is far from what was expected, rather than running the full
backfill on an assumption. Measuring one month of ICON-DREAM-EU's `WS` field this way caught it
running about 35x larger than its solar siblings (6.97 GB against roughly 0.2 GB) before that
field's own bulk fetch began. Ten stacked model levels account for roughly a tenfold factor on their
own; the remaining difference has not been measured and is reported as unexplained rather than
guessed at. Keep the transient whole-domain bytes passing through separate from the small cropped
file that survives: the former is often not worth worrying about if network transfer is unmetered,
but the latter is what actually matters for disk.

## Look up the provider's real parameter names before submitting a request

**Guessing a variable or dataset name and submitting a request to find out it was wrong wastes a
queue slot** on a service that queues jobs seriously — CDS can take minutes to hours to even start a
job. CDS's own catalogue serves everything needed to build a valid request without spending one:
`https://cds.climate.copernicus.eu/api/catalogue/v1/collections/<dataset-id>` returns a `links[]`
list carrying `rel: "form"` (every field's valid values — variable names, height levels, product
types), `rel: "constraints"` (which combinations of those values are actually valid together, since
a field can list a value the constraints then rule out for a given request), and `rel: "costing_api"`
(a request's field count, checked before submission). A dataset whose `form.json` has no `area`
field at all is how to tell in advance that it cannot be cropped server-side — CERRA's
single-levels and height-levels forms both lack one, where ERA5's has it — so the
whole-domain-then-crop pattern below is needed from the start, not discovered after a request comes
back uncropped.

## Whole-domain-then-crop pattern, for a source with no server-side area subsetting

Where a provider either lacks an `area` parameter (some CDS datasets, e.g. CERRA) or serves an
unstructured grid with no lat/lon axes in the file itself (DWD's ICON-DREAM-EU), download the file
whole, crop it locally to the target box, and delete the whole-domain file immediately — never keep
it. Stage the whole-domain file somewhere with real disk behind it, not under `/tmp`: `/tmp` on this
machine is tmpfs, so a multi-gigabyte scratch file there consumes RAM rather than disk and can
exhaust the tmpfs quota mid-run.

## Round continuous *value* columns to a small significand before writing — never a coordinate, an ID, or a time

`delta_store.nwp` rounds every continuous weather variable to 13 significand bits
(`delta_store.nwp.NWP_SIGNIFICAND_BITS`) before writing, so zstd finds repetition in what would
otherwise be incompressible float noise — see `delta_store.precision.round_to_significand_bits` for
the technique and its bounded relative-error guarantee. Apply the same call to a download's
continuous value columns before `write_parquet`, after casting to `Float32`:

```python
from delta_store.nwp import NWP_SIGNIFICAND_BITS
from delta_store.precision import round_to_significand_bits

frame = frame.with_columns(pl.col(unit_column).cast(pl.Float32))
frame = frame.with_columns(
    round_to_significand_bits(pl.col(unit_column), keep_bits=NWP_SIGNIFICAND_BITS)
)
```

Two columns must never go through this rounding. **A latitude or longitude column**: at 13
significand bits a value near 55° carries up to roughly 700 m of rounding error, enough to move a
point into the wrong grid cell — round only a physical measurement, never a coordinate, an ID, or a
timestamp. **A field accumulated or averaged since some earlier reference time** (ECMWF's `ssrd` in
a forecast product, ICON's `ASWDIR_S`): rounding an accumulated series before it is de-accumulated
into per-step values turns the differencing step's night-time zeros into rounding noise.
De-accumulate first, round after.

## Run a long background download unbuffered, through `uv run`, so its log is actually readable

**Output redirected to a file is block-buffered by Python whenever stdout is not a terminal**,
`nohup` or no `nohup` — a `print` statement can sit unwritten for many minutes even though the
process is making steady progress, so a log that looks frozen is not evidence the process has
stalled. Launch with `PYTHONUNBUFFERED=1 uv run python script.py > log 2>&1 &` (or `uv run python -u
script.py`), not a bare `python3 script.py`, which runs outside the workspace's `uv` virtual
environment and will fail to import any workspace package (`delta_store`, `contracts`, and so on) a
checkpointed downloader is likely to need.

## Two traps from this repo's own conventions worth restating here

**Two sessions running the same fetch script share one `data/` folder**, since a git worktree is
backed by one `data/` directory, so two sessions resuming the same download write to the same
`_month_cache` directory — confirm only one session is running a given fetch before relying on its
cache. **A private trial-area box or generator coordinate must never reach a log line, a filename, or
a printed value.** A helper that returns only derived quantities (a point count, a cell-index array,
a grid of coordinates built straight into a request) keeps a caller from leaking the box by accident
when it handles that return value carelessly, but a caller that reads the box's raw bounds directly
is still responsible for never printing or logging them — and a returned coordinate grid is exactly
as sensitive as the box itself, not already safe to print just because it came out of a helper.
