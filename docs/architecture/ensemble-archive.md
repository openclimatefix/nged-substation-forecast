# The ensemble weather archive

**In brief:** A recorder running on a workstation saves six weather products that their providers
delete within days to a month. The recorder crops each forecast run to Great Britain and its
surrounding seas, and stores it in one Icechunk repository per product. The archive is not yet
published. The recorder is a separate program in the
[`nwp-archivist` repository](https://github.com/openclimatefix/nwp-archivist), and this page
describes what it does today.

## Why the archive exists

**Nobody else keeps these forecasts, so a later study could not otherwise score them.** The German
weather service (DWD) keeps only the last 4 runs of ICON-EU-EPS and the last 8 runs of ICON-D2-EPS,
which is about 24 hours. The Met Office deletes each MOGREPS file about 30 days after writing it.
The study that compares the mean of an ensemble with deterministic products (UKV, ICON-EU, ICON-D2,
and ECMWF ENS) needs whole past runs and every ensemble member. The survey found no other source of
per-run, per-member history for these products: see [What we learnt about
MOGREPS-UK](../roadmap/data-sources.md#what-we-learnt-about-mogreps-uk-on-2026-09-26) and the
[survey of whole-run archives](../background/weather-products-survey.md#which-archives-keep-whole-past-forecast-runs).

**A day without the recorder is lost for good for the DWD products.** DWD's deletion is the reason
the recorder polls every 15 minutes and commits each run once it is complete or its deadline has
passed.

## What is recorded

**The archive holds six products, each cropped to 49.0 to 61.5 °N and 10.0 °W to 3.5 °E.** The box
reaches offshore wind farms. It covers Northern Ireland, the Irish Sea, the Celtic Sea off Cornwall,
the seas west of the Hebrides, Shetland, and the North Sea out to the Dutch and Belgian coasts.
The recorder keeps native grid cells whose centre lies inside the box, and does not interpolate.

| Product | Provider | Members | Runs a day | Horizon | Licence |
|---|---|---|---|---|---|
| ICON-EU-EPS | DWD | 40 | 4 | 120 hours | CC BY 4.0 |
| ICON-D2-EPS | DWD | 20 | 8 | 48 hours | CC BY 4.0 |
| ICON-D2 (deterministic) | DWD | none | 8 | 48 hours | CC BY 4.0 |
| ICON-ART-EU (deterministic) | DWD | none | 4 | 120 hours | CC BY 4.0 |
| MOGREPS-UK | Met Office | 3 | 24 | 126 hours | CC BY-SA 4.0 |
| MOGREPS-Global | Met Office | 18 | 4 | 246 hours | CC BY-SA 4.0 |

**Every product keeps every member and its full published horizon.** The recorder does not thin the
ensembles. MOGREPS-Global keeps its whole 246-hour horizon, although MOGREPS-UK stops at 126 hours,
because the extra lead times cost little to store.

**Every variable is stored as delivered, so the meaning of each field depends on its provider.**
DWD's shortwave radiation (`ASWDIR_S`, `ASWDIFD_S`) is an average since the start of the run. The
Met Office's shortwave files carry no time bounds, so the archive treats them as instantaneous. The
root attributes of each repository repeat this. The variables are:

- **ICON-EU-EPS, ICON-D2-EPS, and ICON-D2:** direct and diffuse shortwave radiation, 2 m
  temperature, 10 m wind components, and total cloud cover.
- **ICON-ART-EU:** direct and diffuse shortwave radiation, 2 m temperature, total cloud cover,
  clear-sky net shortwave (`ASOB_S_CS`), and dust optical depth (`TAOD_DUST`). It has no wind.
- **The same three ICON products:** also the wind components on the model levels that
  bracket 100 m, and the half-level heights (`HHL`) needed to interpolate to 100 m. ICON-EU-EPS
  has only one such level, centred at about 95 m.
- **MOGREPS-UK:** total, direct, and diffuse downward shortwave, screen-level temperature, 10 m
  wind speed and direction, total cloud amount, and wind speed and direction at 100 m.
- **MOGREPS-Global:** the same wind fields and the same three shortwave fields as MOGREPS-UK, plus
  net shortwave. It has no temperature or cloud field.

## Where it runs

**Three systemd user timers on one workstation run the recorder, and nothing runs in the cloud.**
One timer starts the DWD recorder every 15 minutes. The other two start the MOGREPS-UK recorder and
the MOGREPS-Global recorder, each one minute after its previous cycle exits, so that each keeps
backfilling older runs between the arrival of new ones. Each of the three has its own cache
directory. The archive itself is on a local disk.

**The archive is not yet published.** No copy exists outside the workstation.

## How the data is stored

**Each product is one Icechunk repository, and each variable is one Zarr array.** A study opens a
repository and slices one array. The dimensions are `(init_time, member, step, cell)`, or
`(init_time, step, cell)` for a deterministic product. The `cell` dimension is the cropped grid
flattened row by row (for MOGREPS products, the smallest rectangle of grid cells that contains the
box), and the coordinates of the cells are stored once per repository. Icechunk makes each commit
atomic across every array. A reader therefore never sees a half-written run, and a crash before a
commit leaves the previous snapshot readable.

**Every variable shares one padded `step` axis.** The axis is as long as the product's longest
field, in minutes. A variable that has fewer lead times is `NaN` at the positions it lacks. For
example, ICON-D2 publishes shortwave every 15 minutes but its other fields every hour, so its axis
has 193 steps and the hourly fields fill only 49 of them.

**The `init_time` axis is a grid of every possible run, and each run goes to the slot its
initialisation time gives.** The first slot is 2026-01-01 00:00 UTC, and each slot is one run cycle
later. A run committed late, such as a `partial` run at its deadline, therefore lands in order. A
run committed twice lands in the same slot, and a run that never appeared is a slot of `NaN`.

**Status arrays along `init_time` say what each slot holds.** They are `status` (0 for not
archived, 1 for `complete`, 2 for `partial`, 3 for `missing`), `files_expected`, `files_received`,
`archived_at`, `generating_process` (the version of the provider's model, where the file carries
it), and `code_version` (the recorder's version and git commit). The MOGREPS products add a
`realization` array, because the Met Office's number for each member changes from run to run. The
status arrays, not any local file, are the record of what is committed.

**Storage settings keep each commit small.** Values are rounded to 13 significand bits and written
with Zstandard compression and a CRC32C checksum. Manifest splitting starts a new chunk-reference
manifest every 64 init times, so that a commit writes about the same amount whether the repository
holds one week of runs or one year. The root attributes of each repository hold `layout_version`.
A recorder that finds a different layout version stops committing that product.

**The recorder builds each run from a local cache, and then commits once.** Each fetched file is
decoded, cropped, and saved to the cache directory as a small file. The recorder counts a file as
received once it is in the cache. When the run is complete, or its deadline has passed, the
recorder writes the run to the repository member by member, so that each chunk is written once. It
deletes the cached files after the commit succeeds. The cache is a checkpoint, not a record: a
rebuilt cache never causes a run to be committed twice, because the recorder checks the repository's
`status` array first.

## How a run is recorded

**Each cycle checks every run that should exist and fetches the files that are missing.** The
recorder computes the run times in a lookback window, and lists each run's expected files from a
single table of products. It starts a run a fixed delay after initialisation, which is set from
when the provider's files usually appear: 30 minutes for ICON-D2 and ICON-D2-EPS, 2 hours for
ICON-EU-EPS and ICON-ART-EU, 1.75 hours for MOGREPS-UK, and 6.5 hours for MOGREPS-Global. A 404 or
a truncated, undecodable, or mismatched file means only "not yet". The recorder tries that file
again in the next cycle.

**A run is `waiting` until it is complete or past its deadline, and nothing is committed part-way
through.** The states are:

- **`waiting`:** files are still missing and the deadline has not passed. Nothing is committed.
- **`complete`:** every expected file is in the cache. The recorder commits the run.
- **`partial`:** the deadline has passed and some files arrived. The recorder commits what exists,
  with the expected and received counts. A run that ended a backfill slice early is not `partial`:
  it stays `waiting` and resumes in the next cycle.
- **`missing`:** no file arrived. For DWD this is recorded at the deadline (23 or 24 hours after
  initialisation, from how long DWD keeps a run). For the Met Office it is recorded 29 days after
  initialisation, one day before the files would leave the bucket, and the recorder looks again
  meanwhile after a delay that doubles from 30 minutes to 12 hours.

**A fault is reported once, and names the product and the run.** The recorder sends each fault to
Sentry, or writes it to the log when no Sentry address is set. The fault types are `partial`,
`missing`, `grid_changed`, `commit_failed`, `run_error`, `disk_low`, and `grid_unchecked`. A fault
ledger on disk remembers which faults are active, so that a fault lasting many cycles raises one
event, and so that a fault that clears can fire again. Every cycle also sends a Sentry cron
check-in, which makes a dead recorder visible. Each run sits in its own `try` block, so one failing
run does not stop the others.

**A change in the grid stops commits for that product.** If a run's grid, member count, or step
list differs from the archive's, the recorder reports one `grid_changed` fault and writes a
`HALTED` file in the product's cache directory. It keeps fetching, so that no run is lost while a
person decides what to do. It commits nothing until the file is deleted.

**The recorder looks back as far as the provider keeps files.** For DWD the lookback is 27 hours.
For the Met Office it is 30 days. A cycle that starts late therefore still fetches what the
provider holds, and it makes one fetch pass before it evaluates a deadline.

## Backfill and the MOGREPS worker pool

**Each cycle handles live runs first and then spends a time budget on older runs, oldest first.**
For MOGREPS-UK a run is live for 6 hours, and for MOGREPS-Global for 24 hours. Older runs still in
the bucket are the backfill, and the provider deletes the oldest first. The recorder therefore
starts with the oldest run, and puts the runs at 00, 06, 12, and 18 UTC before the other hours. The
MOGREPS services allow 10 minutes of backfill a cycle. A run that the time budget cuts short
resumes in the next cycle.

**MOGREPS files are read by byte range in a pool of worker processes.** A file holds all members of
one variable at one lead time, in HDF5 chunks. The recorder downloads only the chunks that overlap
the crop box. The `h5py` library holds a global lock while it reads a chunk index, so threads do not
speed the reads up, and the recorder uses 4 worker processes instead. A worker only writes cropped
files to the local cache. The main process alone opens the repository, commits, and reports faults.
If a worker crashes or the pool times out, the run is not committed, because its files are in an
unknown state.

**Two MOGREPS recorders are designed to share one urgency file, and MOGREPS-UK does not yet take
part.** MOGREPS-UK and MOGREPS-Global have the same 30-day retention, and each runs as its own
process. The design is that each recorder writes to one small shared JSON file how many days remain
before its oldest queued run leaves the bucket, and skips a backfill slice when the other product
is closer to losing a run by more than half a day. An entry older than 1 hour, a missing file, or a
corrupt file counts as no signal, and none of them raises or stops a live run. MOGREPS-Global
already writes to the file. The MOGREPS-UK recorder runs an earlier version of the code, which has
no urgency coordination.

## Reading the archive

**Open a repository with `icechunk` and `xarray`.** The archive's own README shows the same
example.

```python
import icechunk
import xarray as xr

repo = icechunk.Repository.open(
    icechunk.local_filesystem_storage("/mnt/data/nwp-archive/store/icon-d2-eps")
)
session = repo.readonly_session("main")
dataset = xr.open_zarr(session.store, consolidated=False, decode_timedelta=True)
```

The `step` coordinate is in minutes. Select `status == 1` first to keep only complete runs.

## See also

- [The roadmap's data-source catalogue](../roadmap/data-sources.md#weather-data), which lists the
  products this archive records.
- [The weather-products survey](../background/weather-products-survey.md), which compares the
  archives that keep whole past runs.
- [AWS running costs](aws-costs.md#the-ensemble-archives-storage-is-not-yet-costed), for the
  archive's storage size.
