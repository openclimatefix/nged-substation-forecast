# Plan: fetch WeatherNext 3 ensemble means into an Icechunk store (#934)

**The problem.** The matched-lead comparison needs a WeatherNext 3 arm, and other OCF team members want the same data. WeatherNext 3's statistics bucket holds the ensemble mean per variable on 0.1 degree global chunks of about 20 MB (one lead each). Cropping does not shrink what is read, so one run of 7 variables at 360 leads reads about 50 GB. Over the internet the 2026 archive at 4 runs a day would read about 53 TB and cost thousands of pounds in egress.

**The plan.** A fetch script runs on a Compute Engine VM in us-east1, where reads inside the region cost nothing. It crops each run to a wide United Kingdom box (49.0 to 61.5 degrees north, 10.0 degrees west to 3.5 degrees east, chosen to reach offshore wind farms) and writes the crop into an Icechunk store in a Cloud Storage bucket in the same region. The script writes to a `staging` branch, one commit per run. A validation script checks the staging branch, and only then does `main` move to the staging tip. Colleagues read `main`.

## Verdict, size and departures

Worth doing. Departures from the issue text and the first version of this plan: the output is an Icechunk store rather than Parquet files, and the crop is the wide box rather than the private trial-area box, so the store needs no private file. The script reads the `weathernext3_statistics_spatial` bucket, not the 64-member bucket, because the study consumes ensemble means.

Size: medium. One line per trigger:

- What gets stored: an Icechunk store in a bucket owned by the maintainer's Google Cloud project. No Patito contract and no Delta table.
- Production serving path: not touched. The scripts live in `studies/weather_downloads/` and nothing imports them.
- Degradation rule: none touched. This is R&D code and fails fast.
- More than one defensible design: settled with the maintainer (Icechunk, 7 arrays, one shard per run). Plain Zarr and long-format Parquet were the alternatives.
- Callers: none. No reader of WeatherNext 3 data exists yet.

Reviews: one Opus review of this plan and the code, then a second only if the first finds show-stoppers.

## Store layout

One Icechunk repository at `gs://<bucket>/weathernext3_statistics_uk/` in us-east1.

- **Arrays.** Seven float32 arrays, one per variable: `temperature_2m_mean`, `u_component_of_wind_10m_mean`, `v_component_of_wind_10m_mean`, `u_component_of_wind_100m_mean`, `v_component_of_wind_100m_mean`, `surface_solar_radiation_downwards_1hr_mean`, `total_sky_direct_solar_radiation_at_surface_1hr_mean`. Dimensions are `(init_time, lead_time, latitude, longitude)`, in the store's native units.
- **Shards and chunks.** One shard per run: `(1, 360, 128, 136)`, which is one run, all 360 leads, and the whole box. Inner chunks are `(1, 360, 8, 8)`, so a point read fetches about 100 KB. Codec is `zarr.codecs.BloscCodec(cname="zstd", shuffle="shuffle")` (zarr 3.4 has no stand-alone shuffle codec, and this one needs no `numcodecs`), on values rounded to 13 significand bits with the existing `round_to_significand_bits` helper.
- **Coordinates.** `init_time` is preallocated with every 00, 06, 12 and 18 UTC init time from the first to the last date given when the repository is created. That range is stored in the group attributes and never changes, so runs can be written in any order and a resume needs no resize. A later invocation clips its request to the stored range and prints how many runs fall outside it. There is no default end date: `--end-date` is required at creation. `lead_time` is 1 to 360 hours. `latitude` ascends in 0.1 degree steps and `longitude` is signed degrees ascending from -10.0 (converted from the store's 0 to 360 axis). Both are stored rounded to 0.1, after asserting each is within 1e-6 of the source value, so `.sel(latitude=52.0)` is exact.
- **Presence.** A boolean array `run_written(init_time)` is set in the same commit as the run's data. An unwritten slot holds NaN in the data arrays. A `source_init_time(init_time)` array, written in the same commit, holds the init time read from the source store, so the validator can check it equals the coordinate. Runs skipped for a missing `success` marker, and expected runs the bucket does not list, are recorded in the group attributes as the union of the old and new lists across invocations.
- **Lineage.** Group attributes hold the source address pattern, the retrieval time, the box, the rounding bits, the units, and the radiation convention (J m-2 accumulated over the hour ending at the valid time, so dividing by 3600 gives mean W m-2 for that hour).

## What changes, file by file

- `studies/weather_downloads/fetch_weathernext3.py` (rewrite of the write path): keeps the source reading, retry, non-retryable errors, exit and leak rules, cost guard and `--dry-run` of the current script. Drops the private box, the Parquet run cache, the combine step and the grid-cell fingerprint. Adds `--bucket` and `--store-prefix`. Creates the repository on first use (arrays and coordinates), works on the `staging` branch, and commits once per run. Each run gets a fresh `writable_session("staging")`, which is dropped on any exception (chunks reach storage as soon as they are set, so a reused session would carry a failed run into the next commit). The script writes only a fully loaded in-memory run, and writes `run_written` last. Before writing it asserts that the source store's `init_time` equals the slot's. `commit` is not wrapped in the retry helper; after a commit error the script re-reads `run_written[slot]` on the tip before recording a failure. `--init-hours` is limited to the 00, 06, 12 and 18 UTC hours. It sets `ICECHUNK_LOG=error` before importing `icechunk`, so the Rust core's warnings cannot reach stderr, and prints exception type names only. The script never moves `main`.
- `studies/weather_downloads/validate_weathernext3.py` (rewrite): opens the repository read-only on `staging` (or on `main` with a flag). Checks: array names, dtypes, shapes, chunk and shard shapes and codecs; latitude and longitude axes regular and ascending; `init_time` matches the expected runs after allowing the recorded skips; for every written run, no NaN or infinity, physical ranges, radiation night and midday checks, direct at most total, no constant slice; for every unwritten slot, all NaN. Also: `source_init_time` equals the coordinate, no two written runs are identical (hash of lead 1 of `temperature_2m_mean`), and the listed skips cover every unwritten slot. `--compare-source` selects the source cells by value from the stored `latitude` and `longitude` (with `lon % 360`), independently of the fetch's crop code, and compares the full 126 by 136 slice for the first and last lead of every variable across the first run, the last run and about five random runs. `--publish` records the snapshot id when it opens `staging`, validates that snapshot, checks the current `main` tip is an ancestor of it, and calls `repo.reset_branch("main", validated_id, from_snapshot_id=<main tip read at the start>)`, so runs committed during validation are never published. The validator runs on the VM (or in us-east1), because it reads every shard.
- `pyproject.toml`, `uv.lock`: add `icechunk` and `zarr` to the studies dependencies.
- Docs: the WeatherNext 3 row of `docs/roadmap/data-sources.md` and the survey (findings with scope, plus the store location and how to open it); `studies/weather_downloads/README.md` lists the scripts.

## Design-philosophy check

R&D code: fails fast, no degradation path, nothing on the serving path, no asset checks. No project id, account name, bucket credentials, or private coordinate reaches stdout, a log line, a filename or a doc. The wide box is not private.

## Tests

None beyond running the scripts, matching the sibling scripts. Correctness is established by `validate_weathernext3.py` on real data, including the `--compare-source` check, plus a synthetic local-filesystem end-to-end run (a small fake source store, a local Icechunk repository) with mutation tests of each validator check, as in the first round of reviews.

## Verification commands

`uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md && uv run pre-commit run --all-files`, plus pydoclint from an isolated environment. First real run on the VM: `--dry-run`, then one run on one day, then time it, then validate on `staging`.

## Risks and open questions

- **Credentials.** Icechunk uses its own Cloud Storage client (`icechunk.gcs_storage(..., from_env=True)`), not `gcsfs`. Whether it picks up the VM's service account is unconfirmed, so the first VM step creates, commits and deletes a tiny test repository under a test prefix.
- **Sharded writes over Cloud Storage.** Sharding works on a local Icechunk store (tested). Write speed, ranged-read efficiency and the conditional-write warning on a Cloud Storage bucket are untested and get measured on the first VM run.
- **Access.** The maintainer creates the bucket in us-east1 and grants the VM service account write access. Colleagues need read access, and a read from outside Google Cloud is billed to the bucket owner as egress (about $10 for a full read of the roughly 75 to 100 GB store), so Requester Pays on the output bucket is worth considering. The bucket name goes into `docs/` only if it does not contain the project id. Whether WeatherNext 3 data may be shared within OCF under the access terms is the maintainer's call; the data is CC BY 4.0 once it is more than an hour old.
- **Chunk shape.** The 8 by 8 inner chunk is a guess about how colleagues read the data. It is cheap to change before the full archive is written and expensive afterwards.
- **VM time is unmeasured.** The single test run took about 21 minutes over the internet. The time per run in us-east1, and so the VM bill for 1,068 runs, is unknown until one run is timed there.
- **Appending later.** A live service would append runs after the last preallocated init time. That needs an array resize and is not planned here.
