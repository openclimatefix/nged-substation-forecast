# Plan: fetch WeatherNext 3 ensemble means for the matched-lead study (#934)

**The problem.** The matched-lead comparison needs a WeatherNext 3 arm. WeatherNext 3's statistics bucket holds the ensemble mean per variable, on 0.1 degree global chunks of about 20 MB (one lead each). Cropping to the trial-area box does not shrink what is read, so one run of 7 variables at 360 leads reads about 50 GB. Over the internet the 2026 archive (267 runs at one per day) would cost about £900 in egress.

**The plan.** Add a one-off fetch script that runs on a Compute Engine VM in us-east1, where reads inside the region cost nothing. It crops each run to the trial-area box and writes one small Parquet file per run (about 15 MB). Only those files come home. A validation script checks them on the VM before they are copied.

## Verdict, size and departures

Worth doing as described in the issue. Departures from the issue text and the maintainer's brief: the script checkpoints one run per file rather than one month, because one run is about 50 GB of reads and a month would be 1.5 TB; and it reads the `weathernext3_statistics_spatial` bucket, not the 64-member bucket, because the study consumes ensemble means.

Size: medium. One line per trigger:

- What gets stored: files under the gitignored `data/studies/weather/WeatherNext3/`, no Patito contract, no Delta table.
- Production serving path: not touched. The script lives in `studies/weather_downloads/` and nothing imports it.
- Degradation rule: none touched. It is R&D code and fails fast.
- More than one defensible design: the route was decided with the maintainer after testing Cloud Storage, BigQuery and Earth Engine. The script layout follows `fetch_dynamical_zarr.py`.
- Callers: none. New files only, plus two docs rows.

Reviews: zero plan reviews (the route is settled), and two Opus reviews before the script first touches the real store: one of the script alone, one of the diff.

## What changes, file by file

- `studies/weather_downloads/fetch_weathernext3.py` (new): opens each run's `predictions.zarr` lazily with `gcsfs` (Requester Pays, credentials and billing project from the environment, or the VM's default credentials) and `xarray.open_zarr(chunks=None)`. Crops with the trial-area box (longitude converted from the store's 0 to 360 axis to signed degrees). Reads the 7 mean variables in a thread pool, one (variable, lead) chunk per task, with retry and backoff. Writes `_run_cache/<YYYYMMDD_HH>.parquet` through a `.partial` rename. Skips cached runs, skips runs whose `success` marker is absent. Combines the cache into `WeatherNext3.parquet`. Refuses a run outside us-east1 when the estimated transfer exceeds a small cap (`--max-external-gb`, default 5), so the £900 mistake cannot happen by accident. `--dry-run` prints the run count and estimated GB only.
- `studies/weather_downloads/validate_weathernext3.py` (new): the data-validation checklist on the per-run files (run spacing, lead axis 1 to 360, key uniqueness, nulls, physical ranges, night-time and midday radiation, grid-hash agreement, exact row count).
- `pyproject.toml` (studies dependencies): add `gcsfs` and `zarr` only if `uv sync` does not already provide them.
- Docs: WeatherNext 3 row of `docs/roadmap/data-sources.md` and the survey (findings with scope: archive start, hourly cadence to 360 h, statistics-only BigQuery and Earth Engine, 0.1 degree, chunk sizes, us-east1); `studies/weather_downloads/README.md` gets the script listed.

Output schema: the long format `fetch_dynamical_zarr.py` writes: `init_time`, `lead_time` (Duration), `lat_index`, `lon_index`, then the value columns in the store's native units (`*_mean` names), rounded to 13 significand bits. `_grid_cells.parquet` stays private. There is no `ensemble_member` column, because the values are already means. Radiation is J/m2 accumulated over the hour ending at the valid time, so dividing by 3600 gives mean W/m2 for that hour; the README says so.

## Design-philosophy check

R&D code: fails fast, no degradation path, nothing on the serving path. No asset checks. No generator name, ID or coordinate reaches stdout, a log line, a filename or a doc: cell counts and the box stay in private metadata.

## Tests

None beyond running the script. It is a throwaway under `studies/`, like the other fetch scripts, and its correctness is established by `validate_weathernext3.py` on real data plus one hand-checked lead against the Zarr store. Recommendation: no unit tests, matching the sibling scripts.

## Verification commands

`uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md && uv run pre-commit run --all-files`. First real run: `--dry-run`, then one run with `--start-date`/`--end-date` on one day (about 50 GB of reads, so only inside us-east1).

## Risks and open questions

- The lead axis of the 00 UTC run and the `success` marker's meaning are assumed from two inspected runs; the dry run and validation will confirm them for every run.
- The VM needs the repository, `uv sync`, and a copy of the private `_trial_area_box.json`. That file is the box's bounds. Copying it to a VM in the maintainer's own project keeps it private; recommendation: do that, and delete the VM afterwards.
- 4 runs per day is affordable on the VM (about 200 GB per day of reads, no egress) if the study wants it. The default is 00 UTC only.
