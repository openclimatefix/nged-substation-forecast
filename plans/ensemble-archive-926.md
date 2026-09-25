# Plan: always-on archive of short-retention ensemble weather products (#926)

**The problem.** DWD keeps only the last 4 runs of ICON-EU-EPS and the last 8 runs of ICON-D2-EPS (about 24 hours), and the Met Office keeps MOGREPS-UK for 30 days. Nobody archives these products, so a later study cannot compare ensemble means against deterministic products (UKV, ICON-EU, ICON-D2, ECMWF ENS) for Great Britain solar and wind forecasting. Every day the recorder is down is lost permanently for the DWD products.

**The plan.** A new workspace package, `packages/ensemble_archive`, holds a small recorder that runs on an AWS t4g.small in London under systemd timers. Every 15 minutes it works out which product runs should exist, downloads the wanted fields of any complete run, crops them to Great Britain on the native grid, writes one Zarr v3 store per product run to Source Cooperative (direct write from an IAM role, their "Option 3"), and writes a per-run manifest last, as the commit. A daily pass reconciles against the providers' listings, and a GitHub Actions cron reads the public manifests and opens an issue for any missing or partial run. Nothing is stored or published until the accounts in "Blocked on the maintainer" exist.

## Verdict, size and departures

**Verdict:** worth doing as described, with the departures below.

**Size: complex.** The five triggers:

- *What gets stored:* fires. A new public archive with a new on-disk format and manifest schema.
- *Production serving path:* does not fire. The recorder is separate from the forecast service and shares no code with it.
- *A degradation rule:* does not fire for the forecast service. The recorder has its own degradation rule (publish a partial run after the deadline), described below.
- *More than one defensible design:* fires. Host, storage format, and where a per-run commit lives all have serious alternatives.
- *Code whose callers could not be named without searching:* does not fire. The package is new and nothing calls it.

The size buys the plan, both plan reviews, and both diff reviews.

**Departures from the brief and from #801:**

- #801 proposed parquet per run, cropped to a trial-area box. The brief chose Zarr v3 on the native grid cropped to Great Britain, because a later study may use any region and members are needed for calibration. The Zarr choice stands.
- #801 proposed polling every 3 hours. DWD publishes each run over about 1.5 hours, so a 15-minute timer with a file-count check is used instead.
- The brief lists the minimal wind as 10 m. The maintainer asked for about 100 m wind as well; the verification below found it in all three products.

## What was verified before planning (2026-09-25, no account)

- **DWD path:** `https://opendata.dwd.de/weather/nwp/v1/m/{model}/p/{PARAM}[/lvt1/150/lv1/{level}]/r/{run}/e/{member}/s/PT{HHH}H{MM}M.grib2`, where `{run}` is `YYYY-MM-DDTHH%3A00`. Surface fields have no level directory. ICON-D2-EPS has 20 member directories (`01`-`20`) and 50 step files per member for one field, including `PT000H00M`.
- **DWD products in `v1/m`:** `icon-eu-eps`, `icon-d2-eps`, `icon-d2`, `icon-art-eu`, `icon-art-eu-eps`, `icon-art`, `icon-art-eps`, `icon-d2-ruc`, `icon-eu`, `icon`, `aicon`. There is no ICON-ART-D2.
- **ICON-D2-EPS levels:** `U`/`V` model levels 56-65 and `HHL` levels 56-66. Half-level heights above ground (median over the domain): level 62 at 151 m, 63 at 99 m, 64 at 55 m, 65 at 20 m. Layer 63 (between half levels 63 and 64) is centred at about 77 m and layer 62 at about 125 m, so 100 m lies between them.
- **ICON-EU-EPS levels:** `U`/`V` model levels 72-74 and `HHL` levels 72-75. Half-level heights above ground: 72 at 126 m, 73 at 64 m, 74 at 20 m. Layer 72 is centred at about 95 m, so it stands in for 100 m without interpolation.
- **MOGREPS-UK:** `wind_speed_on_height_levels` has 33 heights including exactly 100 m, dimension order (member, height, y, x) on a 970 x 1042 Lambert azimuthal equal-area grid, HDF5 chunks (1, 1, 128, 128) with gzip. A single height for a single member is therefore readable with byte-range requests. A whole file is 112 MB.
- **Not yet verified** (first implementation task, each with an assertion in the validator): the DWD static grid file paths (`CLAT`, `CLON`, `HSURF`, `FR_LAND`); the exact MOGREPS-UK key layout for run time versus valid time (the prefix `T0000Z` held files named `20260924T1200Z-PT0012H00M-...`); whether the DWD and MOGREPS shortwave fields are hourly means or instantaneous values (read `cell_methods` and the GRIB `stepType`); the MOGREPS-UK parameter list per step; `.idx`-style byte-range options (DWD v1 has one field per file, so none are needed); post-crop compression ratio.

## What changes, file by file

New package `packages/ensemble_archive/` (workspace member, `uv_build`, Python 3.14, dependencies `eccodes`, `zarr>=3`, `numpy`, `obstore` or `s3fs` for S3, `h5py` plus `fsspec` for MOGREPS byte-range reads, `httpx`, `patito`/`polars` for the manifest):

- `products.py` — a frozen table of `Product` records: name, provider, licence, run cycle hours, members, leads, fields, and the expected file count per run (fields x levels x members x steps). This is the one place the expected-run table and the fetcher both read.
- `dwd.py` — list a run's directory, download files with keep-alive and a bounded thread pool, retry with exponential backoff and jitter, throttle to a configurable Mbit/s, decode GRIB2 with eccodes, return one array per (variable, member, step).
- `mogreps.py` — list the day prefix on S3 anonymously, read one member-height slab at a time by byte range, convert wind speed and direction to the same components stored for DWD (`u`, `v`, with the speed and direction kept as delivered, because #525 tracks the component convention).
- `crop.py` — the Great Britain box (49.8-61.0 N, 8.3 W-2.0 E; the whole of Northern Ireland is outside it and can be added later). DWD: keep native cells whose `clat`/`clon` fall in the box and store those two coordinate arrays. MOGREPS-UK: an index box on its own projection, stored with the projection attributes.
- `zarr_store.py` — write one store per product run at `<product>/<init>.zarr`, arrays shaped (member, step, cell) or (member, step, y, x) with a 13-bit significand rounding and zstd, staged under `_staging/`, verified, then copied to the final key. Re-running a run rewrites identical objects.
- `manifest.py` — one JSON file per run, `<product>/<init>/manifest.json`, holding status (`complete`, `partial`, `missing`), expected and received file counts, source bytes, SHA-256 of each stored array, code version, ICON model version from the GRIB header where present, and archive time. It is written last and acts as the commit. There is no shared mutable index, so two recorders cannot corrupt each other.
- `recorder.py` — the 15-minute loop: compute expected runs from `products.py`, skip runs whose manifest already says `complete`, fetch runs whose provider file count is complete, retry incomplete runs until the deadline (20 hours after init for DWD, 28 days for MOGREPS-UK), then publish what exists as `partial`.
- `reconcile.py` — the daily pass. DWD: compare the manifests with DWD's `content.log.bz2` (77 MB) and report runs the provider held that the archive lacks. MOGREPS-UK: list days D-29 to D-1 and fetch any run missing a manifest. Both also compare the expected parameter list against the listing and report drift.
- `check_manifests.py` — reads the public manifests over HTTPS, compares them with the expected-run table, and prints missing or partial runs (nonzero exit when any exist). No credentials.
- `validate.py` — the checks from the `data-validation` skill applied to a stored run: no NaN or duplicate keys, expected member and step counts, physically plausible ranges per variable, direct plus diffuse against total shortwave, day-night profile, and grid orientation via the coordinate arrays.
- `cli.py` — entry points `archive-record`, `archive-reconcile`, `archive-check`, `archive-validate`. Credentials come from environment variables or the instance profile only; nothing prints them.
- `README.md` — what is stored, layout, licence and attribution per product, lineage, and how to read a store from Python.
- `deploy/` — systemd service and timer unit files, an instance bootstrap script, and the IAM policy JSON for the Source Cooperative role.
- `.github/workflows/archive-check.yml` — runs `archive-check` hourly and opens (or comments on) one issue per missing or partial run. It uses `GITHUB_TOKEN` only.

Root `pyproject.toml`: add the workspace source. Docs: a new page `docs/architecture/ensemble-archive.md` (what is stored, where, licence, how to read it, the reliability design), linked from `docs/roadmap/data-sources.md` and `docs/background/weather-products-survey.md`, and added to `mkdocs.yml`.

## Field set

- **All products:** `ASWDIR_S`, `ASWDIFD_S` (direct and diffuse shortwave; total is their sum for DWD, so no separate total is fetched), `T_2M`, `U_10M`, `V_10M`, `CLCT`, plus static `CLAT`, `CLON`, `HSURF`, `FR_LAND` once per product.
- **ICON-EU-EPS:** also `U`, `V` at model level 72 (about 95 m).
- **ICON-D2-EPS:** also `U`, `V` at model levels 63 and 62 (about 77 m and 125 m), plus `HHL` levels 62-64 so a user can interpolate to 100 m.
- **ICON-D2 deterministic:** the same six fields, and `U`, `V` at levels 63 and 62.
- **ICON-ART-EU deterministic:** `ASWDIR_S`, `ASWDIFD_S`, `T_2M`, `CLCT`, `ASOB_S_CS`, `TAOD_DUST`.
- **MOGREPS-UK:** `radiation_flux_in_shortwave_{total,direct,diffuse}_downward_at_surface`, `temperature_at_screen_level`, `wind_speed_at_10m`, `wind_direction_at_10m`, `cloud_amount_of_total_cloud`, and the 100 m level of `wind_speed_on_height_levels` and `wind_direction_on_height_levels`.

The maintainer chose full horizons (ICON-EU-EPS 0-120 h, ICON-D2-EPS 0-48 h, MOGREPS-UK 0-126 h) and every member. Stored volume is estimated at about 7.5 TB a year, most of it MOGREPS-UK; ICON-D2-EPS is the product to reduce (mean, spread and a few quantiles) if storage cost later matters.

## Design-philosophy check

The recorder runs in production, unattended, so it follows `docs/design-philosophy/inherent-stability.md`: an absent or late provider file never raises; the run is retried, and at the deadline the archive publishes what exists marked `partial` with expected and received counts on the manifest. Malformed data (a wrong shape, a failed decode) is rejected at the validator and the run is marked `partial`, never silently written. The alert names the product and the init time, so the operator does not read logs. The alert check warns and cannot block anything. The capability lives outside the forecast serving path (principle: keep serving close to "load a model, call predict"). No Patito contract is changed; the manifest is a new model private to the package.

## Tests

- `test_products.py`: the expected file count for one ICON-D2-EPS run equals fields x levels x members x steps. Fails on `main` because the package does not exist.
- `test_dwd.py`: against a stub HTTP server serving a listing with one missing step file, the run is reported incomplete and no manifest is written; after the file appears, the run completes. Also a 500 then 200 response is retried once.
- `test_crop.py`: a synthetic triangular grid with known `clat`/`clon` keeps exactly the cells in the box, and the stored coordinates match. A cell one hundredth of a degree outside is dropped.
- `test_mogreps.py`: a small synthetic NetCDF chunked (1, 1, 128, 128) on a mocked S3 (moto) returns exactly the 100 m slab, and the recorded request count is at most the number of chunks touched.
- `test_zarr_store.py`: writing the same run twice yields identical checksums; an interrupted write (exception before the manifest) leaves no manifest, so the run is still expected.
- `test_recorder.py`: at the deadline, a run with 99 of 100 files is published `partial` with counts 99/100; a fully absent run past the deadline is recorded `missing`. Nothing raises in either case.
- `test_check_manifests.py`: a missing run and a partial run are reported and produce a nonzero exit; a complete day exits zero.
- `test_validate.py`: a run with a NaN column, or with direct plus diffuse differing from total by more than a tolerance, fails validation.
- Network-gated tests (`--run-network -m network`): one real ICON-D2-EPS field, one real ICON-EU-EPS field, one MOGREPS-UK 100 m slab, each decoded and range-checked.

## Docs to update

`docs/architecture/ensemble-archive.md` (new), `docs/roadmap/data-sources.md` and `docs/background/weather-products-survey.md` (links), `mkdocs.yml`, and the `CLAUDE.md` packages table. Issue #801 gets a comment linking #926 and stating what remains open there.

## Verification commands

`uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest && uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md && uv run mkdocs build --strict`, then `uv run pre-commit run --all-files` and the pydoclint and docs-link checks CI runs. Before any run against real sources, an Opus review of every script, then one dry run against one product run with the store directed to a local path.

## Blocked on the maintainer

- **Source Cooperative:** an organisation account in the project's name (an email to hello@source.coop) and beta approval. Until it exists the recorder writes to a local staging directory on the instance.
- **AWS:** an IAM role with `s3:PutObject` and `s3:PutObjectAcl` on the Source Cooperative prefix, and an instance profile for the t4g.small; the role ARN goes to Source Cooperative.
- **Source Cooperative's post-beta pricing** is not asked for yet, by the maintainer's decision.

## Risks and open questions

- **Recorder before the account exists?** The DWD runs vanish in 24 hours, so every day without a recorder is lost. Recommendation: start recording to instance disk (about 20 GB a day stored, so 100 GB of EBS holds five days) as soon as the AWS instance exists, and copy to Source Cooperative once it accepts writes.
- **Data-proxy uploads** are described as both "currently disabled" and supported in Source Cooperative's docs. Option 3 (direct S3 with our own role) avoids the question. Confirm in the ask to Source Cooperative.
- **Second recorder for DWD.** The AWS instance is a single point of failure for the two DWD products. The maintainer's home NUC can run the same code as a second recorder writing the same keys (the manifest-last design makes this safe). Recommendation: add after the first week, not in this issue.
- **Post-beta storage cost.** At S3 London list price, 7.5 TB is about 135 GBP a month. Storage is excluded from the 25 GBP budget by the maintainer's decision.
- **Licence.** MOGREPS-UK is CC BY-SA 4.0 and is published as a separate product from the CC BY 4.0 DWD products; any product combining them inherits share-alike.
- **A silent operational-model change.** DWD switched ICON and ICON-EU to a prognostic aerosol scheme on 2026-09-02, so ICON-EU output has a step change on that date. Each manifest records the model version where the GRIB header carries it.
