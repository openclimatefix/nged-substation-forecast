# Plan: always-on archive of short-retention ensemble weather products (#926)

**The problem.** DWD keeps only the last 4 runs of ICON-EU-EPS and the last 8 runs of ICON-D2-EPS (about 24 hours), and the Met Office keeps MOGREPS-UK for about 33 days. Nobody archives these products, so a later study cannot compare ensemble means against deterministic products (UKV, ICON-EU, ICON-D2, ECMWF ENS) for Great Britain solar and wind forecasting, onshore and offshore. Every day the recorder is down is lost permanently for the DWD products.

**The plan.** A new public repository, `openclimatefix/nwp-archivist` (pull requests limited to collaborators), holds a small recorder that runs on an AWS t4g.small in London under a systemd timer. The recorder computes which product runs should exist, fetches each expected file (a 404 means "not yet"), crops it to a "fat Great Britain" box on the native grid, builds one Zarr v3 store per product run on the instance's disk, and uploads the store to Source Cooperative with the root `zarr.json` last, so that object is the commit. Missed or partial runs are reported to Sentry. The work ships in two pull requests: the DWD products first, because DWD deletes each run after 24 hours, then MOGREPS-UK, which has about four weeks of slack. The recorder starts only after the accounts in "Blocked on the maintainer" exist.

## Verdict, size and departures

**Verdict:** worth doing as described, with the departures below.

**Size: complex.** The five triggers:

- *What gets stored:* fires. A new public archive with a new on-disk layout.
- *Production serving path:* does not fire. The recorder is separate from the forecast service and shares no code with it.
- *A degradation rule:* does not fire for the forecast service. The recorder has its own rule (commit what exists as `partial` after the deadline).
- *More than one defensible design:* fires. Host, storage format, and where the commit lives all have serious alternatives.
- *Code whose callers could not be named without searching:* does not fire. The package is new and nothing calls it.

The size buys the plan, both plan reviews, and both diff reviews.

**Departures from the brief and from #801:**

- #801 proposed parquet per run cropped to a trial-area box. This plan uses Zarr v3 on the native grid cropped to a fat Great Britain box, because a later study may use any region and members are needed for calibration.
- #801 proposed polling every 3 hours. This plan polls every 15 minutes.
- The maintainer asked for 100 m wind and for a box reaching offshore wind sites, and chose full horizons and all members.
- The brief listed a GitHub Actions manifest check and a daily reconciliation pass. The first plan review showed both duplicate the recorder's own status, so both are dropped (see "Review 1").

## What was verified before planning (2026-09-25, no account)

- **DWD path, ensembles:** `https://opendata.dwd.de/weather/nwp/v1/m/{model}/p/{PARAM}[/lvt1/150/lv1/{level}]/r/{run}/e/{member}/s/PT{HHH}H{MM}M.grib2`, where `{run}` is `YYYY-MM-DDTHH%3A00`. Surface fields have no level directory. ICON-D2-EPS has members `01`-`20` and 49 steps per member (`PT000H00M` to `PT048H00M`). ICON-EU-EPS has 40 members and 91 steps (hourly to 48 h, then 3-hourly to 120 h).
- **DWD path, deterministic:** `icon-d2` and `icon-art-eu` have no `e/{member}` directory: `.../r/{run}/s/PT{HHH}H{MM}M.grib2`.
- **Static fields:** `CLAT`, `CLON`, `HSURF`, `FR_LAND` and `HHL` are republished with every run and member as `PT000H00M` files, so grid coordinates are read from member `01` of each run.
- **DWD products in `v1/m`:** `icon-eu-eps`, `icon-d2-eps`, `icon-d2`, `icon-art-eu`, `icon-art-eu-eps`, `icon-art`, `icon-art-eps`, `icon-d2-ruc`, `icon-eu`, `icon`, `aicon`. There is no ICON-ART-D2.
- **Publication timing:** ICON-D2-EPS run 09Z appeared 09:44-10:40 UTC; ICON-EU-EPS run 12Z appeared 14:38-15:06 UTC.
- **ICON-D2-EPS levels:** `U`/`V` model levels 56-65, `HHL` levels 56-66. Half-level heights above ground (domain median): level 62 at 151 m, 63 at 99 m, 64 at 55 m, 65 at 20 m. Layer 63 is centred at about 77 m and layer 62 at about 125 m, so 100 m lies between them.
- **DWD shortwave** (`ASWDIR_S`, `ASWDIFD_S`) is an average since the start of the run (`stepType=avg`, `stepRange` `0-24` or `0m-15m`), not an hourly mean and not instantaneous. It is stored as delivered, and the docs page says how to de-average and that de-averaging multiplies the rounding error by about the step number.
- **ICON-EU-EPS levels:** `U`/`V` model levels 72-74, `HHL` levels 72-75. Half-level heights above ground: 72 at 126 m, 73 at 64 m, 74 at 20 m. Layer 72 is centred at about 95 m and stands in for 100 m.
- **MOGREPS-UK:** prefixes are keyed by run time (`uk-ensemble/YYYY/MM/DD/THHMMZ/`), 24 runs a day each to `PT0126H00M`, 14,330 files and about 258 GB per hourly run, and the bucket holds about 33 days (from 2026-08-23). `wind_speed_on_height_levels` has 33 heights including 100 m, chunked (1, 1, 128, 128) per member and height, so one height reads by byte range. Surface fields are `(3, 970, 1042)` float32 with chunks `(1, 128, 128)`, and several also come at 15-minute steps early in the run. The shortwave field carries no `cell_methods`, so it looks instantaneous (one file, not proof).
- **Region:** Source Cooperative's infrastructure page lists London (`eu-west-2`) among its regions, with bucket `eu-west-2.opendata.source.coop`.
- **`content.log.bz2`** is at `https://opendata.dwd.de/weather/nwp/content.log.bz2` and is not used by the recorder.
- **Still to verify in implementation:** whether the numpy 13-bit rounding matches `delta_store.precision` (`NWP_SIGNIFICAND_BITS`), and how `ty` types the `eccodes` bindings.

## Where the code lives

The maintainer chose a separate repository, `openclimatefix/nwp-archivist`, created public with pull requests restricted to collaborators (the repository setting `pull_request_creation_policy=collaborators_only`). The reasons: the recorder shares no code with the forecasting packages, it should deploy from a pinned tag rather than track `main`, and a public archive with its own licences is easier to cite and reuse as its own repository. Issue #926 and the consumer-facing docs page stay in this repository. This plan file and draft PR #927 stay here until the plan is approved; the implementation PRs then open in `nwp-archivist`.

## Sequencing: two pull requests under #926

1. **PR 1 (this branch): the DWD products** ICON-EU-EPS, ICON-D2-EPS, ICON-D2 and ICON-ART-EU deterministic, the recorder, the package README, the docs page, and the deployment files. This is the urgent PR.
2. **PR 2: MOGREPS-UK**, adding `mogreps.py` (byte-range HDF5 reads, the 15-minute versus hourly steps, the 100 m slab, a 28-day retry window with per-run back-off) and its tests. It ships as a separate product because of the CC BY-SA licence.

## What changes, file by file (PR 1)

New Python package in the `nwp-archivist` repository (its own `pyproject.toml`, `uv_build`, Python 3.14, the house ruff, `ty` and pre-commit settings copied from this repository; dependencies `eccodes`, `zarr>=3`, `numcodecs`, `numpy`, `httpx`, `sentry-sdk`, and `obstore` or `s3fs` for the upload):

- `products.py` — a frozen table of `Product` records (name, provider, licence, cycle hours, members, path template, and per field the levels and the step list) and a function that returns the exact list of expected file URLs for a run. Step lists differ by field, not by product: ICON-D2 `ASWDIR_S` and `ASWDIFD_S` have 193 files a run (every 15 minutes to 48 h) while its other fields have 49; ICON-ART-EU has 75 steps for `ASWDIR_S`, `ASWDIFD_S`, `T_2M` and `CLCT`, 64 for `ASOB_S_CS` and 89 for `TAOD_DUST`. Each field is stored with its own `step` coordinate. The recorder, the tests and the docs read this one table.
- `dwd.py` — fetch one file with keep-alive, retry with exponential backoff and jitter, and an optional rate limit; decode one GRIB2 message with eccodes. A 404, a short body (`Content-Length` mismatch), a decode failure, or a decoded message whose `shortName`, `stepRange`, `perturbationNumber` or level does not match the URL is "not yet": the file is not marked received and is retried next cycle, and it becomes a Sentry warning only if it persists at the deadline. Cells masked by a GRIB bitmap (eccodes returns them as 9999.0; 3,894 of the 92,512 ICON-D2 cells in the box are masked) become NaN.
- `store.py` — crop to the box, write each decoded step into a local checkpoint store with one chunk per `(member, step)` (arrays `(member, step, cell)`; a one-step write into a per-member chunk rewrites the whole chunk, measured at 123 ms against 7 ms), keep a sidecar `state.json` that lists the received file keys, the status and an `uploaded` flag, and write the cropped `clat`, `clon`, `hsurf`, `fr_land` and `hhl` (levels 62-64 for ICON-D2 and ICON-D2-EPS, 72-74 for ICON-EU-EPS) into every store. At commit the store is rechunked once, member by member, to per-member chunks (or shards) so a run is tens of objects, with 13-bit significand rounding applied in numpy before the write and only Zarr v3 specification codecs (`bytes`, `zstd`, `crc32c`), because `numcodecs.BitRound` writes a codec outside the specification that other readers may refuse. The committed local store is immutable. Upload awaits every object except the root, then writes the root `zarr.json` with consolidated metadata last. The root attributes carry status (`complete` or `partial`), expected and received file counts, source bytes, code version, ICON model version where the GRIB header has one, and archive time. After the remote root is confirmed, the local store is deleted and a small tombstone is kept so the run is not fetched again.
- `recorder.py` and a small CLI (`archive-record`) — every 15 minutes: compute expected runs, start each run at a fixed delay after init, fetch every expected file not yet in the local store, commit when the count reaches the expected count or the deadline passes, then upload. DWD keeps a run for about 25-26 hours, so the deadline is set from retention (init + 23 h for ICON-D2-EPS and ICON-D2, init + 24 h for ICON-EU-EPS and ICON-ART-EU), the loop looks back to the retention horizon, and it always makes one fetch pass before it evaluates the deadline, so an instance that restarts late still fetches what DWD holds. The clock is injectable. A `missing` run (nothing arrived by the deadline) is recorded as a tombstone and one Sentry event, reported exactly once. Each run is wrapped in its own `try`/`except` that sends the error to Sentry and moves on, and a free-space check before each run warns through Sentry when the disk is nearly full, so a full disk or a decode error in one run cannot stop the others. Each cycle sends a Sentry cron check-in, and each `partial` or `missing` run sends a warning event tagged with the product and init time. Nothing in the recorder raises on an absent or late file.
- `README.md`, `deploy/` (systemd service and timer, instance bootstrap, IAM policy JSON), and `docs/architecture/ensemble-archive.md` (what is stored, where, licence, how to read it, the reliability design), linked from `docs/roadmap/data-sources.md` and `docs/background/weather-products-survey.md` and added to `mkdocs.yml`.
- In this repository, only `docs/architecture/ensemble-archive.md`, its two inbound links and `mkdocs.yml`; the code, its README and the deployment files live in `nwp-archivist`. The instance deploys a tagged release of that repository, and `nwp-archivist` runs its own tests, including the nightly network tests.

**The crop box** is the "fat Great Britain" box 49.0-61.5 N, 10.0 W-3.5 E, chosen by the maintainer to reach offshore wind farms. It covers Northern Ireland, the Irish Sea, the Celtic Sea off Cornwall, the seas west of the Hebrides, Shetland, and the North Sea out to the Dutch and Belgian coasts. ICON-D2-EPS covers only the part east of about 3.94 W, so its offshore coverage is the North Sea. DWD: keep native cells whose `clat`/`clon` fall in the box. MOGREPS-UK (PR 2): an index box on its own projection.

**Field set.**

- **All DWD products:** `ASWDIR_S`, `ASWDIFD_S` (total is their sum, so no total is fetched), `T_2M`, `U_10M`, `V_10M`, `CLCT`.
- **ICON-EU-EPS:** also `U`, `V` at model level 72 (about 95 m).
- **ICON-D2-EPS and ICON-D2:** also `U`, `V` at model levels 63 and 62 (about 77 m and 125 m), and `HHL` so a reader can interpolate to 100 m.
- **ICON-ART-EU:** `ASWDIR_S`, `ASWDIFD_S`, `T_2M`, `CLCT`, `ASOB_S_CS`, `TAOD_DUST`.
- **MOGREPS-UK (PR 2):** `radiation_flux_in_shortwave_{total,direct,diffuse}_downward_at_surface`, `temperature_at_screen_level`, `wind_speed_at_10m`, `wind_direction_at_10m`, `cloud_amount_of_total_cloud`, and the 100 m level of `wind_speed_on_height_levels` and `wind_direction_on_height_levels`, all as delivered.

Horizons are full (ICON-EU-EPS 0-120 h, ICON-D2-EPS 0-48 h, MOGREPS-UK 0-126 h) with every member kept. The stored volume was estimated at about 7.5 TB a year for the earlier 49.8-61.0 N, 8.3 W-2.0 E box; the fat box is about 1.5 times that area, so roughly 11 TB (an estimate, to be replaced by the measured size of the first dry run). ICON-D2-EPS is the product to reduce (mean, spread and a few quantiles) if storage cost later matters.

## Design-philosophy check

The recorder runs in production, unattended, so it follows `docs/design-philosophy/inherent-stability.md`: an absent or late provider file never raises; the run is retried, and at the deadline the archive commits what exists marked `partial`, with expected and received counts in the root attributes. Structural problems (a decoded shape that differs from the expected grid) are recorded on the run and the data is still kept, because for a source that deletes its copy within 24 hours, discarding a run is losing it. Sentry events name the product and the init time (principle 16), and the cron check-in makes a dead instance visible. The capability lives outside the forecast serving path. No Patito contract is added or changed.

## Tests

Each new test states the assertion that fails on `main` today (the package does not exist), and the interesting ones are these:

- `test_products.py`: the expected file list for one ICON-D2-EPS run has 20 members x 49 steps per surface field; a deterministic product's URLs contain no `/e/` segment; ICON-D2 `ASWDIR_S` has 193 steps and its `T_2M` 49; ICON-ART-EU `TAOD_DUST` has 89.
- `test_dwd.py` (stub HTTP server): a 404 leaves the file unwritten and the run incomplete; a later 200 completes it; a 500 then a 200 is retried once.
- `test_store.py`: a synthetic triangular grid keeps exactly the cells in the box, and a cell 0.01 degrees outside is dropped; writing the same run twice gives identical decoded arrays; a bitmapped message decodes to NaN, never 9999; an upload against a fake store that records PUT order (or moto, reset per test) sends the root `zarr.json` last, and an upload interrupted before the root leaves chunks present but a store that cannot be opened.
- `test_recorder.py`: at the deadline a run with 99 of 100 files is committed `partial` with counts 99/100; a run with none is `missing` and sends one Sentry event; neither raises; a re-run after `complete` fetches nothing. The tests inject the clock and a fake Sentry: a restart at init + 23 h with a half-filled local store fetches the missing files and commits `complete`; a restart at init + 30 h reports the run `missing` exactly once; a truncated 200 response and a header that disagrees with the URL both leave the file not received; an `ENOSPC` error in one run sends one Sentry event and lets the next run proceed.
- Network-gated tests (`--run-network -m network`): one real ICON-D2-EPS field, one real ICON-EU-EPS field and one real ICON-D2 field, each decoded and range-checked.

## Docs to update

`docs/architecture/ensemble-archive.md` (new, here), the two link pages, and `mkdocs.yml`. The `nwp-archivist` README carries the licences, attribution and lineage. Issue #801 gets a comment linking #926 and stating what remains open there.

## Verification commands

`uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest && uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md && uv run mkdocs build --strict`, then `uv run pre-commit run --all-files` and the pydoclint and docs-link checks CI runs. Before any run against real sources, an Opus review of every script. The dry run stores to a local path and is checked with the `data-validation` skill's checklist as a one-off script (plausible ranges, day-night profile, orientation), not as a module on the instance.

## Blocked on the maintainer

- **Source Cooperative:** an organisation account in the project's name (an email to hello@source.coop) and beta approval; the London bucket is `eu-west-2.opendata.source.coop`. The recorder does not start until Source Cooperative has approved the account and accepted the role ARN.
- **AWS:** an IAM role with `s3:PutObject` and `s3:PutObjectAcl` on the Source Cooperative prefix, an instance profile for the t4g.small in `eu-west-2`, and the role ARN sent to Source Cooperative.
- **Sentry:** the project's existing DSN, supplied as an environment variable on the instance.
- **Source Cooperative's post-beta pricing** is not asked for yet, by the maintainer's decision.

## Risks and open questions

- **Recording starts only after Source Cooperative approves the account.** The maintainer chose this to keep the design simple: there is no interim bucket and no local-disk-only mode. Every DWD run published before approval is lost permanently, so approval time is the cost. The local store is a working directory for one run at a time; the 100 GB disk also buffers about 4-7 days of runs if Source Cooperative is unreachable, and the free-space check warns through Sentry before it fills.
- **Ask dynamical.org.** It already records DWD ICON-EU as an Icechunk repository and hosts some datasets on Source Cooperative, and its `reformatters` code decodes DWD GRIB into Zarr. Reading that code before writing `dwd.py` is part of PR 1. Asking whether it would host ICON-EU-EPS, ICON-D2-EPS or MOGREPS-UK is the maintainer's decision, and it does not remove the need to record now.
- **Second recorder for DWD.** The AWS instance is a single point of failure for the DWD products. A second recorder must not write to the same keys: a `partial` upload would overwrite a `complete` store's chunks. It needs its own prefix and a merge, or a conditional write (`If-None-Match: *`) on the root so only the first commit wins. Recommendation: defer, and design it as its own issue.
- **Independent check of the public bucket.** Sentry sees only what the recorder reports. A once-a-day job that lists the public bucket against the expected-run table would catch a recorder that reports healthy and uploads nothing. Recommendation: add it later if the first months show a need.
- **Post-beta storage cost.** At S3 London list price, 11 TB is about 200 GBP a month. The maintainer excludes storage from the 25 GBP budget.
- **Licence.** MOGREPS-UK is CC BY-SA 4.0 and goes out as a separate product from the CC BY 4.0 DWD products.
- **A silent operational-model change.** DWD switched ICON and ICON-EU to a prognostic aerosol scheme on 2026-09-02, so ICON-EU output has a step change on that date. Each store records the ICON model version where the GRIB header carries it.

## Hosting cost (hosting only, storage excluded)

At `eu-west-2` on-demand list prices, with $1 = 0.75 GBP: a `t4g.small` about 13.7 USD a month, 100 GB gp3 about 9.3 USD, and a public IPv4 address about 3.7 USD (`opendata.dwd.de` has no AAAA record), so about 26.7 USD, roughly 20 GBP, inside the 25 GBP cap. Data transfer in, and from the instance to S3 in the same region, is free. Two items could push it over: `t4g` instances default to unlimited CPU credits, so the instance is launched with `CpuCredits=standard` and the dry run measures average CPU, and a Sentry Crons monitor may be a paid feature on the project's plan, which the maintainer must confirm. A line goes into `docs/architecture/aws-costs.md`.

## Review 1 (simplicity): what changed and what was rejected

Adopted: ship DWD first and MOGREPS-UK second; drop `reconcile.py`; fetch expected files and treat 404 as "not yet" instead of listing directories; build a local store and upload with the root `zarr.json` last, with the manifest fields as root attributes (drops `manifest.py`, `_staging/` and the `patito`/`polars` dependency); drop SHA-256 per array in favour of the `crc32c` codec; drop the validator as a module and the physical checks as a gate; store MOGREPS-UK wind as delivered; use Sentry instead of the GitHub Actions manifest check; collapse to four modules; store the grid coordinates in every run; use `numcodecs.BitRound`; chunk per member or shard; correct 50 to 49 steps and the deterministic-product path; correct the location of `content.log.bz2` and the three items listed as unverified.

Rejected: the claim that Source Cooperative has no London region (its infrastructure page lists London and the bucket `eu-west-2.opendata.source.coop`), so the cross-region cost and the 29 GBP estimate do not apply. Also rejected: dropping ICON-D2 deterministic because Open-Meteo holds a rolling window of it, because that window is about 90 days, is not confirmed as permanent, and lacks the ensemble comparison's fields at native resolution; the maintainer asked for it.

## Review 2 (correctness): what changed and what was rejected

Adopted, and now in the sections above: step lists per field (ICON-D2 radiation has 193 files, ICON-ART-EU 64 to 89 steps); bitmapped cells as NaN; per-step checkpoint chunks and a single rechunk at commit; retention-based deadlines, a fetch pass before the deadline check, one `missing` report per run; a `state.json` sidecar, local deletion after upload and immutability after commit; a per-run exception boundary and a free-space check; a stated hosting cost; a "not yet" rule for truncated or mismatched files; numpy bit rounding with specification codecs only; consolidated metadata and awaiting all other objects before the root; HHL levels named; and the removal of the claim that a second recorder can write to the same keys. Also adopted for the tests: an injectable clock, a fake Sentry, a bitmapped-message case, a truncated-body case, an `ENOSPC` case, a fake store that records PUT order, and moto with the reset-per-test rule.

The docs page says to open the run root (never an array path), because a reader that opens an array path bypasses the commit.

Not applicable: the finding that the package must go in the dev dependency group so the nightly network tests can import it. That concerns this workspace, and the code now lives in `nwp-archivist` with its own CI.

Questions for Source Cooperative: whether `s3:PutObjectAcl` is needed given public read, and that buckets keep non-current versions for 90 days, so every overwrite keeps a copy.
