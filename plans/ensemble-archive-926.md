# Plan: always-on archive of short-retention ensemble weather products (#926)

**The problem.** DWD keeps only the last 4 runs of ICON-EU-EPS and the last 8 runs of ICON-D2-EPS (about 24 hours), and the Met Office keeps MOGREPS-UK for about 33 days. Nobody archives these products, so a later study cannot compare ensemble means against deterministic products (UKV, ICON-EU, ICON-D2, ECMWF ENS) for Great Britain solar and wind forecasting, onshore and offshore. Every day the recorder is down is lost permanently for the DWD products.

**The plan.** A new public repository, `openclimatefix/nwp-archivist` (pull requests limited to collaborators), holds a small recorder that runs on an AWS t4g.small in London under a systemd timer. The recorder computes which product runs should exist, fetches each expected file (a 404 means "not yet"), crops it to a "fat Great Britain" box on the native grid, builds the run member by member from cached raw files, and appends it to the product's Icechunk repository on Source Cooperative in one commit. Missed or partial runs are reported to Sentry. The work ships in two pull requests: the DWD products first, because DWD deletes each run after 24 hours, then MOGREPS-UK, which has about four weeks of slack. The recorder starts only after the accounts in "Blocked on the maintainer" exist.

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

- #801 proposed parquet per run cropped to a trial-area box. This plan uses one Icechunk repository per product, holding Zarr v3 arrays on the native grid cropped to a fat Great Britain box (the maintainer requires one array per product), because a later study may use any region and members are needed for calibration.
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

## Storage format: Icechunk, one repository per product

The maintainer requires one array per product, so a study opens a single array and slices `init_time`, member and step. The reviewed plan wrote one Zarr store per run, which cannot give that. Icechunk is the tool that gives it with transactions: one commit per run is atomic across every variable, a crash before the commit leaves the previous snapshot readable, and no reader sees a half-written run. Facts checked on 2026-09-25: the `icechunk` 2.2 series (the lockfile pins 2.2.0; the newest release is 2.2.2) has `manylinux_2_28_aarch64` wheels (the t4g.small is arm64), it is already in this repository's lockfile (through `dynamical-catalog`), it needs conditional writes (fine on direct S3), and dynamical.org and `source.coop/bkr/metoffice` publish Icechunk repositories on Source Cooperative.

**Compute on the small instance is modest.** The reviewer measured about 15 ms to decode one ICON-D2-EPS file and 7 ms for a per-step chunk write on a fast workstation, and about 200,000 files a day, so decode and write come to roughly one to two vCPU-hours a day against a `t4g.small` baseline of 9.6 (a Graviton core is slower, so the dry run measures it). The design avoids the expensive case, rewriting per-member chunks step by step, by caching decoded, cropped values until the run is complete and then building it member by member so each chunk is written once. Peak memory building a full ICON-D2-EPS run was measured at 440 MB on two cores. The instance never stores raw files long-term: keeping raw GRIB would be about 45 TB a year, roughly 4 to 6 times the cropped size.

**Risks specific to Icechunk, for the diff reviews:** commit conflicts (single recorder, so none expected); the size of the chunk-reference manifests, which grow with each run (with a split every 64 init times, a commit writes about 157 kB and takes about 45 ms, flat as the repository grows; the third review measured this); and whether the repository can be written through Source Cooperative's direct-write route (Option 3). Icechunk reads as well as writes, so the IAM role needs `s3:GetObject`, `s3:DeleteObject`, `s3:ListBucket` and the multipart permissions on the prefix (without `ListBucket`, S3 returns 403 rather than 404 for a missing key), and the required `bucket-owner-full-control` ACL is set with `s3_storage(write_headers=...)`. The dry run writes to a local path and cannot confirm the route, so a smoke test against the real bucket runs as soon as the account is approved.

## Where the code lives

The maintainer chose a separate repository, `openclimatefix/nwp-archivist`, created public with pull requests restricted to collaborators (the repository setting `pull_request_creation_policy=collaborators_only`). The reasons: the recorder shares no code with the forecasting packages, it should deploy from a pinned tag rather than track `main`, and a public archive with its own licences is easier to cite and reuse as its own repository. Issue #926 and the consumer-facing docs page stay in this repository. This plan file and draft PR #927 stay here until the plan is approved; the implementation PRs then open in `nwp-archivist`.

## Sequencing: two pull requests under #926

1. **PR 1 (this branch): the DWD products** ICON-EU-EPS, ICON-D2-EPS, ICON-D2 and ICON-ART-EU deterministic, the recorder, the package README, the docs page, and the deployment files. This is the urgent PR.
2. **PR 2: MOGREPS-UK**, adding `mogreps.py` (byte-range HDF5 reads, the 15-minute versus hourly steps, the 100 m slab, a 28-day retry window with per-run back-off) and its tests. It ships as a separate product because of the CC BY-SA licence.

## What changes, file by file (PR 1)

New Python package in the `nwp-archivist` repository (its own `pyproject.toml`, `uv_build`, Python 3.14, the house ruff, `ty` and pre-commit settings copied from this repository; dependencies `eccodes`, `icechunk`, `zarr>=3`, `numpy`, `httpx`, `sentry-sdk`, and `obstore` or `s3fs` for the upload):

- `products.py` — a frozen table of `Product` records (name, provider, licence, cycle hours, members, path template, and per field the levels and the step list) and a function that returns the exact list of expected file URLs for a run. Step lists differ by field, not by product: ICON-D2 `ASWDIR_S` and `ASWDIFD_S` have 193 files a run (every 15 minutes to 48 h) while its other fields have 49; ICON-ART-EU has 75 steps for `ASWDIR_S`, `ASWDIFD_S`, `T_2M` and `CLCT`, 64 for `ASOB_S_CS` and 89 for `TAOD_DUST`. Each field is stored with its own `step` coordinate. The recorder, the tests and the docs read this one table.
- `dwd.py` — fetch one file with keep-alive, retry with exponential backoff and jitter, and an optional rate limit; decode one GRIB2 message with eccodes. A 404, a short body (`Content-Length` mismatch), a decode failure, or a decoded message whose `shortName`, `stepRange`, `perturbationNumber` or level does not match the URL is "not yet": the file is not marked received and is retried next cycle, and it becomes a Sentry warning only if it persists at the deadline. Cells masked by a GRIB bitmap (eccodes returns them as 9999.0; 3,894 of the 92,512 ICON-D2 cells in the box are masked) become NaN.
- `store.py` — crop to the box and append each run to the product's Icechunk repository, one repository per product, on the Source Cooperative bucket directly. Each variable is one array `(init_time, member, step, cell)` (ICON-D2 and ICON-ART-EU have no `member`), so a study opens one array per product and slices it. Each product has one `step` axis long enough for its longest field (ICON-D2: 193 steps every 15 minutes; a field with fewer steps is NaN-padded, which measured 0.6% more storage), with a `step` coordinate. The `init_time` axis is a grid of expected init times, and each run is written to the slot computed from its init time, so runs committed out of order (a `partial` run commits at its deadline, after later runs) leave the axis sorted, a repeated run lands in the same slot, and a missing run is a NaN slot at no storage cost. Icechunk manifest splitting is turned on (a split every 64 init times), because without it each commit rewrites each array's whole manifest and a year of runs grows to about 12 MB per commit. Status lives in its own arrays along `init_time` (status, expected and received file counts, archive time, code version, ICON model version), so an xarray reader sees it; the commit message repeats it. The grid coordinates `clat`, `clon`, `hsurf`, `fr_land` and `hhl` (levels 62-64 for ICON-D2 and ICON-D2-EPS, 72-74 for ICON-EU-EPS) are stored once per product. Every run's grid is compared with the stored one, and a change in grid, member count or step list stops commits for that product and reports through Sentry, once, rather than retrying forever; a new repository version is then a deliberate change. Per-file decoded, cropped values are cached locally as `.npy` files as they arrive (the checkpoint; cropped files are far smaller than raw GRIB, so the disk holds days rather than hours of runs), and once the run is complete or its deadline has passed the recorder builds the run member by member, so each per-member chunk `(1, 1, steps, cells)` is written once. Values get 13-bit significand rounding in numpy before the write, with the Icechunk default zstd codec. One Icechunk commit per run is the commit. The repository's status arrays are the record of what is committed; a local tombstone is only a cache of that record, so a rebuilt instance never commits a run twice (repeats are kept forever, as no snapshot expires). After the commit succeeds the local files for the run are deleted. There is no garbage collection and no snapshot expiry, because the archive never deletes.
- `recorder.py` and a small CLI (`archive-record`) — every 15 minutes: compute expected runs, start each run at a fixed delay after init, fetch every expected file not yet cached locally, commit when the count reaches the expected count or the deadline passes, then append the run to the Icechunk repository. DWD keeps a run for about 25-26 hours, so the deadline is set from retention (init + 23 h for ICON-D2-EPS and ICON-D2, init + 24 h for ICON-EU-EPS and ICON-ART-EU), the loop looks back to the retention horizon, and it always makes one fetch pass before it evaluates the deadline, so an instance that restarts late still fetches what DWD holds. The clock is injectable. A `missing` run (nothing arrived by the deadline) is recorded as a tombstone and one Sentry event, reported exactly once. Each run is wrapped in its own `try`/`except` that sends the error to Sentry and moves on, and a free-space check before each run warns through Sentry when the disk is nearly full, so a full disk or a decode error in one run cannot stop the others. Each cycle sends a Sentry cron check-in, and each `partial` or `missing` run sends a warning event tagged with the product and init time. Nothing in the recorder raises on an absent or late file.
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

The recorder runs in production, unattended, so it follows `docs/design-philosophy/inherent-stability.md`: an absent or late provider file never raises; the run is retried, and at the deadline the archive commits what exists marked `partial`, with expected and received counts in the status arrays. Structural problems (a decoded shape that differs from the expected grid) are recorded on the run and the data is still kept, because for a source that deletes its copy within 24 hours, discarding a run is losing it. Sentry events name the product and the init time (principle 16), and the cron check-in makes a dead instance visible. The capability lives outside the forecast serving path. No Patito contract is added or changed.

## Tests

Each new test states the assertion that fails on `main` today (the package does not exist), and the interesting ones are these:

- `test_products.py`: the expected file list for one ICON-D2-EPS run has 20 members x 49 steps per surface field; a deterministic product's URLs contain no `/e/` segment; ICON-D2 `ASWDIR_S` has 193 steps and its `T_2M` 49; ICON-ART-EU `TAOD_DUST` has 89.
- `test_dwd.py` (stub HTTP server): a 404 leaves the file unwritten and the run incomplete; a later 200 completes it; a 500 then a 200 is retried once.
- `test_store.py`: a synthetic triangular grid keeps exactly the cells in the box, and a cell 0.01 degrees outside is dropped; writing the same run twice gives identical decoded arrays; a bitmapped message decodes to NaN, never 9999; against a `ThreadedMotoServer` with an explicit endpoint (in-process `mock_aws` does not intercept Icechunk's own S3 client, and a test request went to real AWS), reset per test, an Icechunk commit interrupted before it completes leaves the previous snapshot readable and the run still expected, and appending the same run twice, with the local tombstones deleted, leaves one `init_time` slot with identical values; runs committed out of order leave the `init_time` axis sorted; a run whose grid differs from the stored grid is not committed and sends one Sentry event; a storage error at commit sends one Sentry event, raises nothing, and leaves the run expected; a run built member by member equals the same run written all at once.
- `test_recorder.py`: at the deadline a run with 99 of 100 files is committed `partial` with counts 99/100; a run with none is `missing` and sends one Sentry event; neither raises; a re-run after `complete` fetches nothing. The tests inject the clock and a fake Sentry: a restart at init + 23 h with a half-filled local store fetches the missing files and commits `complete`; a restart at init + 30 h reports the run `missing` exactly once; a truncated 200 response and a header that disagrees with the URL both leave the file not received; an `ENOSPC` error in one run sends one Sentry event and lets the next run proceed.
- Network-gated tests (`--run-network -m network`): one real ICON-D2-EPS field, one real ICON-EU-EPS field and one real ICON-D2 field, each decoded and range-checked.

## Docs to update

`docs/architecture/ensemble-archive.md` (new, here), the two link pages, and `mkdocs.yml`. The `nwp-archivist` README carries the licences, attribution and lineage. Issue #801 gets a comment linking #926 and stating what remains open there.

## Verification commands

`uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest && uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md && uv run mkdocs build --strict`, then `uv run pre-commit run --all-files` and the pydoclint and docs-link checks CI runs. Before any run against real sources, an Opus review of every script. The dry run stores to a local path and is checked with the `data-validation` skill's checklist as a one-off script (plausible ranges, day-night profile, orientation), not as a module on the instance.

## Blocked on the maintainer

- **Source Cooperative:** an organisation account in the project's name (an email to hello@source.coop) and beta approval; the London bucket is `eu-west-2.opendata.source.coop`. The recorder does not start until Source Cooperative has approved the account and accepted the role ARN.
- **AWS:** an IAM role with `s3:PutObject` and `s3:PutObjectAcl` on the Source Cooperative prefix, an instance profile for the t4g.small in `eu-west-2`, and the role ARN sent to Source Cooperative.
- **Sentry:** the project's existing DSN, supplied as an environment variable on the instance. The maintainer has confirmed the project's Sentry plan includes Sentry Crons.
- **Source Cooperative's post-beta pricing** is not asked for yet, by the maintainer's decision.

## Risks and open questions

- **Recording starts only after Source Cooperative approves the account.** The maintainer chose this to keep the design simple: there is no interim bucket and no local-disk-only mode. Every DWD run published before approval is lost permanently, so approval time is the cost. The local store is a working directory for one run at a time; the 100 GB disk holds cropped, decoded files for the runs in flight and buffers some days of runs if Source Cooperative is unreachable, and the free-space check warns through Sentry before it fills.
- **Ask dynamical.org.** It already records DWD ICON-EU as an Icechunk repository and hosts some datasets on Source Cooperative, and its `reformatters` code decodes DWD GRIB into Zarr. Reading that code before writing `dwd.py` is part of PR 1. Asking whether it would host ICON-EU-EPS, ICON-D2-EPS or MOGREPS-UK is the maintainer's decision, and it does not remove the need to record now.
- **Second recorder for DWD.** The AWS instance is a single point of failure for the DWD products. Icechunk commits are transactional, so a second recorder appending to the same repository conflicts and must rebase. Design it as its own issue. Recommendation: defer, and design it as its own issue.
- **Independent check of the public bucket.** Sentry sees only what the recorder reports. A once-a-day job that lists the public bucket against the expected-run table would catch a recorder that reports healthy and uploads nothing. Recommendation: add it later if the first months show a need.
- **Post-beta storage cost.** At S3 London list price, 11 TB is about 200 GBP a month. The maintainer excludes storage from the 25 GBP budget.
- **Licence.** MOGREPS-UK is CC BY-SA 4.0 and goes out as a separate product from the CC BY 4.0 DWD products.
- **A silent operational-model change.** DWD switched ICON and ICON-EU to a prognostic aerosol scheme on 2026-09-02, so ICON-EU output has a step change on that date. Each store records the ICON model version where the GRIB header carries it.

## Hosting cost (hosting only, storage excluded)

At `eu-west-2` on-demand list prices, with $1 = 0.75 GBP: a `t4g.small` about 13.7 USD a month, 100 GB gp3 about 9.3 USD, and a public IPv4 address about 3.7 USD (`opendata.dwd.de` has no AAAA record), so about 26.7 USD, roughly 20 GBP, inside the 25 GBP cap. Data transfer in, and from the instance to S3 in the same region, is free. Two items could push it over: `t4g` instances default to unlimited CPU credits, so the instance is launched with `CpuCredits=standard` and the dry run measures average CPU. A line goes into `docs/architecture/aws-costs.md`.

## Review 1 (simplicity): what changed and what was rejected

Adopted: ship DWD first and MOGREPS-UK second; drop `reconcile.py`; fetch expected files and treat 404 as "not yet" instead of listing directories; record status in the commit (drops `manifest.py`, `_staging/` and the `patito`/`polars` dependency); drop SHA-256 per array in favour of the `crc32c` codec; drop the validator as a module and the physical checks as a gate; store MOGREPS-UK wind as delivered; use Sentry instead of the GitHub Actions manifest check; collapse to four modules; store the grid coordinates in every run; use `numcodecs.BitRound`; chunk per member or shard; correct 50 to 49 steps and the deterministic-product path; correct the location of `content.log.bz2` and the three items listed as unverified.

Rejected: the claim that Source Cooperative has no London region (its infrastructure page lists London and the bucket `eu-west-2.opendata.source.coop`), so the cross-region cost and the 29 GBP estimate do not apply. Also rejected: dropping ICON-D2 deterministic because Open-Meteo holds a rolling window of it, because that window is about 90 days, is not confirmed as permanent, and lacks the ensemble comparison's fields at native resolution; the maintainer asked for it.

## Review 2 (correctness): what changed and what was rejected

Adopted, and now in the sections above: step lists per field (ICON-D2 radiation has 193 files, ICON-ART-EU 64 to 89 steps); bitmapped cells as NaN; a local checkpoint of cropped `.npy` files, built member by member at commit; retention-based deadlines, a fetch pass before the deadline check, one `missing` report per run; status arrays in the repository as the record of what is committed, with a local tombstone as a cache only, and local deletion after commit; a per-run exception boundary and a free-space check; a stated hosting cost; a "not yet" rule for truncated or mismatched files; numpy bit rounding; HHL levels named; and the removal of the claim that a second recorder can write to the same keys. Also adopted for the tests: an injectable clock, a fake Sentry, a bitmapped-message case, a truncated-body case, an `ENOSPC` case, and a `ThreadedMotoServer` reset per test.

Superseded by the Icechunk decision in "Storage format": readers open the repository, which is transactional, so partial arrays are never visible.

Not applicable: the finding that the package must go in the dev dependency group so the nightly network tests can import it. That concerns this workspace, and the code now lives in `nwp-archivist` with its own CI.

Questions for Source Cooperative: whether `s3:PutObjectAcl` is needed given public read, and that buckets keep non-current versions for 90 days, so every overwrite keeps a copy.

## Review 3 (Icechunk storage): what changed and what was rejected

Adopted: the manifest split every 64 init times (without it a year of runs grows each commit to about 12 MB); each run written to the slot computed from its init time, so late `partial` commits leave the axis sorted and missing runs are NaN slots; status in its own arrays along `init_time` (the coordinate cannot carry it); one padded `step` axis per product instead of a group per step list; the wider IAM policy and the `bucket-owner-full-control` header; the repository's status arrays as the record of what is committed, with the tombstone a cache only; a grid-change stop that reports once instead of retrying; cropped `.npy` files as the local checkpoint, because raw GRIB fills 100 GB in about a day if uploads stop; the corrected chunk shape `(1, 1, steps, cells)`; `ThreadedMotoServer` in place of in-process moto; and the new tests (out-of-order commits, grid mismatch, storage error at commit, idempotency with tombstones deleted). The review showed Icechunk commits survive `kill -9` at eight points, and that building a full ICON-D2-EPS run peaks at 440 MB.

Nothing from this review was rejected. The review used the lockfile's `icechunk` 2.2.0, so the dry run repeats the manifest-split measurement on the version that ships.
