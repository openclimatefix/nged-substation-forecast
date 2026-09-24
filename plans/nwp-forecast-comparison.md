# Plan: compare weather forecasts for power at matched lead times (#810)

**The problem.** Every weather-product comparison this project has published scores a description
of an hour that has already happened: a reanalysis, a satellite retrieval, or the first hours of an
NWP run. The live service forecasts power from a day-ahead weather forecast, and a product that wins
at lead zero can lose at lead 24 hours. The one forecast measured so far, ECMWF ENS on the
[ENS horizons page](https://openclimatefix.github.io/nged-substation-forecast/studies/ens-forecast-horizons/),
was only compared with itself, with no-weather baselines, and with past-weather references. Nobody
has measured whether another forecast model, or a blend of forecast models, beats ENS at the
day-ahead lead the live service delivers.

**The planned solution.** One new study scores the power forecast an XGBoost model makes from each
forecast product at day-ahead (day 1), and at days 2, 3, 5, and 7, on the same hours at 6 solar
farms (A to F) and 3 wind farms (W1 to W3), from 1 December 2024 to 10 September 2026. Two families
of product are compared. Whole-run archives (ECMWF ENS, NOAA GEFS, and NOAA GFS, all from
Dynamical.org) give each product's 00 UTC run, read at 09:00 UTC, so every row's lead is identical
across products and every run was published before the forecast is issued. Open-Meteo's Previous
Runs archive gives nine more products (UKV, ICON-D2, ICON-EU, ICON global, ECMWF IFS 0.25°, GFS,
ARPEGE, AROME, and two HARMONIE-AROME models) at a fixed day offset, whose lead depends on each
model's run cycle. Those products are compared with ENS by **bracketing**: ENS's day-0 lead is
shorter and its day-1 lead longer than every Previous Runs day-1 lead on the same row, so a product
that beats ENS at day 0, or loses to ENS at day 1, is better or worse at a matched lead. A blend of
the open-data forecasts is scored against ENS alone, with the blending study's permutation control,
at an optimistic and a conservative lead. The page ends with a "What to use" verdict per technology.

## Verdict, size, and departures from the issue

**Verdict: worth doing, roughly as described.** The issue's premise holds: nothing on `main`
compares forecast products at a forecast lead. The data is on disk for the Previous Runs products,
and GFS and GEFS are downloading.

**Size: complex.** Trigger answers:

- **What gets stored:** yes. A published docs page, its charts, and study outputs under
  `data/studies/nwp_forecast_comparison/`.
- **Production serving path:** no. Nothing in `src/` or the production packages changes.
- **Degradation rule:** no.
- **More than one defensible design:** yes. How to match leads across archives with different run
  cycles, which contrasts decide, how to treat ensembles, and which products go in a blend all have
  several defensible answers.
- **Callers not nameable without searching:** no. The study adds new scripts and new functions in
  `packages/studies/`, and moves one fold helper there; its callers are the study scripts named
  below.

**Reviews bought:** both plan reviews (simplicity, then correctness), both diff reviews
(correctness-and-cut, then the mutation pass, which runs because `packages/studies/` changes), two
Opus science reviews each followed by a fixer and any re-run a reviewer asks for, persona reviews
(the builders of each product compared, and the users), a `prose-review` sweep, and a pre-merge
check of the rendered HTML.

**Departures from the issue body:**

- **Previous Runs is not a fixed-lead instrument, so the plan does not treat it as one.** The issue
  calls the Previous Runs archive "the instrument this study wants". Each `previous_dayN` value comes
  from the freshest run made at least N × 24 hours before the hour, so its lead is `24N + (h mod
  n)` for a model run every `n` hours, where `h` is the UTC hour. Leads differ between products by
  up to 5 hours, and no Previous Runs value reproduces a fixed daily issue time. The plan matches
  leads exactly where the archive allows (whole-run products; Previous Runs products at hours where
  `h mod n` is 0 for every product compared) and brackets otherwise.
- **Blends are planned at day 1 only**, with day 2 and 3 blends exploratory. The issue asks for
  blends "at each lead"; five leads of blends with controls and a second setting would triple the
  planned contrasts.
- **The ENS horizons page is not rewritten here.** Issue #892 owns re-running that page with an era
  cut. This study starts its rows on 2024-12-01, after IFS Cycle 49r1, so its own ENS figures do not
  cross the step, and the report prints this study's ENS day-1 error beside the horizons page's as
  an exploratory reconciliation.
- **The `ens_horizons.py` rewrite the issue comment mentions is out of scope.** This study reads the
  ENS members through `ens_forecast_horizons.py`'s upsampling, which already uses
  `studies.cross_validation`, and never calls `ens_horizons.py`.
- **ICON global, ARPEGE, AROME, and the HARMONIE-AROME models are scored though the issue does not
  name them.** Open-Meteo serves them, so the issue's rule ("the models Open-Meteo serves") admits
  them, and they cost one fit each. They are exploratory.

## The question and the products

**Question: at the day-ahead lead the live service delivers, and the days around it, which forecast
product, or which blend of products, gives the most accurate power forecast at these farms?**

| Product | Archive on disk | Runs used | Lead at day 1 | Day offsets | Rows from |
|---|---|---|---|---|---|
| ECMWF ENS (mean of 51 members; control member exploratory) | Dynamical.org via `ens_forecast_horizons/ens_members.parquet` | 00 UTC | 24 + h (day 1); h (day 0) | every day | 2024-04-01 |
| NOAA GEFS (mean of 31 members; control exploratory) | Dynamical.org, `weather/GEFS/` (downloading) | 00 UTC | 24 + h | every day | 2020-10 |
| NOAA GFS | Dynamical.org, `weather/GFS/` (downloading) | 00 UTC | 24 + h | every day | 2021-05 |
| UKV | Open-Meteo Previous Runs | freshest ≥ 24 h before | 24 + (h mod n) | 1 | 2024-08-06 |
| ICON-D2 | Previous Runs | as above | as above | 1 | 2024-01-19 |
| ICON-EU | Previous Runs | as above | as above | 1–4 | 2024-01-19 |
| ICON global | Previous Runs | as above | as above | 1–6 | 2024-01-19 |
| ECMWF IFS 0.25° (the deterministic IFS, ECMWF open data) | Previous Runs | as above | as above | 1–7 | 2024-03-06 |
| GFS (Open-Meteo "GFS seamless") | Previous Runs | as above | as above | 1–7 | 2024-01-19 |
| ARPEGE Europe | Previous Runs | as above | as above | 1–3 | 2024-01-19 (solar only; its 100 m wind offsets start 2026-08) |
| AROME France | Previous Runs | as above | as above | 1 | 2024-01-19 (solar only, same reason) |
| KNMI HARMONIE-AROME Europe, DMI HARMONIE-AROME | Previous Runs | as above | as above | 1 | 2024-06 (KNMI wind from 2024-10) |
| ECMWF IFS HRES 9 km | Previous Runs | as above | as above | 1–7 | 2025-10-01: exploratory, own row set |
| ERA5, CAMS (solar) | past-weather studies | not forecasts | reference rows | — | — |

The run cycle `n` of each Previous Runs product is measured, not assumed (verification V1 below).
The implementation fills in each product's grid, domain, and publication time from its producer's
documentation; the table on the page carries them.

**Every run a row uses was published before the forecast it feeds is issued.** For the whole-run
products the issue time is 09:00 UTC on the run's day, the live service's
`NWP_PUBLICATION_DELAY_HOURS`. Verification V4 checks each 00 UTC run's publication time against its
producer's dissemination schedule (ECMWF disseminates HRES by about 06:55 UTC, and Dynamical.org's
ENS is readable from about 09:00 UTC) and the page cites them. A Previous Runs value comes from a run
initialised at least 24N hours before the hour; the page states each product's publication delay
beside it, and the lead used for matching is the lead from initialisation, which no publication delay
changes.

**CERRA** joins as an exploratory solar reference row if the coordinator confirms it validated; it
is a reanalysis, not a forecast. **Open-Meteo Single Runs** of IFS HRES 9 km (00 UTC runs from
2024-03-14), and of UKV and ICON-EU from 2026-04-02, have been requested from the download
coordinator. If they arrive, they add optional exploratory arms (an exact-lead, 09:00-issue ECMWF
deterministic arm; a 6-month check of what a 09:00 UTC issue really gets from UKV and ICON-EU). No
planned contrast and no verdict depends on them.

## Leads, and what "matched" means

**Lead is measured from the run's initialisation to the valid hour** (the end of the hour for an
hourly-mean radiation value). Every row of every arm carries its lead in the saved inputs.

- **Family 1, whole-run products, exact lead.** Day `d` forecasts calendar day `D + d` from the 00
  UTC run of day `D`, as on the ENS horizons page. On a given row, ENS, GEFS, and GFS have the same
  lead, `24d + h`, and all three runs were published before 09:00 UTC on day `D`. A contrast between
  two Family 1 arms is exactly lead-matched and is exactly what a 09:00 UTC service could have run.
- **Family 2, Previous Runs products, day offset N.** Lead `24N + (h mod n)`. Two Family 2 products
  are lead-matched exactly at hours where `h mod n` is 0 for both, which for every cycle of 1, 3, or
  6 hours means 00, 06, 12, and 18 UTC. At other hours the product with the shorter cycle has the
  shorter lead, by at most 5 hours.
- **Bracketing a Family 2 product against ENS.** On the row for hour `h` of day `D + 1`, ENS day 0
  (the 00 UTC run of `D + 1`) has lead `h`, below 24; the product's day-1 value has lead `24 + (h
  mod n)`, from 24 to 29; ENS day 1 has lead `24 + h`, at or above it. If the product's error is
  below ENS day 0's, the product beats ENS even though ENS had the shorter lead on every row. If it
  is above ENS day 1's, ENS beats the product even though the product had the shorter or equal lead
  on every row. Anything between is **unresolved at matched lead**, and the page says so rather than
  choosing a side. The same bracket works at day `N`, using ENS days `N − 1` and `N`.
- **The bracket assumes error does not fall as lead rises.** The report checks this on the study's
  own rows: ENS day 0 against day 1 (the horizons page found day 1 worse by 0.61 points for solar and
  0.75 for wind), and each multi-day Previous Runs product's day 1 against its day 2. If either
  check fails, the bracket for that product is void and the page says so.

**What a 09:00 UTC service would really get from a Previous Runs product is not measured.** At 09:00
UTC the freshest UKV or ICON-EU run is a morning run, so for a daylight hour the service's lead is
longer than the Previous Runs day-1 lead (for example 18 + h from a 06 UTC run, against 24). Day 1
of Previous Runs therefore flatters those products slightly relative to what the service could
deliver for most daylight hours, and the page says so beside every Family 2 result. Day 2 of
Previous Runs (lead 48 or more) is always staler than the service's lead, which the conservative
blend below uses.

## Rows, folds, and fairness

- **Hours.** The past-solar and past-wind studies' hourly rows (every daylight hour at A to F; every
  hour at W1 to W3), with their commissioning-ramp, zero-hour, and outage exclusions, built through
  the ENS horizons page's `_base_frame`. Solar power is aggregated to the hour ending at the label,
  wind to the hour centred on it. Rows run from 2024-12-01 (the first whole month after IFS Cycle
  49r1, as the past-wind ENS and HRES study, #885, uses) to the rows' last date, 2026-09-10.
- **One shared row set per technology.** A row is kept only if the target and every main arm's input
  at every scored day are present, so every arm on the leaderboard is scored on identical rows. A
  product whose inclusion would drop more than 2% of rows (expected: HRES 9 km, ARPEGE and AROME for
  wind) is scored instead on its own subset, paired against ENS on that subset, as exploratory. The
  report prints the rows each product drops.
- **Folds: #885's design, confirmed by the study coordinator.** Five folds of whole months, cut
  inside three eras: before 2025-10-01; 2025-10-01 to the UKV PS47 upgrade on 2026-01-21; and after
  it, with the part-month that straddles the upgrade dropped as #885 drops it. The third era's fold
  numbers are rotated (`ERA_FOLD_OFFSETS`) so no calendar month is held out of every era at once
  (#868). Every arm gets the same `era_code` column. Before any fit, the script prints the (site,
  fold, calendar month) coverage table and lists the calendar months that occur in one year only.
  The 2025-10-01 cut also covers DWD's ICON low-cloud change of 2025-09-24 to within a week; the page
  names the week.
- **The fold helper moves into `packages/studies/`**, so #885 and this study share one
  implementation. Whichever PR merges first owns it; the other rebases onto it.
- **Robustness to fold design (exploratory):** the planned arms refitted with an extra era cut at
  IFS Cycle 50r1 (2026-05-12), dropping the part-month of May 2026, as #885 does.
- **Equal columns, no column subsampling.** `colsample_bytree=1` (the primary and sensitivity
  settings already use it). Every solar arm gets the calendar columns (hour, day of year,
  `era_code`), the sun position, and two weather columns from its own product at its own lead:
  global horizontal irradiance and 2 m air temperature. Every wind arm gets the calendar columns and
  four weather columns: hub-height speed, hub-height direction as a sine and a cosine, and 10 m
  speed. Each arm type's columns come from one function returning a fixed-length tuple, and every
  arm's column list is printed into the report.
- **No direct-beam columns and no neighbouring-hour columns** for any single-product arm. Some
  products serve a native beam and some a separation-model beam, and the beam/diffuse study found
  the beam moves the error by about 0.1 points; leaving it out keeps the arms equal.
- **Hub height.** ICON's served "100 m" wind is its 120 m wind times about 0.98, which a per-generator
  tree model treats exactly as the 120 m speed, so ICON arms read 120 m against the others' 100 m;
  the page names this. Every speed is converted to m/s.
- **Spatial sampling differs by archive and is named, not equalised.** ENS is averaged over each
  generator's H3 cell (the file on disk); GFS and GEFS are read at the nearest 0.25° cell; Open-Meteo
  products at the nearest land cell Open-Meteo picks. An exploratory refit of GFS at day 1 from the
  mean of the four surrounding cells shows how much the sampling moves one product.
- **Temporal steps.** ENS and GEFS are 3-hourly, upsampled per member through the clear-sky index for
  radiation (the horizons page's chosen method) and by eastward and northward components for wind
  speed, before any member mean. GEFS radiation alternates 3- and 6-hour averaging windows, and
  GFS's averages reset every 6 hours, so both are converted to per-step means first (verification
  V2). Open-Meteo interpolates IFS 0.25°'s 3-hourly steps itself.
- **UKV's served radiation is a snapshot**, so UKV's hourly value is rebuilt as the mean of the
  snapshots at the hour's start and end (`studies.hourly_means.hourly_from_snapshots`), as on the
  past-solar page. Both snapshots come from the day-1 column.
- **Ensembles.** The planned ENS and GEFS arms read the mean of the members' upsampled fields, the
  treatment the horizons page found best at day 1. The control members are exploratory arms, because
  a control member against the mean separates ensemble averaging from the model. Per-member fitting
  is not repeated (the horizons page settled it for ENS).
- **XGBoost model.** `studies.cross_validation.out_of_fold_losses`: per generator, absolute-error
  objective, three fitting seeds, curtailed hours left out of training and kept for scoring, the
  prediction held to the export cap. Each error is divided by its own generator's capacity (99th
  percentile of output, from the `effective_capacity` table; the page states which build) before any
  mean or difference.
- **Intervals.** `studies.bootstrap.bootstrap_difference` and `bootstrap_absolute`: 2,000 resamples of
  whole calendar months, each drawing one fitting seed.
- **Absolute skill beside every contrast.** A day-1 leaderboard per technology (every arm's own error
  with its interval) and an error-by-day table for every product. No-weather baselines at each day
  from `studies.baselines` (climatology and smart persistence, issued at 09:00 UTC) set the floor;
  ERA5, and CAMS for solar, are past-weather reference rows that no forecast could have had.

## Planned contrasts

Written here before any fit. Each is computed separately for solar and for wind, in percentage
points of capacity, first arm minus second, and also at the second hyperparameter setting
(`SENSITIVITY_HYPER_PARAMETERS`). Every other figure is exploratory.

| ID | Contrast | Lead matching | What it decides |
|---|---|---|---|
| P1a | UKV day 1 − ENS mean day 1 | bracket, upper side | Positive and significant: ENS beats UKV at matched lead |
| P1b | UKV day 1 − ENS mean day 0 | bracket, lower side | Negative and significant: UKV beats ENS at matched lead |
| P2a | ICON-EU day 1 − ENS mean day 1 | bracket, upper side | as P1a, for ICON-EU |
| P2b | ICON-EU day 1 − ENS mean day 0 | bracket, lower side | as P1b, for ICON-EU |
| P3 | GEFS mean day 1 − ENS mean day 1 | exact | Is NOAA's ensemble a substitute for ENS at a 09:00 UTC issue? |
| P4a | Blend (ENS mean day 1 + ICON-EU day 1 + IFS 0.25° day 1) − its control | optimistic: the other products' leads are Previous Runs day 1 | Does other products' weather add to ENS at day-ahead? |
| P4b | Blend (ENS mean day 1 + ICON-EU day 2 + IFS 0.25° day 2) − its control | conservative: every other product's lead is at least what a 09:00 UTC service would have | as P4a, as a lower bound |

**Why these products.** UKV is the Met Office's convection-permitting UK model, free on AWS. ICON-EU
was the best partner for UKV on the blending page and is free from DWD. Both cover all of NGED's
licence area. ENS is the incumbent. GEFS is the only other whole-run ensemble on disk. The blend's
products are the open-data forecasts with a day-2 offset in Previous Runs, so the same three products
appear at both bounds and only the lead changes between P4a and P4b. UKV has no day-2 offset, so a
blend with UKV (ENS + UKV + ICON-EU + IFS 0.25° at day 1) is exploratory.

**The blend's control** keeps ENS's real columns and replaces the other products' columns with
`studies.blending.climatology_permutation` (shuffled among hours sharing a generator, calendar month,
and hour of day), so the control has the blend's column count. The blend minus its control is the
other products' weather; the control minus ENS alone is what the extra columns add with no weather in
them (exploratory). A synthetic positive control (measured output plus fixed-seed noise, as on the
blending page) added to ENS at day 1 shows the pipeline can detect a gain (exploratory).

**Verdict rules, fixed now:** a product "beats ENS at matched lead" only if its lower-side contrast
is negative and statistically significant at the 5% level; "loses to ENS at matched lead" only if its
upper-side contrast is positive and significant; otherwise "unresolved at matched lead". A blend
"adds to ENS" if P4b is negative and significant (the gain survives the conservative lead); "may add"
if only P4a is; "adds nothing detectable" if neither is, with the interval's bound on the effect
stated. Where the second setting changes a contrast's sign or significance, the page says so.

**Exploratory, labelled as such on the page:** the full day-1 leaderboard of every product; every
product at days 2, 3, 5, and 7 (Family 1 at every day; Family 2 where the archive offers the day) with
its bracket against ENS; the Family 2 products compared with each other at the exact-lead hours
(00, 06, 12, 18 UTC) beside all hours; the control members; GFS against ENS; IFS 0.25° against ENS;
the blend with UKV; blends at days 2 and 3; HRES 9 km on its own row set from 2025-10-01; CERRA; the
four-cell GFS sampling check; the 50r1 fold-design check; the ENS day-1 reconciliation with the
horizons page; the per-generator errors. About 1 in 20 exploratory intervals reaches significance at
the 5% level by chance, and the page says so.

## What no contrast here can separate

The page states these beside the results they touch:

- **Model against resolution and grid.** A product's win may come from its grid spacing, its grid
  cell, or the archive's cell choice rather than its physics.
- **Ensemble averaging against model quality.** A mean of members is smoother than any one run, and
  mean absolute error rewards smoothness; the control-member arms show how much of an ensemble arm's
  lead is averaging.
- **Archive against producer.** Open-Meteo re-serves each producer's field with its own
  interpolation, cell choice, and 100 m scaling; a gap may belong to the archive.
- **Step width.** ENS and GEFS reach this study 3-hourly and are upsampled; most Previous Runs
  products are hourly.
- **Lead inside a bracket.** An unresolved bracket means the product sits between ENS's day-0 and
  day-1 errors; the study cannot place it more precisely.
- **What a 09:00 UTC service gets from a Previous Runs product**, as set out above.
- **Model upgrades inside the row set.** IFS 50r1 on 2026-05-12 (checked by the extra-cut table),
  DWD's ICON changes of 2025-07-23 and 2026-09-02, ICON-D2's of 2026-02-18, and GEFS's radiation
  precision fix of 2026-06-15 are not era cuts.

## What changes, file by file

- `packages/studies/src/studies/cross_validation.py`: add `assign_era_folds` (era cuts, dropped
  part-months, rotated fold numbers per era) and `calendar_month_coverage` (the #868 table), moved
  from #885's `ens_hres_past_wind.py` if it merges first, written here otherwise. Tests in
  `packages/studies/tests/`.
- `packages/studies/src/studies/previous_runs.py` (new): `previous_runs_lead_hours(valid_time,
  day_offset, cycle_hours)` and `bracket_verdict(lower, upper)` returning "beats", "loses", or
  "unresolved" from two `BootstrapInterval`s. Tests.
- `packages/studies/src/studies/resample.py`: add `step_means_from_window_averages` for GEFS's
  alternating 3- and 6-hour windows and GFS's 6-hour resets. Tests with a hand-built window
  sequence whose true per-step means are known.
- `studies/nwp_forecast_comparison/` (new directory, own README):
    - `verify_previous_runs_leads.py`: V1 below.
    - `build_forecast_inputs.py`: per technology, one long parquet of every arm's weather columns at
      every scored day, with each row's lead, read from the Previous Runs files, the ENS members, and
      the GEFS and GFS month caches. Anonymised labels only.
    - `nwp_forecast_comparison.py`: rows, folds, fits, baselines, intervals, `report.md`, saved
      per-row losses and predictions, fingerprint (floats cast to Float32 before hashing),
      `--report-only` and `--fit-missing`.
    - `nwp_forecast_charts.py`: every chart, numbers read from the report.
- `studies/README.md`: add the study's row.
- `docs/studies/nwp-forecasts-at-matched-leads.md` (new page, the study skill's outline, with the
  AI disclaimer), `docs/studies/assets/nwp_forecasts_*.svg`, `docs/studies/index.md`, `mkdocs.yml`
  nav.
- `docs/roadmap/xgboost-improvements.md`, "Several NWP sources as features": add the measured
  evidence, including the multi-NWP figure the issue quotes (−0.30 points after #823's
  re-normalisation) and this study's P4a and P4b.
- `docs/background/weather-products-survey.md`, the "Candidates for comparing forecast models at a
  fixed lead" section: link to the page and correct any statement V1 contradicts.

## Verification before the first fit

- **V1, Previous Runs lead rule.** Compare Open-Meteo GFS's `previous_dayN` values with Dynamical.org
  GFS runs, and report which run and lead each matches best, per UTC hour; the plan's rule predicts
  the run initialised at `floor6(h − 24N)`. Measure each Open-Meteo product's cycle `n` from where the
  hour-to-hour jumps in its day-1 minus day-0 series fall, and from Single Runs where downloaded. If
  the rule is wrong, this plan's lead formula and bracket are revised before any fit.
- **V2, GEFS and GFS radiation.** After converting windows to step means: no negative means beyond a
  stated tolerance, no night-time radiation, and the member mean's correlation with CAMS highest at
  the right window (as the ENS README's check did).
- **V3, timestamp and height conventions** per Previous Runs product (the lineage notes: UKV
  snapshot, the rest hour-ending), and a power-hour offset scan for wind.
- **V4, publication times.** Each product's 00 UTC run publication time from its producer, cited.
- **V5, domains.** Which products cover NGED's whole licence area (ICON-D2 stops about 2.3°W), from
  `weather_product_domains.py` or each producer's stated domain.
- The data-validation checklist on every input frame the build writes.

## Charts

Every chart per the `dataviz` and study skills: OCF palette, SVG through `svgo`, `aria=False` on data
marks, generators labelled A to F and W1 to W3 only, values as % of capacity, time axes showing
days 1 to 7 with the month and year in text only.

1. Day-1 leaderboard per technology, each arm's error with its interval, smaller is better, family
   and lead band marked.
2. Planned contrasts P1 to P4, with brackets drawn as the two sides.
3. The XGBoost models work: day-1 out-of-fold forecasts against measured output, one week per
   generator chosen by a stated rule, and per-generator errors per arm.
4. Error against day for every product, with ENS's day 0 and day 1 bands shaded so brackets are
   visible.
5. Blend against its control and ENS alone, at both bounds.
6. Exact-lead hours against all hours for the Family 2 ranking.

## Tests

- `assign_era_folds`: a fixture spanning three eras where unrotated folds leave a calendar month with
  no training row, asserting the rotated assignment leaves one; and asserting the straddling
  part-month is dropped. Fails today because the function does not exist in the package.
- `calendar_month_coverage`: a fixture with one single-year month, asserting it is listed.
- `previous_runs_lead_hours`: hours 0, 5, 6, 13 at cycles 1, 3, 6 and offsets 1 and 2, against hand
  values.
- `bracket_verdict`: the three outcomes and the two boundary cases where an interval touches zero.
- `step_means_from_window_averages`: a synthetic series of known step means turned into GEFS-style
  and GFS-style window averages and back, exactly.

Each test must fail when the function's key line is mutated; the mutation pass checks this.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run pydoclint packages/studies/src studies/nwp_forecast_comparison
uv run mkdocs build --strict   # then read the rendered page's HTML
uv run python studies/nwp_forecast_comparison/nwp_forecast_comparison.py --report-only
```

A printed-number guard in the chart script checks every figure the page quotes against
`report.md`, and the pre-merge check compares the rendered page's numbers with the report.

## Process and constraints

- **No fit runs until the study coordinator confirms GEFS, GFS, and CERRA are complete and
  validated,** and no fit that writes under `data/studies/` runs without the coordinator's go-ahead,
  because worktrees share one data folder.
- Sonnet implements; Opus runs the code and mutation review, the two science reviews, and the
  persona reviews; Sonnet fixes. Briefs go in files under `.claude/worktrees/`; sub-agents do not
  dispatch sub-agents; scratch lives under `.claude/worktrees/scratch/nwp-compare/`.
- `data.dynamical.org` access ends on 2026-09-30. Every Dynamical.org input this study reads is
  already on disk or in the download coordinator's fetch, so the study does not depend on that host
  after the downloads finish.

## Risks and open questions

- **The Previous Runs lead rule is unverified.** If V1 finds Open-Meteo picks runs by availability
  rather than initialisation, the bracket still holds only if the lead stays between ENS day 0 and
  day 1; V1 prints the leads it finds. Recommendation: treat V1 as a gate.
- **Solar rows at exact-lead hours are few** (12 UTC all year, 06 and 18 UTC in summer), so the
  exact-lead Family 2 ranking may have wide intervals. That is why it is exploratory.
- **UKV's day-1 radiation from two snapshots** may straddle a run change at a cycle boundary.
  Recommendation: accept, and report the share of hours where it does.
- **The GEFS and GFS downloads reach back to 2020 and 2021, but ENS starts in April 2024**, so the
  longer history is unused except for an optional exploratory GEFS-against-GFS comparison on a long
  row set, which this plan does not schedule.
