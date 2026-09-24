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
forecast product at day-ahead (day 1), and at days 2 and 3, on the same hours at 6 solar farms (A
to F) and 3 wind farms (W1 to W3), from 1 December 2024 to 10 September 2026. Two families of
product are compared. The whole-run archives of ECMWF ENS and NOAA GEFS (Dynamical.org) give each
ensemble's 00 UTC run, read at 09:00 UTC, so every row's lead is identical across the two and both
runs were published before the forecast is issued. Open-Meteo's Previous Runs archive gives nine
more products (UKV, ICON-D2, ICON-EU, ICON global, ECMWF IFS 0.25°, GFS, ARPEGE, AROME, and two
HARMONIE-AROME models) at a fixed day offset, whose lead depends on each model's run cycle. Those
products are compared with ENS by **bracketing**: ENS's day-0 lead is shorter and its day-1 lead
longer than every Previous Runs day-1 lead on the same row, so a product that beats ENS at day 0,
or loses to ENS at day 1, is better or worse at a matched lead. A blend of the open-data forecasts
is scored against ENS alone at an optimistic and a conservative lead, with the blending study's
permutation control as a guard. The page ends with a "What to use" verdict per technology.

## Verdict, size, and departures from the issue

**Verdict: worth doing, roughly as described.** The issue's premise holds: nothing on `main`
compares forecast products at a forecast lead. The data is on disk for the Previous Runs products,
and GEFS is downloading.

**Size: complex.** Trigger answers:

- **What gets stored:** yes. A published docs page, its charts, and study outputs under
  `data/studies/nwp_forecast_comparison/`.
- **Production serving path:** no. Nothing in `src/` or the production packages changes.
- **Degradation rule:** no.
- **More than one defensible design:** yes. How to match leads across archives with different run
  cycles, which contrasts decide, how to treat ensembles, and which products go in a blend all have
  several defensible answers.
- **Callers not nameable without searching:** no. The study adds new scripts, one new function in
  `packages/studies/`, and reuses #885's fold helper; the callers are the study scripts named below.

**Reviews bought:** both plan reviews (simplicity, then correctness), both diff reviews
(correctness-and-cut, then the mutation pass, which runs because `packages/studies/` changes), two
Opus science reviews each followed by a fixer and any re-run a reviewer asks for, persona reviews
(the builders of each product compared, and the users), a `prose-review` sweep, and a pre-merge
check of the rendered HTML.

**Departures from the issue body:**

- **Previous Runs is not a fixed-lead instrument, so the plan does not treat it as one.** The issue
  calls the Previous Runs archive "the instrument this study wants". Each `previous_dayN` value comes
  from the freshest run initialised at least N × 24 hours before the hour, so for a model run every
  `n` hours its lead is `24N + (h mod n)`, where `h` is the UTC hour. Leads differ between products
  by up to 5 hours, and no Previous Runs value reproduces a fixed daily issue time. The plan matches
  leads exactly where the archive allows (ENS against GEFS) and brackets otherwise.
- **Blends are planned at day 1 only**, with day 2 and 3 blends exploratory. The issue asks for
  blends "at each lead"; three leads of blends with guards and a second setting would triple the
  planned contrasts.
- **Days 5 and 7 are left out.** At those days Previous Runs holds only IFS 0.25°, GFS, and ICON
  global, and ENS at long leads is already on the horizons page. "The days around" day-ahead are
  days 2 and 3.
- **The ENS horizons page is not rewritten here.** Issue #892 owns re-running that page with an era
  cut. This study starts its rows on 2024-12-01, after IFS Cycle 49r1, so its own ENS figures do not
  cross the step, and the report prints this study's ENS day-1 error beside the horizons page's as
  an exploratory reconciliation.
- **The `ens_horizons.py` rewrite the issue comment mentions is out of scope.** This study reads the
  ENS members through `ens_forecast_horizons.py`'s upsampling, which already uses
  `studies.cross_validation`, and never calls `ens_horizons.py`.
- **ICON global, ARPEGE, AROME, GFS, and the HARMONIE-AROME models are scored though the issue does
  not name them.** Open-Meteo serves them, so the issue's rule ("the models Open-Meteo serves")
  admits them, and each costs one fit on rows already built. They are exploratory.

## The question and the products

**Question: at the day-ahead lead the live service delivers, and the two days after it, which
forecast product, or which blend of products, gives the most accurate power forecast at these
farms?**

| Product | Archive on disk | Runs used | Lead at day 1 | Day offsets used | Offsets from |
|---|---|---|---|---|---|
| ECMWF ENS (mean of 51 members; control member exploratory at day 1) | Dynamical.org via `ens_forecast_horizons/ens_members.parquet` | 00 UTC | 24 + h (day 1); h (day 0) | 0–3 | 2024-04-01 |
| NOAA GEFS (mean of 31 members) | Dynamical.org, `weather/GEFS/` (downloading) | 00 UTC | 24 + h | 1–3 | 2020-10 |
| UKV | Open-Meteo Previous Runs | freshest initialised ≥ 24N h before | 24N + (h mod n) | 1 | 2024-08-06 |
| ICON-D2 | Previous Runs | as above | as above | 1 | 2024-01-19 |
| ICON-EU | Previous Runs | as above | as above | 1–3 | 2024-01-19 |
| ICON global | Previous Runs | as above | as above | 1–3 | 2024-01-19 |
| ECMWF IFS 0.25° (the deterministic IFS, ECMWF open data) | Previous Runs | as above | as above | 1–3 | 2024-03-06 |
| GFS (Open-Meteo "GFS seamless") | Previous Runs | as above | as above | 1–3 | 2024-01-19 |
| ARPEGE Europe | Previous Runs | as above | as above | 1–3, solar only | 2024-01-19 |
| AROME France | Previous Runs | as above | as above | 1, solar only | 2024-01-19 |
| KNMI HARMONIE-AROME Europe, DMI HARMONIE-AROME | Previous Runs | as above | as above | 1 | 2024-06 (KNMI wind from 2024-10) |
| ERA5, CAMS (solar) | past-weather studies | not forecasts | reference rows | — | — |

ARPEGE and AROME are solar-only because their 100 m wind offsets start in August 2026 (missing on
94.7% of wind rows). The implementation fills in each product's grid, domain, and publication time
from its producer's documentation; the table on the page carries them.

**Every run a row uses was published before the forecast it feeds is issued.** For ENS and GEFS the
issue time is 09:00 UTC on the run's day, the live service's `NWP_PUBLICATION_DELAY_HOURS`.
Verification V4 checks each 00 UTC run's publication time against its producer's dissemination
schedule (ECMWF disseminates by about 06:55 UTC, and Dynamical.org's ENS is readable from about
09:00 UTC) and the page cites them. A Previous Runs value comes from a run initialised at least 24N
hours before the hour; the page states each product's publication delay beside it, and the lead used
for the bracket is the lead from initialisation, which no publication delay changes.

**Not in this study:** ECMWF IFS HRES 9 km (its Previous Runs offsets start on 2025-10-01 and are
missing on about half the rows; IFS 0.25° is the same ECMWF run from the open data), GFS from
Dynamical.org (Open-Meteo's GFS covers every row; the Dynamical.org GFS extract serves only V1),
CERRA (a reanalysis; ERA5 is the past-weather reference), and Open-Meteo Single Runs. The Single
Runs download the coordinator forwarded would support a follow-up issue measuring what a 09:00 UTC
issue really gets from UKV and ICON-EU; no part of this study waits for it.

## Leads, and what "matched" means

**Lead is measured from the run's initialisation to the valid hour** (the end of the hour for an
hourly-mean radiation value).

- **ENS and GEFS, exact lead.** Day `d` forecasts calendar day `D + d` from the 00 UTC run of day
  `D`, as on the ENS horizons page. On a given row both have lead `24d + h`, and both runs were
  published before 09:00 UTC on day `D`. The contrast between them is exactly lead-matched and is
  exactly what a 09:00 UTC service could have run.
- **Previous Runs products, day offset N.** Lead `24N + (h mod n)`, which for any run cycle `n` of
  24 hours or less lies between `24N` and `24N + h`.
- **Bracketing a Previous Runs product against ENS.** On the row for hour `h` of a day, ENS day
  `N − 1` has lead `24(N − 1) + h`, which is below `24N`; the product's day-N value has lead `24N +
  (h mod n)`; ENS day `N` has lead `24N + h`, at or above the product's. At day 1: ENS day 0 has
  lead `h`, the product 24 to 29 hours, ENS day 1 `24 + h`. If the product's error is below ENS day
  `N − 1`'s, the product beats ENS even though ENS had the shorter lead on every row. If it is above
  ENS day `N`'s, ENS beats the product even though the product had the shorter or equal lead on every
  row. Anything between is **unresolved at matched lead**, and the page says so rather than choosing
  a side. The bracket needs no knowledge of `n`, only that the archive follows the rule above, which
  V1 checks.
- **The bracket assumes error does not fall as lead rises.** The report checks this on the study's
  own rows: ENS day 0 against day 1 (the horizons page found day 1 worse by 0.61 points for solar and
  0.75 for wind), and each multi-day Previous Runs product's day 1 against its day 2. If either
  check fails, the bracket for that product is void and the page says so.

**What a 09:00 UTC service would really get from a Previous Runs product is not measured.** At 09:00
UTC the freshest UKV or ICON-EU run is a morning run, so for a daylight hour the service's lead is
longer than the Previous Runs day-1 lead (for example 18 + h from a 06 UTC run, against 24). Day 1
of Previous Runs therefore flatters those products relative to what the service could deliver for
most daylight hours, and the page says so beside every Previous Runs result. Day 2 of Previous Runs
(lead 48 or more) is always staler than the service's lead, which the conservative blend below uses.

## Rows, folds, and fairness

- **Hours.** The past-solar and past-wind studies' hourly rows (every daylight hour at A to F; every
  hour at W1 to W3), with their commissioning-ramp, zero-hour, and outage exclusions, built through
  the ENS horizons page's `_base_frame`. Solar power is aggregated to the hour ending at the label,
  wind to the hour centred on it. Rows run from 2024-12-01 (the first whole month after IFS Cycle
  49r1, as the past-wind ENS and HRES study, #885, uses) to the rows' last date, 2026-09-10.
- **One shared row set per technology: the rows where the target and every planned arm's input at
  every scored day are present.** The planned arms are ENS, GEFS, UKV, ICON-EU, IFS 0.25°, and the
  baselines. UKV's day-1 offset is missing on 4.6% of solar rows and 5.2% of wind rows, mostly about
  30 whole days between 2026-04-06 and 2026-05-29, so the row set loses about half of the spring
  after UKV's January 2026 upgrade, and the page says so. Exploratory arms are fitted on the same
  rows, their missing hours left as missing values (XGBoost routes them natively); the report prints
  each exploratory arm's missing share (measured from 0.04% to 1.3%). The report prints the rows each
  planned product drops.
- **Folds: #885's design, confirmed by the study coordinator, and #885's code.** Five folds of whole
  months, cut inside three eras: before 2025-10-01; 2025-10-01 to the UKV PS47 upgrade on
  2026-01-21; and after it, with the part-month that straddles the upgrade dropped as #885 drops it.
  The third era's fold numbers are rotated (`ERA_FOLD_OFFSETS`) so no calendar month is held out of
  every era at once (#868); a two-era design fails that check on these rows too (February, April,
  May, and June). Every arm gets the same `era_code` column. Before any fit, the script prints the
  (site, fold, calendar month) coverage table and lists the calendar months that occur in one year
  only. **The 2025-10-01 cut is in the design because #885 needs it** (HRES's archive source changes
  that day) and the two studies share one fold helper; here it is harmless, and it happens to fall
  a week after DWD's ICON low-cloud change of 2025-09-24. The page says both.
- **This PR is blocked by #885 for the fold helper** (`with_three_eras`, `ERA_FOLD_OFFSETS`,
  `calendar_month_coverage`). The helper moves into `packages/studies/`, with tests, in whichever of
  the two PRs the coordinator chooses; this plan writes no second copy.
- **IFS Cycle 50r1 (2026-05-12)** is not an era cut. #885 prints the effect of an extra cut there on
  the same span; this study cites that result, and refits with the extra cut only if #885 finds that
  it moves a contrast.
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
  generator's H3 cell (the file on disk); GEFS is read at the nearest 0.25° cell; Open-Meteo products
  at the nearest land cell Open-Meteo picks.
- **Temporal steps.** ENS and GEFS are 3-hourly, upsampled per member through the clear-sky index for
  radiation (the horizons page's chosen method) and by eastward and northward components for wind
  speed, before any member mean. GEFS radiation alternates 3- and 6-hour averaging windows, so it is
  converted to 3-hour step means first (a new tested function, and verification V2). Open-Meteo
  interpolates IFS 0.25°'s 3-hourly steps itself.
- **UKV's served radiation is a snapshot**, so UKV's hourly value is rebuilt as the mean of the
  snapshots at the hour's start and end (`studies.hourly_means.hourly_from_snapshots`), as on the
  past-solar page. Both snapshots come from the day-1 column; the report prints the share of hours
  whose two snapshots straddle a run change.
- **Ensembles.** ENS and GEFS arms read the mean of the members' upsampled fields, the treatment the
  horizons page found best at day 1. An ENS control-member arm at day 1 (exploratory, on data already
  on disk) shows how much of ENS's standing against the single-run products is ensemble averaging.
  Per-member fitting is not repeated (the horizons page settled it for ENS).
- **XGBoost model.** `studies.cross_validation.out_of_fold_losses`: per generator, absolute-error
  objective, three fitting seeds, curtailed hours left out of training and kept for scoring, the
  prediction held to the export cap. Each error is divided by its own generator's capacity (99th
  percentile of output, from the `effective_capacity` table; the page states which build) before any
  mean or difference.
- **Intervals.** `studies.bootstrap.bootstrap_difference` and `bootstrap_absolute`: 2,000 resamples of
  whole calendar months, each drawing one fitting seed.
- **Absolute skill beside every contrast.** A day-1 leaderboard per technology (every arm's own error
  with its interval) and an error-by-day table for every product at days 1 to 3. No-weather
  baselines at each day from `studies.baselines` (climatology and smart persistence, issued at 09:00
  UTC) set the floor; ERA5, and CAMS for solar, are past-weather reference rows that no forecast
  could have had.

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
| P3 | GEFS mean day 1 − ENS mean day 1 | exact | Is the free NOAA ensemble as good as ENS at a 09:00 UTC issue? |
| P4a | Blend (ENS mean day 1 + ICON-EU day 1 + IFS 0.25° day 1) − ENS mean day 1 | optimistic: the other products' leads are Previous Runs day 1 | Does adding the other open-data forecasts to ENS lower the day-ahead error? |
| P4b | Blend (ENS mean day 1 + ICON-EU day 2 + IFS 0.25° day 2) − ENS mean day 1 | conservative: every other product's lead is at least what a 09:00 UTC service would have | as P4a, as a lower bound on the gain |

**Why these products.** UKV is the Met Office's convection-permitting UK model, free on AWS. ICON-EU
was the best partner for UKV on the blending page and is free from DWD. Both cover all of NGED's
licence area. ENS is the incumbent. GEFS is the other free whole-run ensemble, and the only product
this study can compare with ENS at exactly the lead a 09:00 UTC service would use. The blend's
products are the open-data forecasts with a day-2 offset in Previous Runs, so the same three
products appear at both bounds and only the lead changes between P4a and P4b. UKV has no day-2
offset, so a blend with UKV (ENS + UKV + ICON-EU + IFS 0.25° at day 1) is exploratory.

**The blend's guard.** Each blend has a control that keeps ENS's real columns and replaces the other
products' columns with `studies.blending.climatology_permutation` (shuffled among hours sharing a
generator, calendar month, and hour of day), so the control has the blend's column count. The blend
minus its control (the other products' weather) is reported beside P4a and P4b, and must also be
negative for a gain to be attributed to the other products' weather rather than to the extra
columns.

**Verdict rules, fixed now:**

- A product **beats ENS at matched lead** only if its lower-side contrast is negative and
  statistically significant at the 5% level.
- A product **loses to ENS at matched lead** only if its upper-side contrast is positive and
  significant.
- Otherwise the product is **unresolved at matched lead**.
- A blend **lowers the day-ahead error** if P4b is negative and significant and its guard is negative
  and significant (the gain survives the conservative lead and comes from the weather).
- A blend **may lower the error** if only P4a and its guard are negative and significant.
- Otherwise the blend **makes no detectable difference**, and the interval's bound on the effect is
  stated ("a gain as large as x points is not excluded").

Where the second setting changes a contrast's sign or significance, the page says so.

**Exploratory, labelled as such on the page:** the full day-1 leaderboard of every product; every
product at days 2 and 3 with its bracket against ENS; the ENS control member at day 1; ICON-D2, ICON
global, IFS 0.25°, GFS, ARPEGE, AROME, and both HARMONIE-AROME models against ENS; the blend with
UKV; blends at days 2 and 3; each blend's control minus ENS alone; the ENS day-1 reconciliation with
the horizons page; the per-generator errors. About 1 in 20 exploratory intervals reaches
significance at the 5% level by chance, and the page says so.

## What no contrast here can separate

The page states these beside the results they touch:

- **Model against resolution and grid.** A product's win may come from its grid spacing, its grid
  cell, or the archive's cell choice rather than its physics.
- **Ensemble averaging against model quality.** A mean of members is smoother than any one run, and
  mean absolute error rewards smoothness; the ENS control-member arm shows how much of ENS's standing
  is averaging.
- **Archive against producer.** Open-Meteo re-serves each producer's field with its own
  interpolation, cell choice, and 100 m scaling; a gap may belong to the archive.
- **Step width.** ENS and GEFS reach this study 3-hourly and are upsampled; most Previous Runs
  products are hourly.
- **Lead inside a bracket.** An unresolved bracket means the product sits between ENS's day-0 and
  day-1 errors; the study cannot place it more precisely.
- **What a 09:00 UTC service gets from a Previous Runs product**, as set out above.
- **Model upgrades inside the row set.** IFS 50r1 on 2026-05-12, DWD's ICON changes of 2025-07-23 and
  2026-09-02, ICON-D2's of 2026-02-18, and GEFS's radiation precision fix of 2026-06-15 are not era
  cuts.

## What changes, file by file

- `packages/studies/src/studies/cross_validation.py`: the era-fold helper and
  `calendar_month_coverage` from #885, moved here with tests by whichever PR the coordinator
  chooses (see "Rows, folds, and fairness").
- `packages/studies/src/studies/resample.py`: add `gefs_step_means`, converting GEFS's alternating
  3- and 6-hour window averages to 3-hour step means (the 6-hour step's second half is twice the
  6-hour window minus the preceding 3-hour window). Tests.
- `studies/nwp_forecast_comparison/` (new directory, own README):
    - `verify_previous_runs_leads.py`: V1 below.
    - `build_forecast_inputs.py`: per technology, one long parquet of every arm's weather columns at
      days 0 to 3, with each row's lead where it is known, read from the Previous Runs files, the ENS
      members, and the GEFS month caches. The ENS and GEFS upsampling reuses
      `ens_forecast_horizons.py`'s functions, generalised to take the member count and the source
      frame as arguments instead of `ENSEMBLE_SIZE` and `OUTPUT_PATH`; if that generalisation needs
      more than argument changes, the shared pieces move into `packages/studies/` with tests.
      Anonymised labels only.
    - `nwp_forecast_comparison.py`: rows, folds, fits, baselines, intervals, the bracket verdicts,
      `report.md`, saved per-row losses and predictions, fingerprint (floats cast to Float32 before
      hashing), `--report-only` and `--fit-missing`.
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

- **V1, Previous Runs run-selection rule (a gate).** Compare Open-Meteo GFS's `previous_dayN`
  `temperature_2m` (an instantaneous field) with the Dynamical.org GFS runs in
  `weather/GFS_window_2025-07-01_2025-07-02/`, and report, per UTC hour, which run and lead matches
  best. The plan's rule predicts the run initialised at `floor6(h − 24N)`. If the archive instead
  picks runs by publication time, the day-1 lead could exceed `24 + h` at small `h` and break the
  bracket's upper side; the plan is then revised before any fit.
- **V2, GEFS radiation.** After converting windows to step means: no negative means beyond a stated
  tolerance, no night-time radiation, and the member mean's correlation with CAMS highest at the
  right window (as the ENS README's check did).
- **V3, timestamp and height conventions** per Previous Runs product (the lineage notes: UKV
  snapshot, the rest hour-ending), and a power-hour offset scan for wind.
- **V4, publication times.** Each ensemble's 00 UTC run publication time from its producer, cited.
- **V5, domains.** Which products cover NGED's whole licence area (ICON-D2 stops about 2.3°W), from
  `weather_product_domains.py` or each producer's stated domain.
- The data-validation checklist on every input frame the build writes.

## Charts

Every chart per the `dataviz` and study skills: OCF palette, SVG through `svgo`, `aria=False` on data
marks, generators labelled A to F and W1 to W3 only, values as % of capacity, time axes showing
days 1 to 7 with the month and year in text only.

1. Day-1 leaderboard per technology, each arm's error with its interval, smaller is better, family
   and lead band marked.
2. Planned contrasts P1 to P4, with the two sides of each bracket drawn together.
3. The XGBoost models work: day-1 out-of-fold forecasts against measured output, one week per
   generator chosen by a stated rule, and per-generator errors per arm.
4. Error against day (0 to 3) for every product, with ENS's day-0 and day-1 errors drawn so the
   brackets are visible.
5. Each blend against ENS alone and against its control, at both bounds.

## Tests

- `gefs_step_means`: a synthetic series of known 3-hour step means turned into GEFS-style
  alternating 3- and 6-hour window averages and back, exactly; and a negative-producing input
  handled as the function documents. Fails today because the function does not exist.
- The fold helper's tests move with it from #885.

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

- **No fit runs until the study coordinator confirms the GEFS download is complete and validated,**
  and no fit that writes under `data/studies/` runs without the coordinator's go-ahead, because
  worktrees share one data folder. The study reads GEFS only from 2024-11-30.
- Sonnet implements; Opus runs the code and mutation review, the two science reviews, and the
  persona reviews; Sonnet fixes. Briefs go in files under `.claude/worktrees/`; sub-agents do not
  dispatch sub-agents; scratch lives under `.claude/worktrees/scratch/nwp-compare/`.
- `data.dynamical.org` access ends on 2026-09-30. Every Dynamical.org input this study reads is
  already on disk or in the download coordinator's fetch.

## Risks and open questions

- **The Previous Runs rule is unverified.** V1 is a gate.
- **UKV's day-1 radiation from two snapshots** may straddle a run change. Recommendation: accept,
  and report the share of hours where it does.
- **The GEFS download reaches back to 2020, but ENS starts in April 2024**, so the longer history is
  unused.

## Review log

### Plan review 1 (simplicity), Opus

Accepted:

- The 2% rule would have moved UKV, a planned arm, off the shared rows (UKV day 1 is missing on 4.6%
  of solar and 5.2% of wind rows). The shared rows are now the planned arms' intersection;
  exploratory arms keep their small gaps as missing values.
- The blend verdict is now read from blend minus ENS alone, the difference the live service sees;
  blend minus control is a guard that must also be negative.
- Cut: the Dynamical.org GFS arm and its four-cell sampling check (Open-Meteo GFS covers every row),
  GFS's window conversion, HRES 9 km, CERRA, the synthetic positive control, the Single Runs arms,
  the GEFS control member, the 50r1 refit (cite #885), days 5 and 7, the exact-lead-hours subset,
  the per-product cycle measurement, and the `previous_runs.py` module (the bracket needs no cycle
  length; the verdict function is a few lines in the study script).
- V1 now uses `temperature_2m` on the existing Dynamical.org GFS extract, so it does not wait for
  the full download.
- The fold helper comes from #885 rather than a second copy; the page says the 2025-10-01 cut is
  inherited from #885.
- The fit gate is GEFS only; CERRA and GFS are no longer inputs.

Rejected:

- **Demote P3 (GEFS against ENS) or cut GEFS.** Kept as planned: GEFS is the other free whole-run
  ensemble, the download exists for this issue, and P3 is the only contrast this study can run at
  exactly the lead a 09:00 UTC service uses, so it answers an ingest question (could the service
  read a free ensemble instead of ENS) without any bracket.
- **Put the GEFS window conversion in a one-line helper beside the extraction instead of a tested
  package function.** A wrong window conversion produces plausible radiation rather than an error,
  which is the study skill's criterion for tested machinery in `packages/studies/`.
- **Cut the ENS control-member arm.** Kept at day 1 only: it costs one fit on data on disk, and it
  is the one arm that shows how much of ENS's standing against the single-run products is ensemble
  averaging on this study's rows.
