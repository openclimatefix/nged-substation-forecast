# Plan: ECMWF ENS and ECMWF IFS HRES wind in the past-wind study (phase 1)

**Problem.** [Which weather product best describes past wind?](../docs/studies/weather-products-for-
past-wind.md)
scores ERA5, UKV, ICON-D2, ICON-EU and ICON global at three wind farms (W1 to W3). The page scores
no ECMWF IFS HRES (checked: no HRES arm in `wind_products.py`, no HRES text on the page) and no
ECMWF ENS. The past-solar page has both. The ENS horizons page compares ENS with ERA5 for wind at
day 0 and day 1, but not with UKV, ICON, or HRES.

**Solution.** One new study script that reads the page's own row set, adds an HRES arm and an ENS
ensemble-mean arm, and refits every product on the rows all of them cover. It scores three
planned contrasts fixed in this file before any fit. It reuses the horizons study's ENS wind
inputs and the `wind_icon_dream.py` structure, so no new ENS code is written. One results section,
one "What to use" bullet and one chart script follow.

## Verdict, size and the five trigger answers

Worth doing, with the design changes below (adopted from the first plan review). **Size: complex**,
so both plan reviews and both diff reviews run, then the reviews the maintainer asked for.

1. **What gets stored:** yes. A published page section and new outputs under
   `data/studies/beam_diffuse_split/past_weather_v2/ens_hres_past_wind/`. No Patito contract, no
   Delta table.
2. **Production serving path:** no. Study scripts only.
3. **A degradation rule:** no.
4. **More than one defensible design:** yes. Which ENS band, how the 3-hourly steps become hourly
   wind, and which HRES source. The choices are fixed below and the alternatives run as
   exploratory arms.
5. **Callers not nameable without searching:** no. New scripts only. `packages/studies/` changes
   only if a helper proves reusable; the mutation pass then runs on it.

## Departures from the request, for the coordinator to veto

- **HRES source.** The request named the 9 km grid file, which holds wind speed only, in km/h. The
  Previous Runs file `data/studies/weather/ECMWF-IFS-HRES/previous_runs/combined.parquet` holds each
  farm's HRES wind speed and direction at 10 m and 100 m, hourly, from the same `ecmwf_ifs` model
  (bare columns are the freshest run). Using it keeps every arm at the page's seven columns and
  needs no grid-coordinate recovery. The script checks that its speed agrees with the grid file at
  the nearest point, and the page cites both files. The grid file is otherwise unused.
- **ENS band.** The request listed the `T+3` wind file. That file covers leads 3 to 21, so valid
  hours 03 to 21 UTC only, and its download script is not in the repository. The ENS horizons study
  already holds the ensemble-mean wind for the same download at day 0 (leads 0 to 23, every hour
  of the day), rebuilt to hourly by the treatment its own pre-registered rule chose, with the
  tested code and saved inputs (`data/studies/ens_forecast_horizons/wind_inputs.parquet`). The
  plan reads day 0 from there. The `T+3` wind file goes unused.
- **The ENS-against-ERA5 contrast re-estimates a published number.** The horizons page already
  reports the day-0 ensemble mean against ERA5 for wind, on 50,268 rows from 12 August 2024. The
  new estimate is on a shorter row set and is labelled exploratory, with the horizons figure
  printed beside it.
- **Three planned contrasts, not four** (coordinator's advice on multiplicity): the two that answer
  the page's question about historical features (does either ECMWF product beat UKV) and one for
  training history (does HRES beat ERA5). The intervals are not adjusted; the page says the three
  share their months, and an exploratory row gives Bonferroni-adjusted intervals.

## Planned contrasts (written before any fit)

Every planned contrast uses: ENS band **day 0** (valid hours 00 to 23 UTC of the run's own day,
leads 0 to 23 hours from the 00 UTC run, Dynamical.org's archive, 0.25 degrees, all 51 members);
members combined as the horizons study does (`reduce_members`: the mean of the members' speeds, the
direction of the mean wind vector) after each member's 3-hourly steps are rebuilt to hourly by the
wind treatment below; HRES as Open-Meteo's freshest-run `ecmwf_ifs` hourly wind; every arm given
the page's seven columns (`SHARED_FEATURES` plus `_wind_columns`), at hub height 100 m for HRES,
ENS,
ERA5 and UKV.

| # | Contrast (first minus second) | Question it answers |
|---|---|---|
| P1 | `hres_wind` − `ukv_wind` | Does ECMWF's deterministic forecast describe past wind better than UKV, the page's product for historical features? |
| P2 | `ens_mean_day0_wind` − `ukv_wind` | Does ENS's ensemble mean describe past wind better than UKV? |
| P3 | `hres_wind` − `era5_wind` | Does ECMWF's deterministic forecast describe past wind better than the reanalysis a training history could use? |

Exploratory, reported and labelled so: `ens_mean_day0_wind` − `hres_wind`; `ens_mean_day0_wind` −
`era5_wind` (a re-estimate of the horizons page's contrast, with its figure printed beside it);
the ICON contrasts; Bonferroni-adjusted intervals for P1 to P3. Every contrast is rerun at the
second hyperparameter setting (`SENSITIVITY_HYPER_PARAMETERS`).

**What no planned contrast can separate.** Each contrast mixes: served lead (ENS day 0 spans leads
0 to 23 from a 00 UTC run; HRES's freshest-run wind leads are 1 to 12 hours before 1 October 2025
and 0 to 5 from it, measured from where the hour-to-hour jumps in the served series fall, with
the jump table printed in the report; ERA5 is an analysis; UKV is T+0); step width (ENS is 3-hourly,
rebuilt to hourly); native and served resolution (both about 9 km native; ENS served at 0.25
degrees, HRES from 1 October 2025 on the native grid, before then not established); IFS cycle (49r1
until 12 May 2026, then 50r1); the source of HRES's archive (changed on 1 October 2025); height;
how each value is read (ENS is the area-weighted mean of the 0.25-degree cells that each farm's H3
resolution-5 cell overlaps, its speed the magnitude of the cell-mean wind vector; HRES is the
single nearest land cell Open-Meteo picks); and, for the exploratory ENS-against-HRES contrast,
ensemble averaging against a single run. The page says so beside each result.

## The wind-specific treatment (fixed before the fit)

**Wind is instantaneous at its label, so the solar clear-sky-index reconstruction does not apply.**
The planned ENS input is the horizons study's `components` combination: each member's speed and
direction become eastward and northward components at each 3-hourly step, the components are
interpolated linearly to hourly, and the hourly speed is the magnitude of the interpolated vector
and the direction comes from the same vector, entering as sine and cosine. **Direction is never
averaged or interpolated as a plain number.** The horizons study's own chosen combination,
`speed_components`, interpolates direction linearly in degrees, which crosses north the wrong way
on about 1% of day-0 hours; the plan departs from it for direction only, and the speed treatment
is the one that study's pre-registered rule chose (the magnitude of the interpolated vector). The
alternatives run as exploratory arms on the same rows and columns, each reading the same saved
horizons inputs: `speed_components` (the horizons study's choice), `direction_components` (speed
interpolated linearly), and `linear`. Their spread also shows how large a difference the pipeline
produces from a change in the interpolation alone; the plan has no other negative or positive
control.

**Hub-height mismatch.** The three farms' hub heights are unknown. As the page's existing arms do,
each arm takes its product's native hub-height speed and its 10 m speed, and the per-farm XGBoost
model absorbs the mismatch. Phase 1 adds no shear extrapolation.

## Rows, folds, fairness

- **Row set.** The page's rows (`wind_products.common_rows(joined(...))`) joined to ENS day 0 and
  HRES, from 1 December 2024, the first whole month after IFS Cycle 49r1 (12 November 2024), to the
  page's end date (2026-09-10). The date filter runs before eras and folds are assigned, as
  `wind_icon_dream.icon_dream_common_rows` does. The row set holds 43,555 rows (W1 14,489, W2
  14,994, W3 14,072). The row set starts later than the page's (12 August 2024) because HRES's
  served
  grid before 2025 is not established, as on the solar page. Every product is refit on this row
  set.
- **Arms.** Every product's wind arm at the primary setting (ERA5, UKV, ICON-D2, ICON-EU, ICON
  global, HRES, ENS); the second setting for the four products in the planned contrasts (HRES, ENS,
  UKV, ERA5). Speeds are converted to m/s explicitly (HRES's Open-Meteo columns are km/h, ENS is
  m/s).
- **Fairness.** Every arm has exactly the same rows and seven columns; `colsample_bytree=1`;
  per-arm column lists in the report; folds cut inside each of three eras: before 1 October 2025
  (HRES's archive source changes), 1 October 2025 to 20 January 2026, and from 1 February 2026 (the
  UKV upgrade; the rest of January
  2026 dropped as on the page), with `era_code` taking three values for every arm, so IFS Cycle
  50r1 on 12 May 2026 is left to the period split below; month-resampled
  paired bootstrap of whole months and seeds.
- **Calendar-month coverage check (issue #868).** Cutting folds inside three eras ranks each era's
  months separately and reuses fold numbers 0 to 4, so one calendar month can be held out of every
  era at once and leave no training data for that season. Before any fit, the script asserts that
  for every scored calendar month, in every fold, some training rows carry that calendar month
  from some era, and prints the table into `report.md`. If the check fails, the script offsets the
  fold numbers between eras (or assigns folds by calendar month), and this file is updated with
  the choice before the first fit. The page states the fold design as a limitation either way.
- **Row fingerprint.** Deterministic, floats cast to Float32 before hashing, saved beside the
  losses, and checked on `--report-only`.
- **Servable-hours split.** Wind power for label T covers T minus 30 minutes to T plus 30 minutes,
  so labels 00 to 08 UTC (hours ending before about 09:00 UTC, when Dynamical.org's archive has the
  00 UTC run) are compared with labels 10 to 23, dropping label 09. The split also separates ENS
  leads 0 to 8 from 10 to 23 and morning from afternoon, so the page does not present it as a clean
  test of servability. HRES's archive delay is not established, so HRES gets no split.
- **Period splits (exploratory):** before and after 1 October 2025 (HRES's archive source
  changed), and before and after 12 May 2026 (IFS Cycle 50r1), from the saved losses.
- **Printed-number guard.** A committed check takes the new section by its heading and every
  decimal number and every `[a, b]` pair in it, and requires each to equal a `report.md` value
  rounded to the page's precision, with the sign mapped (the report prints `+0.142 [+0.031,
  +0.254]`; the page writes `0.14 points [0.03, 0.25]`). Heights, the 51 members, 0.25 degrees and
  dates are excluded. It is shown able to fail by perturbing one number once.
- **Three farms.** Every pooled interval on the page states that three wind farms are few
  independent sites.

## Product facts to verify before writing

A fact report (`scratch/facts-report.md`) covers the ENS and HRES grids, runs, step widths,
dissemination and archive timing, and IFS cycle dates. Its quotes were summarised by a small
model, so an Opus science reviewer re-checks every fact against ECMWF's, Dynamical.org's and
Open-Meteo's pages. Limits belonging to ECMWF's open-data subset or to either archive are
attributed to them, not to ENS.

## Files

- `studies/beam_diffuse_split/ens_hres_past_wind.py` (new): build rows, jobs, fingerprint, report,
  copying `wind_icon_dream.py`'s structure; imports the Float32 fingerprint and printed-number
  check from `ens_past_solar.py` and `ens_past_solar_charts.py` where importable.
- `studies/beam_diffuse_split/ens_hres_past_wind_charts.py` (new): leaderboard, planned contrasts,
  and a days-1-to-7 out-of-fold check for the two new arms (% of capacity, `aria=False`).
- `docs/studies/weather-products-for-past-wind.md`: one results section, a Key-findings bullet, a
  "What to use" bullet, two product-table rows, "Data and methods" and "Limitations" additions,
  and the reproducing commands.
- `docs/studies/assets/`: the new SVGs, optimised with svgo.
- `packages/studies/`: unchanged unless a helper proves reusable.

## Tests and verification

Study scripts are not unit-tested; the report is their check. Any helper moved into
`packages/studies/` gets tests and the mutation pass. Verification: `ruff check`, `ruff format`,
`ty check`, `pytest`, `pymarkdown scan`, `mkdocs build --strict`, `check_docs_links.py`,
`pydoclint`, and reading the built HTML under `site/`.

## Risks and open questions

- **HRES's archive source changed on 1 October 2025**, so HRES's lead and grid differ across the
  row set. Folds are cut on that date; the page names it as a confound.
- **The HRES cross-check** compares the Previous Runs file's bare speeds with the 9 km grid file's
  nearest few points, because Open-Meteo picks its own model cell, which is not always the nearest
  0.05-degree point. The check requires that some point among the nearest five matches exactly on
  every hour, and logs only the rank, never a coordinate.
- **The period after 12 May 2026** holds about 4 months, so intervals there under-cover; the
  page says so.
- **Day 0 is a best case.** At this repository's assumed 09:00 UTC read time, only ENS day 0's
  hours 00 to 08 UTC have passed; hours 09 to 23 are still a forecast up to 14 hours ahead. The ENS
  day-0 scores are therefore past weather delivered late for the early hours, and the best a 00 UTC
  archive offers for the rest.
- **The IFS cycle changes inside the row set** on 12 May 2026.

## Reviews this plan buys

Both plan reviews, then both diff reviews, then two science reviews, the persona reviews, a prose
sweep and a pre-merge check of the rendered HTML.

## Findings rejected

- Reading ERA5, UKV and the ICON arms from the page's published losses: rejected because the row
  set starts later, so their losses are refit.

## Decisions made during implementation

Written before the first fit. Each was made where this plan is silent or could not be followed as
written.

- **The coverage check cannot pass for October and November.** The row set runs from 1 December 2024
  to 10 September 2026, so October and November occur in one year only (2025). Whichever fold holds
  October or November 2025 out leaves no training row for that calendar month, under any fold
  design. The script prints every (site, fold, calendar month) cell, lists the single-year months,
  and raises only for a calendar month that occurs in two years. The page states the October and
  November limitation.
- **The coverage check failed with the era folds unrotated, and the fold numbers are now offset.**
  With no offset, six (site, fold, calendar month) cells failed, across the three farms: July (fold
  3 in both 2025 and 2026) and September (fold 4 in both). The third era's fold numbers are
  rotated by 2 (`ERA_FOLD_OFFSETS`), which leaves the folds contiguous within an era and the check
  with 0 failing cells.
- **The HRES served-lead evidence is read from the data, not assumed.** The script prints the
  hour-to-hour change by UTC hour and lists the hours that reach a ratio of 1.15, and states how
  many of the plan's expected handover hours (01 and 13 before 1 October 2025; 00, 06, 12 and 18
  from it) reach it. The lead statement on the page is limited to what those hours support.
- **The Bonferroni intervals use the script's own resampler**, because `studies.bootstrap` fixes its
  percentiles at 2.5 and 97.5. The script asserts that its resampler reproduces
  `bootstrap_difference`'s 95% interval on the first planned contrast before it prints an adjusted
  one, so only the level differs.
- **Speeds.** HRES speeds are converted to m/s where the frame is built. ERA5, UKV and the ICON
  products stay in the km/h the page fitted them in, since XGBoost's splits are invariant to a
  rescaling of one column. Mean speeds are printed in m/s for every product.
- **The report adds a Months column to every contrast table**, so the period splits state how many
  calendar months each interval rests on.

## Post-review additions (exploratory, added after the first results)

Written after the first fit and the first science review, before any of the fits below. Every item
is post hoc: the review saw the first results, so none of these is a planned contrast, and the page
labels each "exploratory (added after the first results)". The first-fit outputs (`losses.parquet`
and its fingerprint) are not refitted or overwritten.

1. **Long-row-set reconciliation.** ERA5, UKV, HRES and ENS (the `components` and
   `speed_components` combinations) are refitted at the primary setting on the longer row set from
   2024-08-12, the rows common to all products and the horizons study's start. Two designs: (a) the
   horizons study's design, two UKV eras and no cut at IFS 49r1, which should reproduce that page's
   ENS-minus-ERA5 figure; (b) the same rows with one extra era cut at 2024-12-01. Each design
   prints ENS-ERA5, HRES-ERA5, HRES-UKV, ENS-UKV and ENS-HRES on all rows and on the rows from
   2024-12-01 only, plus the monthly ratio of each product's 10 m speed to ERA5's, which shows any
   step in November 2024. The losses go to `losses_long_rows.parquet` with its own fingerprint.
2. **Fold-design robustness table** for P1 to P3 and ENS-HRES: the study design; three eras with
   no fold rotation; two UKV eras (the page's `with_eras`); the study's folds with a two-valued
   `era_code`; and an extra era cut at IFS 50r1 (12 May 2026) that drops the part-month of May 2026.
   The losses go to `losses_fold_designs.parquet` with its own fingerprint.
3. **HRES served-lead table.** The statistic is each UTC hour's mean absolute hour-to-hour change
   over the mean of its two neighbouring hours' changes, for `wind_speed_100m`, `wind_speed_10m` and
   `temperature_2m`, before and from 2025-10-01. It replaces the median-normalised ratio, which the
   wind's daily cycle swamps. The lead statement is "1 to 12 h before 1 October 2025 and 0 to 5 h
   from it, inferred from where the hour-to-hour jumps fall". Open-Meteo does not document the
   lead: its documentation says only that each run's first few hours are stitched into a continuous
   series.
4. **Rename the servable split.** It is a lead and time-of-day split (labels 00-08 UTC against
   10-23 UTC). No text calls it a test of servability. ENS's deficit against UKV and HRES sits in
   the later hours, which are also the longer leads.
5. **Wording fixes.** ENS runs four times a day; the 3-hourly steps after ECMWF's hourly steps to
   T+90, the single 00 UTC run and the roughly 09:00 UTC read time belong to ECMWF's open-data
   subset and Dynamical.org's archive, and 09:00 UTC is this repository's
   `NWP_PUBLICATION_DELAY_HOURS` assumption, not a documented Dynamical.org latency (ECMWF
   disseminates ENS day 0 at about 06:40 UTC). IFS 50r1 went live with the 06 UTC run of 12 May
   2026, so the 00 UTC ENS run of that day is still 49r1 and the period split's "from 2026-05-12"
   holds one day of 49r1 ENS data. The HRES fetch set no `cell_selection`, so Open-Meteo's default
   ("land") applies, as it does for every other product on the page. The grid-file cross-check
   shows that two Open-Meteo downloads agree, not that the grid or the lead is right.
6. **Sign-safe printed-number guard.** A page pair `[a, b]` must match a report pair including
   sign. A bare magnitude may match either sign, and the guard lists every such magnitude so a
   reviewer can audit it. The page writes a difference as "X points lower" or "X points higher"
   and, in tables, as a signed value.

## Post-review additions 2 (added after the second science and the code review)

Written after the second science review and the code review, before any of the refits below. Every
item is post hoc. The first-fit outputs (`losses.parquet` and its fingerprint) are neither refitted
nor overwritten. Two corrections to this file come first.

**Corrections to this file.**

- **Provenance.** The last revision of this file before any fit is `83dbbad3`. That commit added the
  implementation decisions above, and `c7f32f95` only re-wrapped them. The first-fit `losses.parquet`
  was written after both. The page and the script docstring cited `715681a7`, which is an earlier
  revision, and now cite `83dbbad3`.
- **The servable-hours split bullet** above holds its wording from before the first fit. Its
  rename to a lead and time-of-day split is recorded in post-review item 4, not in that bullet.
- **The Risks sentence "most day-0 hours end before the 00 UTC run can be read" was false.** At the
  09:00 UTC read time only hours 00 to 08 UTC have passed (9 of 24), and at ECMWF's 06:40 UTC
  dissemination 7 of 24 have. The bullet now says so.
- **"ERA5 is read at the nearest cell" (post-review item 5) was false.** `fetch_wind_point.py` sets
  `cell_selection="land"` for every product it fetches, ERA5 included. The Previous Runs fetch
  (`fetch_open_meteo_previous_runs.py`) sets no `cell_selection`, so HRES is read at Open-Meteo's
  default land cell, as every other product on the page is. The report, the script docstring and the
  page now say that.
- **The lead threshold changed after the results.** The implementation decisions above fix the
  hour-to-hour jump threshold at a ratio of 1.15. The script's `JUMP_RATIO_THRESHOLD` became 1.10 in
  `0b2230e4`, after the first results and without a record here, because at 1.15 the 18 UTC hour
  (ratio 1.14 from 1 October 2025) does not count. The page states that 1.10 was chosen after the
  results. The lead statement in the report is the statement written in item 3 above, printed
  whatever the table shows, and the report now labels it that way.
- **Item 4's clause "which are also the longer leads" is withdrawn**, because the second science
  review found the later-hours widening is mostly a time-of-day difference in UKV (addition 3 below).

**Additions.**

1. **Coverage counts for every design (issue #868).** The report prints the number of (site, fold,
   calendar month) cells with no training row for a month that occurs in two years, for every
   design in the fold-design table and the long-row table, not only for the main design. Counts
   found before the refit, with the designs as they stood: main rows, study design 0; three eras
   without rotation 6 (months 7, 9; intended, that design is the no-rotation control); two UKV eras
   (the page's design) 12 (months 2, 4, 5, 6); study folds with a two-valued `era_code` 0; extra
   era cut at IFS 50r1 0. Long rows: horizons design 6 (months 6, 7); extra era cut at 2024-12-01
   15 (months 2, 4, 5, 6, 11).
2. **Offsets for the long rows' extra-cut design.** The design has three eras (2024-08 to 2024-11,
   2024-12 to 2026-01, 2026-02 onward). A search over every pair of offsets for eras 1 and 2 (era 0
   fixed at 0, modulo 5 folds) finds 6 of 25 pairs with 0 uncovered cells: (1, 3), (1, 4), (2, 0),
   (2, 4), (3, 0) and (3, 1). The map `{0: 0, 1: 2, 2: 0}` is used, because it has one non-zero
   offset, the fewest, and of the two such pairs the smaller. The horizons design and the page's own
   two-UKV-era design stay as they are, because the first has to keep the horizons study's fold
   layout to reproduce its +0.170 and the second has to be the page's design. Their uncovered counts
   are printed and the page states them. The extra era cut at IFS 50r1 keeps its offset map
   `{0: 0, 1: 0, 2: 2, 3: 4}`, which gives 0 uncovered cells (a search over era 3's offset finds 0
   and 4 both do), so its folds and numbers do not move.

   **A third long-row design isolates the coverage defect from the IFS 49r1 cut.** The horizons
   design (two UKV eras, no cut at 49r1) leaves 6 cells uncovered, and the extra-cut design leaves
   0, so a difference between them mixes the 49r1 cut with that defect. The third design uses the
   horizons design's two eras with era 1's fold numbers rotated by 2, which leaves 0 uncovered
   cells (rotations of 2, 3 and 4 all do; 2 is the smallest). Comparing it with the horizons
   design isolates the coverage defect, and comparing it with the extra-cut design isolates the
   49r1 cut. Both refits (`losses_long_rows.*` and `losses_fold_designs.*`) are moved to
   `superseded/` and refitted.
3. **More rows in the split and period tables.** The lead and time-of-day split adds `ukv_wind −
   era5_wind`, `hres_wind − era5_wind` and `hres_wind − ukv_wind`, because ERA5 is an analysis with
   no lead, so its gap to UKV between the halves shows how much of the change is time of day. The
   period splits add `ukv_wind − era5_wind`, because UKV's own January 2026 upgrade falls in the
   later period. Figure 19 gets the same control rows.
4. **The report prints four decimals**, and the printed-number guard reads full precision from
   `intervals.parquet` wherever a report row has a value there, so a three-decimal print can no
   longer be rounded a second time by the guard. The guard matches whole triples `X [a, b]`
   against report rows, enforces a "+" sign as it does a "-", and runs on the results section, the
   Key-findings bullets and "How ECMWF's ENS and HRES were added". The "dates are excluded" clause of
   the guard bullet above has no code behind it and is not needed, because a date has no decimal
   point.
5. **Season-controlled speed ratios.** The report adds each product's mean 10 m speed over ERA5's,
   averaged over August to October 2024 and over August to October 2025, computed from the long row
   set, so the November 2024 step in the monthly ratios is not read from month-to-month noise
   alone. It also adds, from the Previous Runs file, the share of null `_previous_day*` cells before
   and from 1 October 2025 and the UTC hours at which `wind_speed_100m_previous_day1` changes most,
   which independently support the source change.
6. **Wording.** The threshold, provenance, coverage and cell-selection corrections above; "before 1
   October 2025" for HRES's served grid; ENS's cadence and dissemination sentences as in the science
   review; and no work commitment on the page.

**Fold helpers moved into the shared package.** `cut_eras`, `rotate_folds`, `calendar_month_coverage`,
`uncovered_months` and `raise_on_uncovered_months` now live in `studies.cross_validation`, beside
`assign_folds`, with tests in `packages/studies/tests/test_cross_validation.py`, so that other
studies import them rather than copy them. The script's `with_three_eras` is `cut_eras` called with
`ERA_START_MONTHS`. No fold assignment or printed number changed: `--report-only` reproduces
`report.md`, `intervals.parquet` and `README.md` byte for byte, and every design's saved fold column
matches the package's output.
