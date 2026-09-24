# Plan: nearby weather-station wind in the past-wind study (phase 2)

**Problem.** [Which weather product best describes past wind?](../docs/studies/weather-products-for-past-wind.md)
scores gridded weather products at three wind farms (W1 to W3). It scores no weather-station
observation. The Met Office's MIDAS Open archive holds hourly 10 m wind from 18 stations in and
around the trial area. A network planner with no forecast product to hand might read the nearest
station instead, and a live service might blend a station with a forecast.

**Solution.** One new study script and one chart script, and one new results section on the page.
An XGBoost model per farm is given the nearest wind-reporting station's 10 m wind and compared with
one given ERA5's 10 m wind, and a blend of UKV and the station is compared with UKV given the same
number of its own columns. Two planned contrasts are fixed in this file before any fit. The tested
MIDAS reader and nearest-station helper in `packages/studies/src/studies/midas.py` are reused.

## Verdict, size and the five trigger answers

Worth doing. **Size: complex**, so both plan reviews and both diff reviews run, then the reviews
the maintainer asked for.

1. **What gets stored:** yes. A published page section and outputs under
   `data/studies/beam_diffuse_split/past_weather_v2/station_wind_arms/`. No Patito contract, no
   Delta table.
2. **Production serving path:** no.
3. **A degradation rule:** no. The page states that a station arm is not a live product: MIDAS Open
   is a retrospective archive, so the arm measures value, not a delivery route.
4. **More than one defensible design:** yes. Which station, how a missing station hour is handled,
   how direction is read, and which product the blend uses.
5. **Callers not nameable without searching:** no. New scripts only. `packages/studies/` changes
   only if a helper proves reusable, and then the mutation pass runs on it.

## Privacy rule (binding on every artefact)

**Never write which station serves which wind farm, a station coordinate or name, or a per-farm
distance** in any chart, page, report, script output, log, PR body, brief or commit message. The
script reads the private station metadata in memory and reports only pooled distance ranges (the
minimum and maximum over all farms and ranks together). Four further rules follow from the plan
review, because the hours a station misses are public in MIDAS and could be matched to a farm:

- No per-farm row counts, per-farm drop counts, per-farm station coverage or per-farm lag figures
  appear on the page, in a chart or in a PR body; the row count and the number of rows dropped are
  reported pooled over the three farms. Per-farm error tables labelled W1 to W3 are allowed.
- No station identifier appears in any output. The page states that quality-control flag 106
  marks whole stations, so no row is filtered on a flag, and names the one candidate station
  affected (62265) only as the dataset fact the request asked to state, never in relation to a
  farm and never saying whether it was chosen.
- No chart plots a station's wind series against dates or under a W label.

## Data and row set

- Stations: the 18 hourly-weather stations that report wind speed, listed in the request; the 8
  stations with one 09:00 return a day and wind estimated on Beaufort-scale midpoints are excluded,
  and the 12 stations that carry no wind at all are never candidates. Quality-control flag 106
  marks whole stations, so no row is ever filtered on a flag, and every candidate is treated alike:
  eligible only by the coverage rule below.
- Wind columns: `wind_speed_m_s` (knots converted, 10 m; a 10-minute mean ending at `time` by WMO
  convention, to be verified in the fact check; whole knots, so 0.51 m/s steps, and direction in
  10-degree steps) and `wind_direction` (degrees, north written as 360, calm as 0). Every wind row
  that carries a unit code has code 4 (anemometer, knots), so no estimated row enters; the report
  prints that check. The calm flag (direction 0) is set before any normalisation of 360 to 0. A calm
  row enters as sine and cosine both zero; 360 is north.
- "Observed": a station hour counts as observed when speed and direction are both non-null.
- Nearest station: `studies.midas.select_nearest_stations` with `k=1` and `min_coverage=0.9` of
  the farm's required hours, where the required hours are the page's rows in the window below. The
  rule reads no score and no target. A station hour that is missing drops the farm-hour from every
  arm, so every arm is scored on exactly the same rows.
- Rows: the page's own rows (`wind_products.common_rows(wind_products.joined(...))`), then
  restricted to the window 2024-08-12 to 2025-12-31 (17 calendar months, August 2024 partial) and
  to hours where the nearest station has a reading. **The window ends on 2025-12-31 because the
  MIDAS Open download holds calendar years 2017 to 2025** (`dataset-version-202607`; the page states
  what CEDA holds for 2026 as verified by the fact check). The window holds one UKV era only (the UKV
  upgrade of January 2026 is outside), so `era_code` is constant and is kept so every arm carries the
  page's columns. The plan review measured 34,183 page rows in the window and 34,156 after the
  station rule; the script reports the pooled figures.

## Planned contrasts (written before any fit)

| # | Contrast (first minus second) | Question it answers |
|---|---|---|
| S1 | `station_wind` − `era5_10m_wind` | Does the nearest wind-reporting station's 10 m wind describe past wind better than ERA5's 10 m wind? |
| S2 | `ukv_station_wind` − `ukv_padded_wind` | Does adding the nearest station to UKV, the page's product for historical features, lower the error? |

Every arm carries the shared columns (`hour_of_day`, `day_of_year`, `era_code`) plus:

- `station_wind`: the station's speed, sine and cosine of its direction (3 columns).
- `era5_10m_wind`: ERA5's 10 m speed and the sine and cosine of ERA5's 100 m direction (3
  columns). The download holds no 10 m ERA5 direction, and the page says so.
- `ukv_station_wind`: UKV's page columns (100 m speed, sine and cosine of its 100 m direction, 10 m
  speed) plus the station's three columns (7 wind columns).
- `ukv_padded_wind`: UKV's page columns plus UKV's own served 80 m speed and the sine and cosine
  of its 80 m direction (7 wind columns), so the blend gains no advantage from column count. Only
  the three 80 m columns are taken from `UKV_80M_COLUMNS`, which also holds the 10 m speed.

Every contrast is rerun at the second hyperparameter setting. Every arm shown "on the same rows",
including `ukv_wind` and every other product, is refitted on the station rows and never read from
the published losses, because fold layout alone moves an error by 0.02 to 0.03 points.

**What no planned contrast can separate.** S1 compares a point anemometer at 10 m, 6 to 18 km from
the farm (the pooled range of the nearest eligible station), with a 0.25-degree gridded value. It
tests how well a 10 m anemometer stands in for a hub-height turbine, and does not separate station
distance, height, terrain, the station's own siting, whole-knot and 10-degree quantisation, and the
grid. S2 mixes the station's information with the extra-columns control built from UKV's 80 m
fields. Three wind farms are few independent sites, and 17 months are few independent weather
episodes; the page states both beside every pooled interval. MIDAS Open is a yearly retrospective
archive, so the page scopes "not a live product" to that archive, and does not claim that station
data cannot be had live.

## Exploratory arms (labelled so)

- `station_k3_wind`: the mean of the available speeds among the three nearest eligible stations
  (`k=3`), direction from the mean wind vector (calm hours contribute zero components), on the same
  rows (the nearest station's hours decide the rows).
- **Shear control, not a fitted arm.** A power-law exponent of 1/7 (a common open-country value,
  chosen a priori; it is a guess, not fitted) scales the nearest station's speed to 100 m. A tree
  model is invariant to a monotone rescaling of one column, and the plan review confirmed the
  predictions are bitwise identical. The script therefore fits one fold of one farm on the raw and
  the scaled speed, prints the maximum difference in predictions into the report, and fits no arm.
  The page says in one sentence that a fixed exponent cannot help a tree model. An informative
  shear arm would need an exponent that varies by hour, which imports another product's
  information, so none is planned.
- `ukv_station_wind` − `ukv_wind`, with `ukv_wind` refitted on the station rows.
- Optional, and only if cheap: S1 and S2 restricted to August to December, the calendar months
  that occur in two years of the window.
- The station's absolute error beside every product's on the same rows (with the column counts
  stated), by farm (W1 to W3, without any station identity), and the S1 and S2 contrasts by farm.
- A Bonferroni-adjusted interval for S1 and S2. The window holds one UKV era, so S2 scores UKV
  before its January 2026 upgrade only; the page says so, because the page's UKV recommendation
  rests largely on later results. S2 tests a station against more UKV columns (the padding also
  gives UKV its own 80 m to 100 m shear), not against nothing.

## Fairness and checks

- Equal column counts within each contrast; `colsample_bytree=1`; every arm's column list printed
  into the report; month-block folds (one era, so five contiguous blocks per farm of 4, 3, 4, 3 and
  3 months); month-resampled paired bootstrap; a deterministic row fingerprint with floats cast to
  Float32 before hashing; the printed-number guard from PR #885 (`check_page_numbers.py`) run on the
  new section after the rebase.
- **Calendar-month coverage check (issue #868):** the report prints, before any fit, the months
  each fold holds and whether every scored calendar month has training rows from some other fold.
  The script raises on an uncovered calendar month that occurs in two years, as phase 1 does, and
  prints the single-year months (January to July), which are 42.2% of the scored rows and are
  scored by models that never trained on that calendar month, so `day_of_year` extrapolates across
  gaps of up to 4 months for every arm. The page states that figure.
- Station data checks printed into the report, pooled: hours missing per eligible station,
  coverage of the chosen stations, wind speed range, direction bins, the calm rule, the unit-code
  check, and a check that the station's speed correlates with UKV's 10 m speed best at zero hour
  offset (the plan review found 0.848 at zero, 0.837 at +1 hour, 0.814 at -1 hour).
- Pooled distance ranges only (see the privacy rule).

## Product facts to verify before writing

A fact check (Sonnet, then an Opus reviewer) verifies from the Met Office and CEDA documentation:
the MIDAS Open release schedule and why the data end in December 2025, that MIDAS hourly wind is a
10-minute mean ending at the hour (the wording in the user guide), anemometer height (10 m
standard), and unit codes. Nothing in the page states a station's identity.

## Files

- `studies/beam_diffuse_split/station_wind_arms.py` (new): builds rows, jobs, fingerprint, report,
  in the structure of `wind_icon_dream.py` on `main`, with `wind_products.joined` and
  `common_rows`, `weather_products.with_eras`, and the MIDAS helpers. The printed-number guard
  (`check_page_numbers.py`) and the calendar-month coverage check come from PR #885, not yet on
  `main`: this branch is rebased onto `main` once #885 merges, and shared helpers go into
  `packages/studies/` with tests if the diff review asks.
- `studies/beam_diffuse_split/station_wind_arms_charts.py` (new).
- `docs/studies/weather-products-for-past-wind.md`: one results section, one short "What to use"
  bullet, one Key-findings bullet, about two charts, "Data and methods" and "Limitations" additions,
  reproducing commands.
- `docs/studies/assets/`: the new SVGs, optimised with svgo.
- `packages/studies/`: unchanged unless a helper proves reusable.

## Tests and verification

Study scripts are not unit-tested; the report is their check. Verification: `ruff check`,
`ruff format`, `ty check`, `pytest`, `pymarkdown scan`, `mkdocs build --strict`,
`check_docs_links.py`, `pydoclint`, and reading the built HTML under `site/`. A grep of every
artefact for station identifiers and coordinates runs before each push.

## Risks and open questions

- **A station is a proxy for a hub-height turbine 6 to 18 km away**, so a poor result is expected
  and says little about station data in general. The page says so plainly.
- **Seventeen months and three farms** give wide intervals and a weak season coverage.
- **A per-farm view could identify a station.** Only pooled distances and counts are reported.

## Reviews this plan buys

Both plan reviews, then both diff reviews, two science reviews, the persona reviews (including a
Met Office observations-team member), a prose sweep and a pre-merge check of the rendered HTML.

## Decisions made during implementation

These decisions were taken while writing `studies/beam_diffuse_split/station_wind_arms.py`, where
this plan was silent. They are recorded before the first fit.

- **Bonferroni family.** The plan asks for adjusted intervals for two contrasts. Each contrast is
  reported at two hyperparameter settings, so the family is 4 planned intervals and the adjusted
  level is 98.75% (0.05 / 4 in each tail pair), not the 97.5% that two intervals alone would give.
  The resampler is the month-and-seed resampling of `studies.bootstrap`, and the script asserts that
  it reproduces `bootstrap_difference`'s 95% interval before using it at the adjusted level.
- **Pre-fit mode.** The script has a `--checks-only` flag that runs every pre-fit check and prints
  it, fitting and writing nothing, so the checks can be run and committed before the first fit.
- **Copied helpers.** The interval log, fingerprint, script-commit check, calendar-month coverage
  check and the Bonferroni resampler are copied in the smallest form from PR #885's
  `ens_hres_past_wind.py`, each marked as mirroring it. They are replaced by imports after the
  rebase onto `main`.
- **Coverage table is pooled.** The calendar-month coverage table prints rows per (fold, calendar
  month) summed over the three farms, never per farm, because a per-farm row count would carry the
  hours a station misses. The raise on an uncovered calendar month uses the per-farm cells.
- **Per-farm tables carry no counts.** Per-farm error and contrast rows print the error, its
  interval and the folds agreeing, and no row or month count; `intervals.parquet` stores null counts
  for per-farm intervals.
- **k=3 arm.** The mean speed is the mean of the speeds of whichever of the three nearest eligible
  stations have an observed hour. The direction is that of the mean of the stations' wind vectors
  (speed times sine and cosine, a calm hour contributing zero), normalised to a unit vector, and
  sine and cosine are both zero when the mean vector is zero. `station_k3_wind` is compared with
  `station_wind`, as the only exploratory k=3 contrast.
- **Exploratory contrasts.** `ukv_station_wind` minus `ukv_wind` is reported at both settings. The
  k=3 contrast is reported at the primary setting only, since the k=3 arm is not fitted at the
  second setting.
- **August-to-December restriction.** S1 and S2 scored on August to December only restrict the
  scored rows of the fitted arms (models trained on every month), at both settings. This is cheap
  and exploratory.
- **Hour-offset sign.** An offset of +1 hour pairs UKV at time `t` with the station reading stamped
  `t` + 1 hour. The script raises if the correlation of the station's speed with UKV's 10 m speed
  does not peak at offset 0.
- **Station-hour join.** A station hour is joined to the farm's hour of the same UTC label, with no
  shift; the check above is what tests that this is the right alignment.
- **Shear control.** Farm W1's fold 0, seed 0, at the primary setting, fitted on the raw and the
  speed scaled by (100 / 10) ^ (1 / 7); the script prints the largest prediction difference in MW.
- **Station facts printed.** Pooled ranges over the three farms only: distances (nearest and third
  nearest), coverage, nearer stations skipped, and missing-hour share over eligible (farm, station)
  pairs, plus the counts of rows before and after the station rule.
- **Pooled caveat line.** Every pooled table is preceded by "Three wind farms are few independent
  sites; N rows, M months."
