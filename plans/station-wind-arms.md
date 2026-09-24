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
minimum and maximum over all farms and ranks together). Per-farm result tables are labelled W1 to
W3 and never carry a station identifier or distance. Station identifiers never appear in output.

## Data and row set

- Stations: the 18 hourly-weather stations that report wind speed, listed in the request; the 8
  stations with one 09:00 return a day and wind estimated on Beaufort-scale midpoints are excluded,
  and the 12 stations that carry no wind at all are never candidates. Quality-control flag 106
  marks whole stations, so no row is ever filtered on a flag; station 62265 is affected (almost
  every wind row), the page says so, and 62265 stays eligible or not only by the coverage rule
  below.
- Wind columns: `wind_speed_m_s` (knots converted, 10 m, instantaneous at `time`) and
  `wind_direction` (degrees, north written as 360, calm as 0). A row with direction 0 is calm and
  enters as sine and cosine both zero; 360 is north.
- Nearest station: `studies.midas.select_nearest_stations` with `k=1` and `min_coverage=0.9` of
  the farm's required hours, where the required hours are the page's rows in the window below. The
  rule reads no score and no target. A station hour that is missing drops the farm-hour from every
  arm, so every arm is scored on exactly the same rows.
- Rows: the page's own rows (`wind_products.common_rows(wind_products.joined(...))`), then
  restricted to the window 2024-08-12 to 2025-12-31 and to hours where the nearest station has a
  reading. **The window ends on 2025-12-31 because the MIDAS Open download holds calendar years 2017
  to 2025** (`dataset-version-202607`; the page states what CEDA holds for 2026 as verified by the
  fact check). The window holds one UKV era only (the UKV upgrade of January 2026 is outside), so
  `era_code` is constant and is kept so every arm carries the page's columns.

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
  of its 80 m direction (7 wind columns), so the blend gains no advantage from column count.

Every contrast is rerun at the second hyperparameter setting.

**What no planned contrast can separate.** S1 compares a point anemometer at 10 m, tens of
kilometres from the farm, with a 0.25-degree gridded value; it tests how well a 10 m anemometer
stands in for a hub-height turbine, and does not separate station distance, height, terrain, the
station's own siting, and the grid. S2 mixes the station's information with the extra-columns
control built from UKV's 80 m fields. Three wind farms are few independent sites, and the page
states that beside every pooled interval.

## Exploratory arms (labelled so)

- `station_k3_wind`: the mean of the three nearest eligible stations' speeds (`k=3`), direction
  from the mean wind vector, on the same rows (the nearest station's hours decide the rows).
- `station_shear_constant_wind`: the nearest station's speed scaled to 100 m with a power-law
  exponent of 1/7 (a common open-country value, chosen a priori and a guess, not fitted). A tree
  model is invariant to a monotone rescaling of one column, so this arm is expected to reproduce
  `station_wind` exactly; the report prints whether it does, which makes it a control.
- `station_shear_era5_wind`: the nearest station's speed multiplied, hour by hour, by ERA5's own
  100 m to 10 m speed ratio. The exponent here is not chosen by hand, but it imports ERA5's
  information, so the page reads the arm as station-plus-ERA5-shear, not a station alone.
- The station's absolute error beside every product's on the same rows (with the column counts
  stated), by farm (W1 to W3, without any station identity), and the S1 and S2 contrasts by farm.
- A Bonferroni-adjusted interval for S1 and S2.

## Fairness and checks

- Equal column counts within each contrast; `colsample_bytree=1`; every arm's column list printed
  into the report; month-block folds (the date range is short, so the plan states the number of
  months and the fold count); month-resampled paired bootstrap; a deterministic row fingerprint
  with floats cast to Float32 before hashing; the printed-number guard from phase 1
  (`check_page_numbers.py`) run on the new section.
- **Calendar-month coverage check (issue #868):** before any fit, every scored calendar month must
  have training rows from some other fold. With a 16-month window, each calendar month from
  August to December occurs in two years and January to July in one, so the report prints which
  months cannot be covered and the page states it.
- Station data checks printed into the report: hours missing per eligible station pooled, coverage
  of the chosen stations, wind speed range, direction bins, the calm rule, and a check that the
  station's speed correlates with UKV's 10 m speed best at zero hour offset.
- Pooled distance ranges only (see the privacy rule).

## Product facts to verify before writing

A fact check (Sonnet, then an Opus reviewer) verifies from the Met Office and CEDA documentation:
the MIDAS Open release schedule and why the data end in December 2025, that MIDAS hourly wind is a
10-minute mean before the hour or an hourly value (the wording in the user guide), anemometer height
(10 m standard), and unit codes. Nothing in the page states a station's identity.

## Files

- `studies/beam_diffuse_split/station_wind_arms.py` (new): builds rows, jobs, fingerprint, report,
  in the structure of `ens_hres_past_wind.py`.
- `studies/beam_diffuse_split/station_wind_arms_charts.py` (new).
- `docs/studies/weather-products-for-past-wind.md`: one results section, one "What to use" bullet,
  one Key-findings bullet, "Data and methods" and "Limitations" additions, reproducing commands.
- `docs/studies/assets/`: the new SVGs, optimised with svgo.
- `packages/studies/`: unchanged unless a helper proves reusable.

## Tests and verification

Study scripts are not unit-tested; the report is their check. Verification: `ruff check`,
`ruff format`, `ty check`, `pytest`, `pymarkdown scan`, `mkdocs build --strict`,
`check_docs_links.py`, `pydoclint`, and reading the built HTML under `site/`. A grep of every
artefact for station identifiers and coordinates runs before each push.

## Risks and open questions

- **A station is a proxy for a hub-height turbine tens of kilometres away**, so a poor result is
  expected and says little about station data in general. The page says so plainly.
- **Sixteen months and three farms** give wide intervals and a weak season coverage.
- **A per-farm view could identify a station.** Only pooled distances are reported.

## Reviews this plan buys

Both plan reviews, then both diff reviews, two science reviews, the persona reviews (including a
Met Office observations-team member), a prose sweep and a pre-merge check of the rendered HTML.
