# Plan: nearby weather-station observations in the past-solar study

**The question.** Instead of estimating the sunshine at a solar farm from a gridded product (CAMS, ERA5), what if a neighbouring Met Office weather station is used, or a blend of a gridded product and a station? The plan adds station arms to `docs/studies/weather-products-for-past-solar.md`, following the ENS section's precedent (#879): a section with its own shorter row set, every existing product refit on it, and a printed-number guard.

**The solution.** A new script `studies/beam_diffuse_split/station_past_solar.py` builds a row set from the page's common rows plus the MIDAS Open station observations, fits every arm with the existing out-of-fold XGBoost loop, and writes `report.md`. A chart script draws the figures from the report and the saved losses, and the page gets a section, a table row, key findings, "What to use" bullets and limitations.

## Verdict, size and the five triggers

- **What gets stored:** yes. The section is published prose and figures, and the run saves per-row losses, a fingerprint and a report under `data/studies/`.
- **The production serving path:** no. A study script and page only.
- **A degradation rule:** no.
- **More than one defensible design:** yes. The station-selection rule, the number of stations averaged, and how a blend keeps equal column counts each admit several designs.
- **Callers that could not be named without searching:** no. The only shared code touched is `studies.midas` (PR #886, merged first).

**Size: complex.** Reviews: an Opus code and mutation review of the diff, a Sonnet fixer, two Opus science reviews each with a Sonnet fixer (re-running whatever a reviewer asks), persona reviews (Met Office observations team, network planner, forecasting researcher, sceptical statistician), a one-rule-at-a-time prose sweep, and a Sonnet pre-merge check that reads the rendered HTML. Plan reviews are replaced by the first Opus science review of the design and first results, because the maintainer's brief fixed the design's shape. This shortening is stated in the PR body.

## Data and row set

- **Radiation:** `data/studies/weather/MIDAS-OPEN/uk_radiation_obs_hourly.parquet`, 10 stations, global irradiation only (diffuse and direct are null everywhere, so no beam split), read by `studies.midas.read_radiation` (W m⁻², hour ending at `time`, clipped at zero) then `null_night_spikes`.
- **Air temperature:** `uk_hourly_weather_obs.parquet`, 38 stations, an instant at `time`, read by `read_hourly_weather`.
- **Row set:** the page's common rows (`blend_products._solar_frame`) restricted to `time < 2026-01-01 00:00 UTC`, then to hours where every arm's input is present. The station data ends on 2025-12-31, so the station-arm row set ends there, about eight months before the main row set ends on 2026-09-10. Every product (CAMS, ERA5, and any other refit) is refit on this shorter row set, and its score is reconciled with the main row set's.
- **Folds:** `weather_products.with_eras` recomputed on the restricted row set (contiguous blocks of whole months, cut within each era). **Intervals:** `bootstrap_difference` and `bootstrap_absolute`, resampling whole calendar months and one of three fitting seeds, 2,000 resamples.
- **Metric:** mean absolute error as a percentage of each site's capacity, the study's `METRIC`.
- **Fits:** `PRIMARY_HYPER_PARAMETERS`, plus `SENSITIVITY_HYPER_PARAMETERS` on every planned contrast's arms. `colsample_bytree=1` everywhere.

## The station-selection rule (fixed before any score exists)

`studies.midas.select_nearest_stations`, with `MIN_COVERAGE = 0.99`. Stations are ranked by great-circle distance (rounded to the metre, ties to the lower `src_id`). A station is eligible when it has a usable value at no less than 99% of the site's candidate hours (the common rows before `2026-01-01`). The rule reads no score. Two stations are chosen independently: the nearest **radiation** station and the nearest **air-temperature** station, each from its own file's stations, so the temperature station may differ from the radiation station. Hours where any station an arm needs is missing are dropped from **every** arm's rows (the intersection), so all arms are scored on the same rows.

The coverage threshold is 0.99 rather than 1.0 because no station is complete: a threshold of 1.0 leaves at least one site with no eligible station. The rule as chosen skipped no nearer station at any site.

**Privacy:** the station-to-site mapping, every station coordinate and name, and each site's distance never appear in any output. The report and the page carry only pooled distance ranges (minimum to maximum across sites) and counts of distinct stations. Station ids are public and may appear in code.

## Arms

Every arm carries the same number of feature columns unless stated, `colsample_bytree=1`, and each arm's columns come from one function that returns a fixed-length tuple, printed into the report. `shared` is the study's seven shared columns (`solar_zenith_deg`, `solar_azimuth_deg`, `extraterrestrial_horizontal_w_m2`, `temp_c` (ERA5's), `hour_of_day`, `day_of_year`, `era_code`); `shared_no_temp` drops `temp_c`.

| Arm | Columns | Count |
|---|---|---|
| `cams_global` | `shared` + `ghi_cams` | 8 |
| `era5_global` | `shared` + `ghi_era5` | 8 |
| `station_global` | `shared_no_temp` + station air temperature + station ghi (nearest of each) | 8 |
| `cams_station_xgb` | `shared` + `ghi_cams` + station ghi | 9 |
| `cams_station_control` | `shared` + `ghi_cams` + the station ghi column permuted within (site, month, hour of day) | 9 |

## Planned contrasts (written before the first fit)

1. **Station against CAMS:** `station_global − cams_global`.
2. **Station against ERA5:** `station_global − era5_global`.
3. **CAMS blended with the station against CAMS alone, at equal column counts:** `cams_station_xgb − cams_station_control`. The control holds CAMS plus a climatology-permuted copy of the station's irradiance, so a padded CAMS-alone model carries the same number of columns as the blend.

All three also run at `SENSITIVITY_HYPER_PARAMETERS`. Each is reported pooled and, exploratorily, per generator.

## Exploratory arms (labelled so on the page)

- `station_ghi_era5temp`: `shared` + station ghi (the temperature swap removed), to separate what the station's temperature adds. 8 columns.
- `station_mean3`: `shared_no_temp` + the mean temperature and the mean irradiance of the three nearest stations of each kind. 8 columns.
- `station_rank2`, `station_rank3`: `station_global`'s columns from the second-nearest and third-nearest eligible station, which test how the score falls with distance. Their pooled distance ranges are reported. 8 columns each.
- `station_era5_xgb` and `station_era5_control`: `shared` + `ghi_era5` + station ghi, against the same with the station ghi permuted. 9 columns each.
- `station_shuffled`: `shared_no_temp` + station temperature + a permuted station ghi, the negative control for `station_global`. 8 columns.
- `cams_station_xgb − cams_global`: the blend against the plain, one-column-narrower CAMS arm (descriptive, because the columns are unequal).

Every arm's absolute error is reported, not only the contrasts. The "km-band" wording in the brief is read as the second- and third-nearest station arms, reported with their own pooled distance ranges, because only 10 radiation stations exist and a fixed 3 km or 6 km band would leave most sites empty.

## Honest limits the page must state

- A station arm tests how well a pyranometer some 17 to 31 km away stands in for the site's sunshine. It does not test the station's accuracy.
- The stations have no beam or diffuse split.
- All six sites take the same nearest radiation station (the report prints the count of distinct stations, not a mapping), so 60,000 rows are far fewer independent sites than they suggest. The intervals cover month-to-month weather and the fitting seed, not differences between sites or between stations.
- The station row set ends on 2025-12-31.
- QC flag 106 marks whole stations, so no flag is filtered on. Three pre-dawn or negative-value defects are repaired by the reader.

## Tests and reproducibility

- `packages/studies` changes only in PR #886, which carries the reader's tests. This PR adds no shared code, so the mutation pass on `packages/studies` is not repeated, and the study script has no unit tests. Its check is its own output: `report.md`, a row fingerprint (float columns cast to `Float32` before hashing), an arm-column-count guard, a duplicate-key guard, `check_no_missing` on every arm's inputs, and a printed-number guard in the chart script and the page check.
- Every number on the page comes from `report.md`. A number from a diagnostic run goes on the page only after a committed script prints it.
- One agent runs the published script at a time (worktrees share one data folder). Scratch work lives under `.claude/worktrees/scratch/`.

## Docs to update

`docs/studies/weather-products-for-past-solar.md`: the summary and key findings, the product table, a new section, "What to use", limitations, reproduction commands. Charts under `docs/studies/assets/`. The `studies/beam_diffuse_split` README if it lists scripts. Anonymisation: solar labels A to F only; time-series charts show days 1 to 7, percent of capacity, `aria=False` on data marks.

## Risks and open questions

- The "3 and 6 km-band distances" wording is interpreted as above. The coordinator may correct that reading.
- With one nearest station for every site, the blend contrast measures the value of one station, not of a network.
