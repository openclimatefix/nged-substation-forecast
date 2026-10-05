# Plan: CEDA UKV or ERA5 as the estimate of past wind and temperature, 2019 to 2026 (issue #1024)

**The question is whether the main Flexpectation work should take past wind speed and past air
temperature from the Met Office's UKV as archived by CEDA (UKV-CEDA) or from ERA5.** The main work
already takes past irradiance from CAMS, and this study does not reopen that choice. No published
page compares UKV with ERA5 before August 2024, so nothing says whether UKV's lead over ERA5 on wind
power (0.44 points of capacity [0.24, 0.63], August 2024 to September 2026) holds in 2019 and 2020.
The CEDA stores start in September 2019, so the study can test those years.

**The plan runs two experiment sets and fixes the rule that combines them before any result
exists.** Set A scores each product against Met Office station observations with no machine
learning. Set B fits one XGBoost model per metered generator and per product, and scores the model's
power error. Five planned contrasts answer the question. The rule defaults to ERA5 unless UKV-CEDA
clears a stated margin. Every score is split by year, by season, and into an early window (September
2019 to December 2020) and a late window. The plan finds four facts that change the issue's design,
listed under "Verdict, size and departures".

## 1. Verdict, size and departures

**Verdict: worth doing, with four departures from the issue body.**

- **Set A has four stations, not 18.** The CEDA stores are cropped to the private trial-area box.
  Counting stations against that box (counts only, no bounds printed) finds 4 of the 30 hourly
  MIDAS stations inside it, and 1 of the 10 radiation stations. All 4 report wind speed, direction,
  temperature, dew point, and sea-level pressure on 98% or more of hours since September 2019. The
  18 stations that report wind lie mostly outside the crop. Reading them needs a wider UKV download,
  which this study does not do (section 9).
- **ERA5 data is missing for set A.** ERA5 wind at station cells covers 2019-09 to 2023-12 only. The
  4 stations need 2024-01 to 2025-12 (a small Climate Data Store request, about 1.5 hours, the same
  recipe as `fetch_era5_wind_2019_2023.py`). ERA5 temperature is on disk for the 20 grid cells that
  contain all 4 stations.
- **2019 holds three scored months.** The Met Office's Parallel Suite 43 (PS43) took effect on
  2019-12-04, so the pre-PS43 era is September to November 2019. A calendar year with fewer than
  `MIN_MONTHS_FOR_INTERVAL` (6) months gets a point estimate and no interval. The early-years test
  therefore pools 2019 and 2020 into one window of 15 months.
- **Solar differs only in temperature.** The solar models read CAMS irradiance and, in the existing
  studies, ERA5 air temperature. Temperature is the one non-irradiance variable they use, so the
  solar contrast tests temperature alone, and 10 m wind speed is an exploratory addition.

**Size: complex.** All five triggers from `plan-issue` step 3 are answered below.

| Trigger | Answer |
|---|---|
| Changes what gets stored | Yes. A published page, a write-once results folder, and a new package module |
| Touches the production serving path | No. Production never imports `studies` |
| Touches a degradation rule | No |
| More than one defensible design | Yes. Station set, UKV lead, wind heights, and how sets A and B combine |
| Callers not nameable without searching | Yes. `packages/studies` is shared by every study folder |

**Reviews: all four.** Two plan reviews (simplicity, then correctness) and two diff reviews
(correctness, then mutation, because `packages/studies/` changes). The `study` skill adds two Opus
scientific-validity reviews, one fresh Opus review of each script before its first run, and the
persona and prose reviews. This plan has had no review yet. The PR is a draft until the maintainer
approves.

## 2. The question, the arms, and the variables

**Set A scores each product's hourly value against the station's reading.** The two products are
ERA5 (nearest 0.25 degree cell) and UKV-CEDA (nearest 2 km cell, at most 2 km from the station).
Variables: 10 m wind speed (primary), and 2 m (ERA5) or 1.5 m (UKV) air temperature (primary).
Exploratory: 10 m wind direction (circular error), dew point, and sea-level pressure. The last two
need a small ERA5 download of dew point and pressure at the 4 cells from Open-Meteo's mirror.

**Which stations.** The 4 hourly stations inside the UKV crop. The 14 other wind stations are out
of reach (section 9). The one radiation station in the crop gives an exploratory irradiance row
that cannot change the choice of CAMS. `studies.midas.select_nearest_stations` returns the mapping
to the script, and the page prints only pooled distance ranges, because a station distance to a
generator helps locate the generator. A station farther than 40 km from every metered generator is
context for set A and is not read as evidence about the power scores. The coverage script (section
7) prints the pooled count.

**Which UKV lead stands in for past weather.** The primary reads the freshest archived run for each
hour: the run at 00, 06, 12, or 18 UTC at or before the hour, so leads are 0 to 5 hours (mean 2.5).
This is the lowest lead the 6-hourly store can give, and the closest match to the T+0 analysis that
the wind page scored from Open-Meteo. Because lead equals hour of day modulo 6, the XGBoost models
can learn lead-specific bias from `hour_of_day`. The study states this as part of what a consumer
gets. Two exploratory checks: set A prints each product's error against UKV lead (0 to 54 hours),
and set B adds a "previous run" arm that reads leads 6 to 11. Neither check changes the primary.

**Station-versus-grid-cell handling.** A station reads one point, so part of every gap is the
cell's spatial averaging. The primary set A score is the mean absolute error after removing each
(station, product, calendar month) mean error. This matches what the XGBoost models in set B do,
because they recalibrate per generator, and it removes systematic offsets from station exposure and
cell orography. The raw mean absolute error is printed beside it as exploratory. Where the store
covers at least 90% of the 2 km cells inside the station's ERA5 cell, an exploratory arm averages
UKV over those cells. If UKV averaged to ERA5's scale still beats ERA5, the gap is not resolution.

**Set B wind arms (3 wind farms, W1 to W3).** Each arm is a fixed-length tuple built by one function
per arm type, printed into the report, 7 columns each:

- Shared (3): `hour_of_day`, `day_of_year`, `era_code`.
- `era5_wind` (as available, 4): 100 m speed, sine and cosine of 100 m direction, 10 m speed.
- `ukv_ceda_wind` (as available, 4): 10 m speed, sine and cosine of 10 m direction, 925 hPa speed.
- `era5_10m` and `ukv_ceda_10m` (matched height, 4 each): 10 m speed, sine and cosine of 10 m
  direction, and a duplicate of the 10 m speed to equal the count. `colsample_bytree` is absent
  (1), so the duplicate carries no information.

UKV-CEDA has no 100 m wind, so the as-available contrast mixes product and height, and the matched
arms separate them. The 1000 hPa level is excluded because the files hold NaN wherever the level is
below ground, and a rule on NaN would filter by one product's values. ERA5 reads the nearest cell
(the centre of each 3 by 3 block already on disk). The 3 by 3 block mean is an exploratory arm.
Power is the hour centred on the label (stamps shifted back 30 minutes before
`studies.power.hourly_from_half_hourly`), because both products' wind is instantaneous. An
exploratory `hour_ending` pair scans the offset for both.

**Set B solar arms (6 solar farms, A to F).** Each arm has 10 columns: shared (7: `solar_zenith_deg`,
`solar_azimuth_deg`, `extraterrestrial_horizontal_w_m2`, `hour_of_day`, `day_of_year`, `era_code`,
and the arm's temperature), plus CAMS global, beam, and diffuse irradiance (3). The two planned arms
differ only in the temperature column: `era5_temp` and `ukv_ceda_temp`. Temperature is the mean of
the instants at the hour's two ends for both products (the power hour ends at the label). A `no_temp`
arm swaps temperature for a duplicate of `hour_of_day`, to show whether temperature adds anything.
Exploratory: both products with 10 m wind speed added (needs ERA5 10 m wind at the 20 grid cells,
about 20 points for 7 years from Open-Meteo's mirror, whose wind matches the Climate Data Store to
the mirror's 0.1 km/h rounding).

**The main work's other weather features, and who can score them.** The list comes from
`conf/model/xgboost.yaml`.

| Feature | Stations | UKV-CEDA | ERA5 on disk | Scored here |
|---|---|---|---|---|
| Temperature 2 m | yes | 1.5 m | yes | set A and B (planned) |
| Wind speed 10 m | yes | yes | yes | set A and B (planned) |
| Wind direction 10 m | yes | yes | yes | set A (exploratory) |
| Dew point 2 m | yes | yes | no | set A after a small fetch (exploratory) |
| Sea-level pressure | yes | yes | no | set A after a small fetch (exploratory) |
| Wind speed and direction 100 m | no | no | yes | cannot be compared |
| Surface pressure, 500 hPa height | no | no | no | cannot be scored |
| Shortwave radiation | 1 station | yes | yes | context row only |
| Precipitation type, windchill | no | no | derived | cannot be scored |

## 3. Decision rule, fixed before any result

**Margins.** A contrast "clearly" favours a product only when its 95% interval lies wholly on one
side of zero and the point estimate is beyond a margin. Margin for set A is 5% of ERA5's error on
the same rows (the wind page's 0.44 points was 6% of ERA5's error, so the margin sits just below the
one effect already seen). Margin for set B is the minimum detectable difference below, rounded to
0.01 points. A statistically significant difference smaller than the margin reads "small, not
clear".

**Smallest detectable difference, computed from the design.** The detectable difference is 2.8
times the standard error of the paired month-resampled difference (80% power, 5% level, two sided).
The planning figures use published intervals, whose half-width divided by 1.96 gives the standard
error: solar about 0.040 points at 21 months (the day-1 blend contrast on the UKV-CEDA blends page)
and wind about 0.10 points at 25 months (the Open-Meteo UKV against ERA5 contrast). Standard error
scales with the square root of the number of months, and set B has about 83 scored months, so the
detectable difference is **about 0.06 points for solar and about 0.16 points for wind**. The real
solar contrast changes only temperature and so may be smaller than the 0.06 points it can resolve.
`design_resolution.py` recomputes both numbers from the saved per-row losses of the blends and wind
studies before any fit, writes them to `design.md`, and the margins are then frozen from that file.

**Planned contrasts (five, labelled "(planned)").** Each is UKV-CEDA minus ERA5, with a negative
value favouring UKV-CEDA.

1. **P1, set A:** 10 m wind speed, bias-removed mean absolute error against the 4 stations, 2019-09
   to 2025-12.
2. **P2, set A:** air temperature, the same score and rows.
3. **P3, set B wind:** the as-available arms, error as percent of capacity, at both hyperparameter
   settings.
4. **P4, set B solar:** `ukv_ceda_temp` minus `era5_temp`, at both settings.
5. **P5, set A wind:** the early-window contrast minus the late-window contrast of P1, which asks
   whether UKV's relation to ERA5 differs in 2019 and 2020.

Each planned contrast is read at its interval and margin. The five are not corrected for multiple
comparisons, and the rule below needs two to agree before it recommends UKV-CEDA.

**How the sets combine, by variable.**

- **Wind speed: set B decides when the sets disagree.** Set B measures what the main work does with
  wind, which is to predict power after per-generator recalibration. Set A uses 4 stations in one
  box, and UKV's 2 km cell favours a point observation. Set A is context on disagreement.
- **Temperature: set A decides.** The generators are not demand, so set B cannot say how temperature
  helps a demand forecast. Set B vetoes only: if P4 clearly favours ERA5, the study does not
  recommend UKV-CEDA temperature for solar models.

| Deciding leg reads | Study reports |
|---|---|
| UKV-CEDA clearly better | Recommend UKV-CEDA, subject to the early-years test and the licence |
| ERA5 clearly better | Recommend ERA5 |
| No clear difference, or small | Recommend ERA5 (default: no licence limit, no run gaps, one physics version) |

**Early-years test.** UKV-CEDA is recommended for training history from 2019 only if the deciding
leg's early-window point estimate is negative and its upper 95% bound is below zero plus the margin.
Otherwise the study recommends UKV-CEDA from 2021 only, and says that an era-mixed design is
untested. P5 shows whether the windows differ.

**A conditional on the licence.** Every UKV-CEDA recommendation carries the licence verdict in
section 9.

## 4. Row set, gap rule, and era rule

**Rows are the intersection across every arm, decided from the target and availability, never from
a product's values.** An hour stays only if the target exists and every arm's input exists. Set B
also drops the hours the other wind and solar studies drop: wind hours holding an exactly zero
half-hour (`wind_product_frames.common_rows`), and for solar the commissioning ramp and the
export-cap and active-network-management hours (`solar_product_frames.common_rows`). Set A keeps
a station-hour if the station has a reading and both products have a value.

**Gap rule for CEDA runs.** Across the three stores the counts are 9,981 complete, 15 partial, and 82
missing runs, plus 260 runs on 65 days that CEDA has no directory for. That is 342 of 10,336 runs
(3.3%). Per store, from `lineage.json`: `UKV-CEDA` 3,420 complete, 4 partial, 16 missing;
`UKV-CEDA-part2` 3,202, 11, 59; `UKV-CEDA-part3` 3,359, 0, 7. An hour whose freshest run is missing
or partial is dropped for every arm. It is not served by an older run, because that changes the lead
mid-series. The coverage script prints the share of hours lost per month, re-checks the 65 days
against CEDA and the 15 partial runs, and stops if one month loses more than 25% of hours. An
exploratory "previous run" arm reads leads 6 to 11 and keeps those hours.

**Eras.** UKV-CEDA has three eras. Era 0 is runs before 2019-12-04 (PS43, RAL2-M), era 1 is 2020-01
to 2025-12, and era 2 is 2026-02 onward (PS47, 2026-01-21, RAL3). The straddling months, 2019-12 and
2026-01, are dropped for every arm in both sets. PS45 (May 2022) and PS46 (May 2025) are not era
boundaries: the repo's NWP-upgrades table finds no UKV physics change at either. **The date and
content of PS44 are unknown in the repo.** Before the first build, the Met Office release notes are
checked for PS44. As a data check, `verify_inputs.py` also plots the monthly UKV-minus-ERA5 mean and
UKV-minus-station mean for wind and temperature, to find any step the table misses. A step found
becomes an era boundary and a deviation from this plan, recorded on the page. Every arm gets
`era_code` (0, 1, 2). Folds are cut within each era.

## 5. Folds, intervals, settings, metric, hardware

- **Folds:** `studies.cross_validation.cut_eras` with `first_months=("2020-01", UKV_UPGRADE_MONTH)`,
  five blocks of whole months per (site, era). Era 0 has 3 months, so three of its folds are empty.
  `search_fold_offsets` runs on the real row set, and `raise_on_uncovered_months` stops the build if
  any calendar month is held out of every training row. Months of 2019 and 2026 also occur in other
  years, so no one-year-only month is expected.
- **Intervals, set B:** `studies.bootstrap` resamples whole calendar months, paired across arms,
  with one of the three fitting seeds, 2,000 resamples. **Set A:** `bootstrap_row_difference`
  resamples whole months and has no seed. Both cover month-to-month weather, not differences between
  stations or generators. Neighbouring generators and the 4 stations share their weather.
- **Hyperparameters:** `PRIMARY_HYPER_PARAMETERS` and `SENSITIVITY_HYPER_PARAMETERS` on P3, P4, and
  their control contrasts. A verdict needs both settings to agree, and the page says where a
  contrast changes sign or significance.
- **Metric:** mean absolute error as percent of capacity for set B, each row divided by its own
  generator's `effective_capacity` before any mean. Set A reports metres per second and kelvin.
  Every arm's absolute error is reported, not only the contrasts.
- **Splits:** each calendar year with at least 6 months (point estimate only below that, 2019
  included), each half-year (October to March, April to September), and the early and late windows.
  All are exploratory except the early and late reading of the deciding leg.
- **Hardware:** `device="cuda"`. Run `nvidia-smi` and `uptime` first. The workstation's GPU is idle
  now and the load average is about 1.6. The maintainer's note is that using all cores is fine, so
  CPU load is not a reason to wait. State the device on the page and refit one arm on the CPU for the
  noise floor.

## 6. Controls

- **Negative control, set B:** each product's weather columns shuffled within (site, year-month,
  hour of day), under shuffle seeds 0 and 1000 (the blends study's recipe). The shuffled-UKV minus
  shuffled-ERA5 contrast shows the difference the pipeline produces from nothing. Run for wind and
  solar. A product contrast smaller than the control's own spread is not read.
- **No-information arm, solar:** `no_temp` shows how much temperature adds at all. If neither
  product's temperature beats `no_temp`, P4 says nothing about the products.
- **Positive control, set B:** a planted target. The target is an arm's out-of-fold prediction plus
  that arm's residuals permuted within site and month. It is planted once on ERA5's arm and once on
  UKV-CEDA's arm, at both technologies, primary setting. The planted product's arm must win with an
  interval that excludes zero. A null reading of P3 or P4 is read as "no effect" only if the
  positive control passes or the interval bounds the effect, as in "an effect as large as 0.1 points
  is not excluded".
- **Set A:** a lag scan of each product against the station at -3 to +3 hours must put the minimum
  error at lag 0. It checks the timestamp convention. Station wind covers the 10 minutes ending 10
  minutes before the label, and both products are instantaneous at the label.

## 7. Scripts, files, and outputs

**A new folder `studies/ukv_vs_era5/`, because a script may import only from its own folder,
`studies.*`, and reviewed packages.** The existing build and fit scripts live in
`studies/nwp_forecast_comparison/` and cannot be imported. The folder follows the build, verify, fit,
chart pattern. Each script has one job.

| Script | Job |
|---|---|
| `design.py` | Constants: windows, margins, station rule, and one arm-columns function per arm type |
| `design_resolution.py` | Computes the detectable differences into `design.md` before any fit |
| `check_coverage.py` | The pre-fit coverage check (below) |
| `build_inputs.py` | Builds set A station-hours and set B wind and solar frames. `--dry-run` builds one month and writes nothing |
| `verify_inputs.py` | Column counts, lag scan, monthly step plots, lead distribution |
| `score_stations.py` | Set A scores and intervals, no fit |
| `fit_arms.py` | Set B fits. `--dry-run` lists fits, `--check` runs the padding check, `--verified` runs after verify |
| `report_decision.py` | Reads the intervals and applies section 3 by code, writing `decision.md` |
| `ukv_vs_era5_charts.py` | Figures. Every shared number is read from the reports |
| `README.md` | Maps each script to the page |

**Pre-fit coverage check (`check_coverage.py`).** It prints, with no value-based filter: runs by
status per store, hours lost per month, ERA5 coverage per product, station coverage per year, the
pooled station distance range, power hours per generator-year (a generator-year below 2,000 scored
hours is not shown separately), scored months per era and per window, `search_fold_offsets`, and the
detectable difference. It exits non-zero if a guard fails. No build runs until it exits 0.

**New code in `packages/studies/`, with tests.**

- `ukv_ceda_stores.py`: open the three stores, map a slot to its store and status, read the freshest
  run's leads 0 to 5 at chosen cells. `build_ukv_ceda_inputs.py` has similar readers tied to the
  120-hour profile. The plan copies the needed logic into the new tested module and leaves
  `build_ukv_ceda_inputs.py` untouched, so no published output needs a bit-for-bit proof. Removing
  the duplicate is a later change (section 9).
- `station_scoring.py`: pair a station with a product hour, remove the per-(station, product,
  month) mean error, and wrap `bootstrap_row_difference`.
- `sources.py` gains `UKV_CEDA_PRODUCT_DIR`, `UKV_CEDA_PART2_PRODUCT_DIR`,
  `UKV_CEDA_PART3_PRODUCT_DIR`, and `UKV_VS_ERA5_DIR = study_dir_for(study="ukv_vs_era5")`.

**Data paths.** Scripts name data only by product and constant (`ERA5_PRODUCT_DIR`,
`ERA5_WIND_2019_2023_PRODUCT_DIR`, `CAMS_PRODUCT_DIR`, `MIDAS_OPEN_PRODUCT_DIR`, the three UKV-CEDA
constants). **The data layout is migrating from `data/studies/weather/<PRODUCT>` to
`data/studies/downloads/...`, so no script runs until the migration has finished and the constants
resolve to the new paths.** The new downloads go in `studies/weather_downloads/` as resumable
scripts under the `data-download` skill, each with a validator.

**Output.** `UKV_VS_ERA5_DIR` is write-once: `refuse_to_overwrite` stops a second run, and a result
a merged page quotes moves to `superseded/` first. The folder holds the frames and build stamp, each
arm's per-row losses and out-of-fold predictions (site, time, fold, seed, arm, setting, actual
value), the interval tables as parquet, and `report.md`, `design.md`, and `decision.md`. Stamps
record the arm columns, the XGBoost version, the device, and the hashes of the inputs.

**One agent per published script.** One session owns running the scripts, because every worktree
writes to the main checkout's `data/studies/` and a second re-run would overwrite its evidence. Each
script gets a fresh Opus review before its first run, and the PR body names the review.

## 8. Compute estimate

The saved AIFS blend fits took about 8 to 10 seconds per (arm, site) fit-set (five folds times three
seeds) on the GPU, on 6,000 to 14,000 rows. The README of `nwp_forecast_comparison` records that the
315 fit-sets of the product-blends batch take about 1 hour with 2 workers. This study has about 4
times the rows (wind about 50,000 per farm, solar about 26,000 daytime rows per farm), so plan
**15 to 40 seconds per fit-set**.

| Block | Fit-sets |
|---|---|
| Wind: 8 arms at 2 settings (as-available, matched, 4 shuffled) times 3 farms | 48 |
| Wind: exploratory (3 by 3 block, hour-ending pair, previous run) and 4 planted, 1 setting | 27 |
| Solar: 7 arms at 2 settings times 6 farms | 84 |
| Solar: 4 planted and 2 wind-speed arms, 1 setting | 36 |
| **Total** | **about 195** |

At 15 to 40 seconds each, that is **about 1 to 2.5 hours of fitting**, less with 2 workers. The
#1016 reports recorded no build time, so `build_inputs.py --dry-run` on one month is multiplied by 83
for the build. Set A needs seconds. The ERA5 wind fetch for station cells is about 1.5 hours of
Climate Data Store time, queued one request at a time.

## 9. The page, cross-references, and docs

**The page:** `docs/studies/past-weather/ukv-ceda-vs-era5.md`, under Studies > Past weather in
`mkdocs.yml` after the blending page. The question is about hours already past. It follows the
`study` skill's twelve sections. **Headline figure (Figure 1):** the five planned contrasts with 95%
intervals, grouped by variable and window, with each product's absolute error in the row label, the
margin drawn as a band, and the second hyperparameter setting as a marker. Results sections, each
with a chart: "The XGBoost models work" (out-of-fold power against measured for A to F and W1 to W3
across weeks chosen by a stated rule); the station scores by year and season; the early-years test;
the wind arms by height; the solar temperature contrast against its detectable difference; the
controls. Where the decision table gives "ERA5 by default", the page says so.

**Cross-references in `docs/studies/`:** the studies index, the past-weather index, a row-set entry
and the five planned contrasts in `past-weather/methods.md`, and a row in `studies/README.md`. The
study skill, `studies/ukv_vs_era5/README.md`, and `packages/studies/README.md` change as needed.

**Edits outside the study confinement.** `docs/roadmap/training-history.md` (lines saying the check
"is not yet scheduled"), `docs/roadmap/data-sources.md` (the CEDA UKV row's 🚧 status), and the
licence cell of `docs/background/weather-products-survey.md` change when the result lands. **A PR
touching those files stops at a reviewed PR, so the plan puts them in a separate small PR and keeps
the study PR confined to `studies/`, `packages/studies/`, `docs/studies/`, and `mkdocs.yml`.**

## 10. Risks and open questions, with recommendations

1. **Licence (verdict).** I fetched the CEDA catalogue record for NWP-UKV once on 2026-10-05
   (uuid `f47bc627...`, the record the fetch scripts cite). It says access is for "any registered
   CEDA user" and that use is covered by Creative Commons Attribution-NonCommercial-ShareAlike 4.0,
   with a required citation. The download notes on disk agree. **Verdict: the study is non-commercial
   research and may run. It may publish error scores and charts that carry no UKV values, with the
   citation "Met Office (2016): NWP-UKV ... CEDA".** The stores, frames, and predictions stay in the
   private data store. **Whether the main work's use of UKV-CEDA as training history is
   "non-commercial" is not settled by this study. The maintainer decides it, and every UKV-CEDA
   recommendation says so.** ShareAlike may reach adapted material, and the repository is MIT
   licensed. Published scores are aggregate statistics and probably not adapted material. Question:
   does the maintainer want that read confirmed with CEDA? The survey cites a second CEDA record
   (uuid `78f23c53...`) that states no terms, so the survey should cite the record that does.
2. **Four stations are few.** Recommendation: do not widen the UKV crop in this study. The
   month-resampled interval is the honest measure, and a wider crop is a multi-day CEDA download.
   If P1 and P2 intervals are too wide to read, a follow-up decides whether to widen it.
3. **CEDA gaps.** 3.3% of runs are missing or partial. Recommendation: drop, never fill from an
   older run, and report the loss per month.
4. **No 100 m wind in UKV-CEDA.** Recommendation: the matched 10 m arms separate product from height,
   and the page reports both contrasts.
5. **ERA5 wind: 3 by 3 block or nearest cell.** Recommendation: nearest cell as primary, matching
   the wind page, and the block mean as an exploratory arm.
6. **CEDA UKV differs from live UKV** (the repo's note). The study cannot say how live UKV would do,
   and the page says so.
7. **Multiple comparisons.** Five planned contrasts at 95% each, no correction. The rule needs two
   to agree. Every year, season, station, and arm row is exploratory, and the page says that about
   one row in 20 with no real effect reaches the 5% level by chance.
8. **Power data in 2019 and 2020.** The solar record starts in mid-September 2019, and one solar
   site spends its first 8 months in commissioning (cut by `drop_commissioning_ramp`). Start dates
   for the three wind farms are not documented in the pages read. The coverage check measures
   scored hours per generator-year. If the early window has fewer than 6 months at fewer than 2
   generators, set B makes no early-window reading and set A alone speaks for 2019 and 2020.
9. **Duplicate store-reading code.** The new `ukv_ceda_stores.py` repeats logic in
   `build_ukv_ceda_inputs.py`. Recommendation: leave it, and file a later issue to share one module
   with a bit-for-bit proof on the blends outputs.
10. **PS44.** Unknown date and content. The step check is the backstop (section 4).
11. **Which margin.** The 5% set A margin is a judgement, taken from the wind page's 6%. The
    detectable-difference margin for set B follows the design. The maintainer may prefer other
    values, which must be set before any result.
12. **Does `mkdocs.yml` count as confined?** Earlier study PRs edited its nav. Recommendation: treat
    the nav line as part of the study docs.

## 11. Docs to update and verification

Docs: section 9. Every edit describes the code as it is, with no history.

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/README.md
uv run mkdocs build --strict      # and read the rendered page and its figures
uv run python studies/ukv_vs_era5/check_coverage.py     # must exit 0 before any build
uv run python studies/ukv_vs_era5/build_inputs.py --dry-run
uv run python studies/ukv_vs_era5/fit_arms.py --dry-run
```

Also run the CI steps `pydoclint` and the docs-link checker locally, and the mutation pass whenever
`packages/studies/` changes. Run `studies.*` tests for the two new modules, each with a test that
fails on the bug it exists to catch: a slot mapped to the wrong store, a missing run served by an
older run, a bias removed across months instead of within one, and unequal capacities in the
normalisation.

## 12. What this plan does not commit the project to

The plan commits only to this study. It does not commit to widening the UKV crop, to ingesting
UKV-CEDA in production, or to deleting the duplicate reader.
