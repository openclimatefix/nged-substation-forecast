# Plan: CEDA UKV or ERA5 as the estimate of past wind and temperature, 2019 to 2026 (issue #1024)

**The question is whether the main Flexpectation work should take past wind speed and past air
temperature from the Met Office's UKV as archived by CEDA (UKV-CEDA) or from ERA5.** The main work
already takes past irradiance from CAMS, and this study does not reopen that choice. No published
page compares UKV with ERA5 before August 2024, so nothing says whether UKV's lead over ERA5 on wind
power (0.44 points of capacity [0.24, 0.63], August 2024 to September 2026) holds in 2019 and 2020.
The CEDA stores start in September 2019, so the study can test those years. The recommendation covers
wind speed and temperature only, because the issue asks to start with those two variables. The other
variables in `conf/model/xgboost.yaml` (dew point, sea-level pressure, surface pressure, 500 hPa
height, precipitation type, windchill) are not scored, and the page says so.

**The plan runs two experiment sets and fixes the rule that combines them before any result
exists.** Set A scores each product against Met Office station observations with no machine
learning. Set B fits one XGBoost model per metered generator and per product, and scores the model's
power error. Four planned contrasts answer the question, and the rule defaults to ERA5 unless
UKV-CEDA clears a stated margin. Every score is split by year, by season, and into an early window
(September 2019 to December 2020) and a late window.

## 1. Verdict, size and departures

**Verdict: worth doing, with four departures from the issue body.**

- **Set A has four stations, not 18.** The CEDA stores are cropped to the private trial-area box.
  Counting stations against that box (counts only, no bounds printed) finds 4 of the 30 hourly
  MIDAS stations inside it. All 4 report wind speed and temperature on 98% or more of hours since
  September 2019. The 14 other stations that report wind lie outside the crop, and reading them
  needs a wider UKV download that this study does not do.
- **ERA5 wind at the stations covers 2019-09 to 2023-12 for all 4, and to 2025-12 for 3 of them.**
  `ERA5-WIND-2019-2023/` holds a 3 by 3 block of cells around each of the 18 wind-reporting
  stations. For 3 of the 4 in-crop stations, the nearest ERA5 cell also lies inside the wind-farm
  blocks of `ERA5/wind_native_cds.parquet`, which runs from 2024-01-01 to 2026-09-20. Set A
  therefore scores wind for those 3 stations over 2019-09 to 2025-12, and scores the fourth station
  over 2019-09 to 2023-12 only. The plan adds no download. ERA5 temperature
  (`beam_diffuse_open_meteo.parquet`, 20 cells, 2019-09 to 2026-09) covers all 4 stations over the
  whole window.
- **2019 holds three scored months.** The Met Office's Parallel Suite 43 (PS43) took effect on
  2019-12-04, so the pre-PS43 era is September to November 2019. A calendar year with fewer than
  `MIN_MONTHS_FOR_INTERVAL` (6) months gets a point estimate and no interval. The early window
  therefore pools 2019 and 2020 into one window of 15 months.
- **Solar differs only in temperature.** The solar models read CAMS irradiance and, in the existing
  studies, ERA5 air temperature. The solar contrast therefore tests temperature alone.

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
persona and prose reviews. The simplicity review has run, and "Considered and rejected" records what
it changed. The PR stays a draft until the maintainer approves.

## 2. The arms and the variables

**Set A scores each product's hourly value against the station's reading.** The two products are
ERA5 (nearest 0.25 degree cell) and UKV-CEDA (nearest 2 km cell, at most 2 km from the station).
Variables: 10 m wind speed (primary), and 2 m (ERA5) or 1.5 m (UKV) air temperature (primary). The
primary score is the mean absolute error after removing each (station, product, calendar month)
mean error. This matches what the XGBoost models in set B do, because they recalibrate per
generator, and it removes offsets from station exposure and cell orography. Bias removal does not
remove the advantage a 2 km cell has over a 0.25 degree cell when scored against a point
observation, so the page does not attribute a UKV lead to the product alone. The raw mean absolute
error is printed beside the bias-removed score. The page prints each of the four stations'
contrasts as exploratory rows, so a reader can see whether one station drives the pooled result.

**Which UKV lead stands in for past weather.** The primary reads the freshest archived run for each
hour: the run at 00, 06, 12, or 18 UTC at or before the hour, so leads are 0 to 5 hours (mean 2.5).
This is the lowest lead the 6-hourly store can give, and the closest match to the T+0 analysis that
the wind page scored from Open-Meteo. Because lead equals hour of day modulo 6, the XGBoost models
can learn lead-specific bias from `hour_of_day`. The page states this as part of what a consumer
gets.

**Which stations.** The 4 hourly stations inside the UKV crop. `studies.midas.select_nearest_stations`
returns the mapping to the script, and the page prints only pooled distance ranges, because a
station distance to a generator helps locate the generator. A station farther than 40 km from every
metered generator is context for set A and is not read as evidence about the power scores.

**Set B wind arms (3 wind farms, W1 to W3).** Each arm has 7 columns, built by one function per
arm type and printed into the report:

- Shared (3): `hour_of_day`, `day_of_year`, `era_code`.
- `era5_wind` (as available, 4): 100 m speed, sine and cosine of 100 m direction, 10 m speed.
- `ukv_ceda_wind` (as available, 4): 10 m speed, sine and cosine of 10 m direction, 925 hPa speed.

UKV-CEDA has no 100 m wind, so the as-available contrast mixes product and height, and the page
does not attribute the contrast to either. The 1000 hPa level is excluded because the files hold
NaN wherever the level is below ground, and a rule on NaN would filter by one product's values.
ERA5 reads the nearest cell (the centre of each 3 by 3 block already on disk). Power is the hour
centred on the label (stamps shifted back 30 minutes before `studies.power.hourly_from_half_hourly`),
because both products' wind is instantaneous. An `hour_ending` pair at the primary setting scans the
offset for both products, as the study skill requires for every product.

**Set B solar arms (6 solar farms, A to F).** Each arm has 10 columns: shared (7: `solar_zenith_deg`,
`solar_azimuth_deg`, `extraterrestrial_horizontal_w_m2`, `hour_of_day`, `day_of_year`, `era_code`,
and the arm's temperature), plus CAMS global, beam, and diffuse irradiance (3). The two arms differ
only in the temperature column: `era5_temp` and `ukv_ceda_temp`. Temperature is the mean of the
instants at the hour's two ends for both products (the power hour ends at the label).

## 3. Decision rule, fixed before any result

**Margins.** A contrast "clearly" favours a product only when its 95% interval lies wholly on one
side of zero and the point estimate is beyond a margin. The margin for set A is 5% of ERA5's error
on the same rows (the wind page's 0.44 points was 6% of ERA5's error, so the margin sits just
below the one effect already seen). The margin for set B is the smallest difference the design
detects, taken from published intervals. The detectable difference is 2.8 times the standard error
of the paired month-resampled difference (80% power, 5% level, two sided). Half-widths divided by
1.96 give standard errors of about 0.040 points for solar at 21 months (the day-1 blend contrast on
the UKV-CEDA blends page) and about 0.10 points for wind at 25 months (the Open-Meteo UKV against
ERA5 contrast). Standard error scales with the square root of the number of months, and set B has
about 83 scored months. **The margins are frozen now: 0.06 points of capacity for solar and 0.16
points of capacity for wind.** A statistically significant difference smaller than the margin reads
"small, not clear". The real solar contrast changes only temperature and may be smaller than 0.06
points.

**Planned contrasts (four, labelled "(planned)").** Each is UKV-CEDA minus ERA5, with a negative
value favouring UKV-CEDA.

1. **P1, set A:** 10 m wind speed, bias-removed mean absolute error against the stations. Context
   only: P1 never decides.
2. **P2, set A:** air temperature, the same score and rows. P2 decides temperature.
3. **P3, set B wind:** the as-available arms, error as percent of capacity, at both hyperparameter
   settings. P3 decides wind.
4. **P4, set B solar:** `ukv_ceda_temp` minus `era5_temp`, at both settings. P4 vetoes a UKV-CEDA
   temperature recommendation.

The four are not corrected for multiple comparisons. Each is read at its own interval and margin,
and the rule below names the one contrast that decides each variable.

**The rule, by variable.**

- **Wind speed: P3 decides.** Set B measures what the main work does with wind, which is to predict
  power after per-generator recalibration. P3 must agree at both hyperparameter settings. P1 is
  printed beside P3, and where P1 and P3 disagree the page says so and P3 still decides.
- **Temperature: P2 decides.** The generators are not demand, so set B cannot say how temperature
  helps a demand forecast. If P4 clearly favours ERA5 at either setting, the study does not
  recommend UKV-CEDA temperature for solar models.

| Deciding contrast reads | Study reports |
|---|---|
| UKV-CEDA clearly better | Recommend UKV-CEDA, subject to the early-years test and the licence |
| ERA5 clearly better | Recommend ERA5 |
| No clear difference, or small | Recommend ERA5 (default: no licence limit, no run gaps, one physics version) |

**Early-years test.** UKV-CEDA is recommended for training history from 2019 only if the deciding
contrast's early-window point estimate is negative and its upper 95% bound is below zero plus the
margin. Otherwise the study recommends UKV-CEDA from 2021 only, and says that an era-mixed design
is untested. The difference between the early and late windows of the deciding contrast is printed
as an exploratory row.

**A conditional on the licence.** Every UKV-CEDA recommendation carries the licence verdict in
section 10.

## 4. Row set, gap rule, and era rule

**Rows are the intersection across every arm, decided from the target and availability, never from
a product's values.** An hour stays only if the target exists and every arm's input exists. Set B
also drops the hours the other wind and solar studies drop: wind hours holding an exactly zero
half-hour (`wind_product_frames.common_rows`), and for solar the commissioning ramp and the
export-cap and active-network-management hours (`solar_product_frames.common_rows`). Set A keeps
a station-hour if the station has a reading and both products have a value. All 3 wind farms have
power from 2019-09-17, with roughly 10,900 to 11,200 hours before 2021, so set B can read the early
window for wind.

**Gap rule for CEDA runs.** Across the three stores the counts are 9,981 complete, 15 partial, and 82
missing runs, plus 260 runs on 65 days that CEDA has no directory for. That is 10,338 runs, of which
357 (3.45%) are missing, unlisted, or partial. An hour whose freshest run is missing or partial is
dropped for every arm. It is not served by an older run, because that changes the lead mid-series.
The build's `--check-only` prints the share of hours lost per month, re-checks the 65 days against
CEDA and the 15 partial runs, and stops if one month loses more than 25% of hours.

**Eras.** UKV-CEDA has three eras. Era 0 is runs before 2019-12-04 (PS43, RAL2-M), era 1 is 2020-01
to 2025-12, and era 2 is 2026-02 onward (PS47, 2026-01-21, RAL3). The straddling months, 2019-12 and
2026-01, are dropped for every arm in both sets. PS45 (May 2022) and PS46 (May 2025) are not era
boundaries: the repo's NWP-upgrades table finds no UKV physics change at either. **The date and
content of PS44 are unknown in the repo.** Before the first build, the Met Office release notes are
checked for PS44. As a data check, the verify script plots the monthly UKV-minus-ERA5 mean and
UKV-minus-station mean for wind and temperature, to find any step the table misses. A step found
becomes an era boundary and a deviation from this plan, recorded on the page. Every arm gets
`era_code` (0, 1, 2). Folds are cut within each era.

## 5. Folds, intervals, settings, metric, hardware

- **Folds:** `studies.cross_validation.cut_eras` with `first_months=("2020-01", UKV_UPGRADE_MONTH)`,
  five blocks of whole months per (site, era). Era 0 has 3 months, so `assign_folds` gives it folds
  0, 1, and 3, and two of its five folds (2 and 4) are empty before rotation.
  `cerra_past_solar.with_covering_folds` picks the rotation, and `raise_on_uncovered_months` stops
  the build if any calendar month is held out of every training row.
- **Intervals, set B:** `studies.bootstrap` resamples whole calendar months, paired across arms,
  with one of the three fitting seeds, 2,000 resamples. **Set A:** `bootstrap_row_difference`
  resamples whole months and has no seed. Both cover month-to-month weather, not differences between
  stations or generators. Neighbouring generators and the 4 stations share their weather.
- **Hyperparameters:** `PRIMARY_HYPER_PARAMETERS` and `SENSITIVITY_HYPER_PARAMETERS` on P3 and P4.
  A verdict needs both settings to agree, and the page says where a contrast changes sign or
  significance. Controls and the hour-ending pair run at the primary setting only.
- **Metric:** mean absolute error as percent of capacity for set B, each row divided by its own
  generator's `effective_capacity` before any mean. Set A reports metres per second and kelvin.
  Every arm's absolute error is reported, not only the contrasts.
- **Splits:** each calendar year with at least 6 months (point estimate only below that), each
  half-year (October to March, April to September), and the early and late windows. All are
  exploratory except the early and late reading of the deciding contrast.
- **Hardware:** `device="cuda"` after `nvidia-smi` and `uptime`. The page states the device, and one
  arm is refitted on the CPU for the noise floor. `colsample_bytree` stays at 1 and folds stay cut
  within eras.

## 6. Controls

- **Negative control, set B:** each product's weather columns shuffled within (site, year-month,
  hour of day), at one shuffle seed and the primary setting. The shuffled-UKV minus shuffled-ERA5
  contrast shows the difference the pipeline produces from nothing, and its own month-resampled
  interval is its spread. Run for wind and solar. A product contrast smaller than the control's
  spread is not read. Reading `era5_temp` minus its shuffled arm also shows whether temperature
  adds anything, so the plan has no separate no-temperature arm.
- **No positive control.** The study skill accepts an interval that bounds the effect instead.
  Every null reading of P1 to P4 therefore carries its bound, as in "an effect as large as 0.1
  points is not excluded".
- **Set A:** a lag scan of each product against the station at -3 to +3 hours must put the minimum
  error at lag 0. The scan checks the timestamp convention. Station wind covers the 10 minutes
  ending 10 minutes before the label, and both products are instantaneous at the label.

## 7. Scripts, files, and outputs

**The scripts go in `studies/past_weather/`, named `ukv_ceda_*`, because the folder holds the
scripts of one family of pages and can import the fit and report helpers its siblings already
use.** The new scripts import `cerra_past_solar.with_covering_folds` and `check_column_counts`, the
`ens_past_solar` report and fingerprint helpers, and `station_wind_arms.py`'s station reading and
timestamp handling, and model the fit script on `reanalysis_past_wind.py` (new product against
ERA5, covering folds, both settings, `--check-only`, `--report-only`).

| Script | Job |
|---|---|
| `ukv_ceda_vs_era5_build.py` | Constants (windows, margins, one arm-columns function per arm type); builds the set A station-hours and the set B wind and solar frames. `--check-only` runs the coverage check below and exits non-zero if a guard fails. `--dry-run` builds one month and writes nothing |
| `ukv_ceda_vs_era5_verify.py` | Column counts, lag scan, monthly step plots, lead distribution |
| `ukv_ceda_station_scores.py` | Set A scores and intervals, no fit. Holds the bias-removal function |
| `ukv_ceda_vs_era5_fit.py` | Set B fits. `--dry-run` lists fits, `--verified` runs after verify, `--report-only` reads the saved losses and set A's interval tables, applies section 3 by code, and writes `report.md` and `decision.md` |
| `ukv_ceda_vs_era5_charts.py` | Figures. Every shared number is read from the reports |

**Coverage check (build `--check-only`).** It prints, with no value-based filter: runs by status per
store, hours lost per month, ERA5 coverage per product and per station, station coverage per year,
the pooled station distance range, power hours per generator-year (a generator-year below 2,000
scored hours is not shown separately), scored months per era and per window, and
`search_fold_offsets`. No build runs until the check exits 0.

**New code in `packages/studies/`, with tests.**

- `ukv_ceda_stores.py`: map an hour to its freshest 6-hourly run and to the store holding that run,
  read that run's status, and read leads 0 to 5 at given cells. The module takes a `Profile` from
  `studies.ukv_ceda_profiles` and takes the product names, slot epoch, cycle, and `STATUS_*` codes
  from there, never re-declaring them. `build_ukv_ceda_inputs.py` has similar readers tied to the
  120-hour profile and stays untouched, so no published output needs a bit-for-bit proof. Because
  the module takes a `Profile`, removing the duplicate later is a pure rewiring.
- `sources.py` gains `UKV_CEDA_PRODUCT_DIR`, `UKV_CEDA_PART2_PRODUCT_DIR`,
  `UKV_CEDA_PART3_PRODUCT_DIR`, and `UKV_VS_ERA5_DIR = study_dir_for(study="ukv_vs_era5")`.
- The bias-removal function stays in `ukv_ceda_station_scores.py` and is tested under
  `packages/studies/tests/past_weather/`. It moves into the package when a second study needs it.

**Data paths.** Scripts name data only by product and constant (`ERA5_PRODUCT_DIR`,
`ERA5_WIND_2019_2023_PRODUCT_DIR`, `CAMS_PRODUCT_DIR`, `MIDAS_OPEN_PRODUCT_DIR`, the three UKV-CEDA
constants). Issue #861 ("Give each study its own folder under studies/, and move shared study
plumbing into packages/studies") plans to move the downloads from `data/studies/weather/<PRODUCT>`
to a `data/studies/downloads/` layer, and `sources.py` already defines `DOWNLOADS_DIR`, equal to
`STUDIES_DATA_DIR` until the data moves. A script that reads only these constants runs before and
after the move, so no script waits for it. The plan adds no download script.

**Output.** `UKV_VS_ERA5_DIR` is write-once: `refuse_to_overwrite` stops a second run, and a result
a merged page quotes moves to `superseded/` first. The folder holds the frames and build stamp, each
arm's per-row losses and out-of-fold predictions (site, time, fold, seed, arm, setting, actual
value), the interval tables as parquet, and `report.md` and `decision.md`. Stamps record the arm
columns, the XGBoost version, the device, and the hashes of the inputs.

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
| Wind planned arms (2 arms, 2 settings, 3 farms) | 12 |
| Wind shuffled controls (2 products, 1 setting, 3 farms) | 6 |
| Wind hour-ending pair (2 products, 1 setting, 3 farms) | 6 |
| Solar planned arms (2 arms, 2 settings, 6 farms) | 24 |
| Solar shuffled controls (2 products, 1 setting, 6 farms) | 12 |
| **Total** | **60**, plus one CPU refit |

At 15 to 40 seconds each, that is **15 to 40 minutes of fitting**, less with 2 workers. The #1016
reports recorded no build time, so `--dry-run` on one month is multiplied by 83 for the build. Set A
needs seconds.

## 9. The page, cross-references, and docs

**The page:** `docs/studies/past-weather/ukv-ceda-vs-era5.md`, under Studies > Past weather in
`mkdocs.yml` after the blending page. It follows the `study` skill's twelve sections. **Headline
figure (Figure 1):** the four planned contrasts with 95% intervals, grouped by variable and window,
with each product's absolute error in the row label, the margin drawn as a band, and the second
hyperparameter setting as a marker. Results sections, each with a chart: "The XGBoost models work"
(out-of-fold power against measured for A to F and W1 to W3 across weeks chosen by a stated rule);
the station scores by year and season, with the four per-station rows; the early-years test; the
solar temperature contrast against its detectable difference; the controls, in one section. Where
the decision table gives "ERA5 by default", the page says so.

**Cross-references in `docs/studies/`:** the studies index, the past-weather index, a row-set entry
and the four planned contrasts in `past-weather/methods.md`, and a row in `studies/README.md`.
`studies/past_weather/README.md` maps the five scripts to the page. The study skill and
`packages/studies/README.md` change as needed.

**Edits outside the study confinement.** `docs/roadmap/training-history.md` (lines saying the check
"is not yet scheduled"), `docs/roadmap/data-sources.md` (the CEDA UKV row's status), and the
licence cell of `docs/background/weather-products-survey.md` change when the result lands. **A PR
touching those files stops at a reviewed PR, so the plan puts them in a separate small PR and keeps
the study PR confined to `studies/`, `packages/studies/`, `docs/studies/`, and `mkdocs.yml`.**

## 10. Risks and open questions, with recommendations

1. **Licence (verdict).** The CEDA catalogue record for NWP-UKV (uuid `f47bc627...`, the record the
   fetch scripts cite), fetched once on 2026-10-05, says access is for "any registered CEDA user" and
   that use is covered by Creative Commons Attribution-NonCommercial-ShareAlike 4.0, with a required
   citation. The download notes on disk agree. **Verdict: the study is non-commercial research and
   may run. It may publish error scores and charts that carry no UKV values, with the citation "Met
   Office (2016): NWP-UKV ... CEDA".** The stores, frames, and predictions stay in the private data
   store. **Whether the main work's use of UKV-CEDA as training history is "non-commercial" is not
   settled by this study. The maintainer decides it, and every UKV-CEDA recommendation says so.**
   ShareAlike may reach adapted material, and the repository is MIT licensed. Published scores are
   aggregate statistics and probably not adapted material. Question: does the maintainer want that
   read confirmed with CEDA? The survey cites a second CEDA record (uuid `78f23c53...`) that states
   no terms, so the survey should cite the record that does.
2. **Four stations are few.** Each month holds about 4 times 720 station-hours, so the pooled
   interval will probably be narrow, but the interval covers month-to-month weather only. It covers
   neither differences between stations nor places outside one box, and a 2 km cell favours a point
   observation. Recommendation: do not widen the UKV crop in this study, print the per-station
   rows, and let a follow-up decide whether to widen the crop.
3. **CEDA gaps.** 3.45% of runs are missing, unlisted, or partial. Recommendation: drop, never fill
   from an older run, and report the loss per month.
4. **No 100 m wind in UKV-CEDA.** The as-available wind contrast mixes product and height.
   Recommendation: the page reports the contrast as what a consumer of each product gets and claims
   no cause.
5. **CEDA UKV differs from live UKV** (the repo's note). The study cannot say how live UKV would do,
   and the page says so.
6. **Multiple comparisons.** Four planned contrasts at 95% each, no correction, one deciding
   contrast per variable. Every year, season, station, and arm row is exploratory, and the page says
   that about one row in 20 with no real effect reaches the 5% level by chance.
7. **Early-window power in solar.** The solar record starts in mid-September 2019, and one solar
   site spends its first 8 months in commissioning (cut by `drop_commissioning_ramp`). The build's
   `--check-only` measures scored hours per generator-year. If the solar early window has fewer
   than 6 months at fewer than 2 generators, set B makes no early-window solar reading. Set A alone
   decides temperature, so the early-years test still has a reading.
8. **Duplicate store-reading code.** `ukv_ceda_stores.py` repeats logic in
   `build_ukv_ceda_inputs.py`. Recommendation: leave it, and file a later issue to share one module
   with a bit-for-bit proof on the blends outputs.
9. **PS44.** Unknown date and content. The step check is the backstop (section 4).
10. **Which margin.** The 5% set A margin is a judgement, taken from the wind page's 6%. The set B
    margins follow the design. The maintainer may prefer other values, which must be set before any
    result.
11. **Does `mkdocs.yml` count as confined?** Earlier study PRs edited its nav. Recommendation: treat
    the nav line as part of the study docs.

## 11. Docs to update and verification

Docs: section 9. Every edit describes the code as it is, with no history.

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/README.md
uv run mkdocs build --strict      # and read the rendered page and its figures
uv run python studies/past_weather/ukv_ceda_vs_era5_build.py --check-only   # must exit 0
uv run python studies/past_weather/ukv_ceda_vs_era5_build.py --dry-run
uv run python studies/past_weather/ukv_ceda_vs_era5_fit.py --dry-run
```

Also run the CI steps `pydoclint` and the docs-link checker locally, and the mutation pass whenever
`packages/studies/` changes. Each new test fails on the bug it exists to catch: a slot mapped to the
wrong store, a missing run served by an older run, a bias removed across months instead of within
one, and unequal capacities in the normalisation.

## 12. What this plan does not commit the project to

The plan commits only to this study. It does not commit to widening the UKV crop, to ingesting
UKV-CEDA in production, or to deleting the duplicate reader.

## Considered and rejected

**Changed after the simplicity review.** The design shrank from about 195 fit-sets to 60. The plan
dropped the matched 10 m arms, the no-temperature arm, the planted-target positive controls, the 3
by 3 block arm (issue #1025 owns it), the previous-run arm and lead profile, the solar 10 m wind
arms, the dew-point and pressure fetch and rows, wind direction error, the UKV-averaged-to-ERA5
arm, the irradiance context row, and the second shuffle seed and second setting on the controls.
The decision rule now names one deciding contrast per variable, P5 is an exploratory row, the
scripts moved from nine in a new folder to five in `studies/past_weather/`, and the ERA5 wind
2024 to 2025 request is gone.

**Rejected, with one-line reasons.**

- Starting at 2020-01 and dropping era 0: the issue asks for 2019, and the three months cost little.
- Dropping the hour-ending pair: the study skill scans the power-hour offset for every product.
- Dropping the second hyperparameter setting on P3 and P4: the study skill requires it for planned
  contrasts.
- Putting `station_scoring.py` in the package: only this study uses it, so it stays in the script.
- Moving the store readers into `ukv_ceda_stores.py` and rewiring `build_ukv_ceda_inputs.py`: that
  needs a bit-for-bit proof on published outputs, so the duplication stays.
- Fetching the missing ERA5 wind cell for the fourth station: the plan scores that station on the
  covered window instead.
- Gating the scripts on the `data/studies/downloads/` move: the constants in `sources.py` already
  hide the layout.
