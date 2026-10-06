# Plan: UKV from CEDA or from Open-Meteo for temperature, wind power, and solar power (issue #1051)

**The question is whether the two archives of the Met Office's UKV that the project can read give
the same temperature, wind power, and solar power forecasts, and what that means for training
history.** CEDA's UKV archive reaches back to September 2019 and is the only UKV history we hold
for 2019 to 2024. The UKV that Open-Meteo serves matches the Met Office's live feed (its irradiance
agrees with the Met Office's own files to within 0.55 W m⁻² for hours since 12 August 2024), so it is
the lineage a live service would read. The CEDA download notes record that CEDA's archive differs
statistically from the live feed, and that a model trained on one should not be used on the other.
Nothing yet measures that difference in temperature, wind power, or solar power.

**The plan compares the two archives on the 23 whole months they share, in four ways.** A model-free
comparison measures how far apart the values are. A station comparison scores each archive against
the Met Office's station readings. A power comparison fits an XGBoost model per generator on each
archive. A transfer comparison trains an XGBoost model on CEDA's values and scores it on
Open-Meteo's values, which is the train-on-history, serve-live case. Five planned contrasts
(section 3) answer the question. Each has a margin frozen in this plan, and the rule that turns the
contrasts into a recommendation defaults to the conservative reading when a contrast is unresolved.

## 1. Verdict, size and departures

**Verdict: worth doing, with five departures from the issue body.**

- **The overlap is 23 whole months, not "from 12 August 2024 to September 2026".** The kept months
  are 2024-09 to 2024-12, 2025-01 to 2025-12, and 2026-02 to 2026-08. The plan drops 2024-08 and
  2026-09 because each holds fewer than 25 days of Open-Meteo's wind extract, and drops 2026-01
  because UKV's Parallel Suite 47 (PS47) upgrade of 2026-01-21 falls inside it.
- **Only one era boundary falls inside the overlap.** Question 3 can be answered for PS47 only (era 0
  is 2024-09 to 2025-12 with 16 months, era 1 is 2026-02 to 2026-08 with 7). Parallel Suite 43
  (2019-12-04) and PS44 (date unknown) precede Open-Meteo's UKV, and no overlap exists in which to
  test them. The page says so.
- **Open-Meteo's UKV is on disk at the generators and not at the stations.**
  `previous_runs/combined.parquet` holds temperature, 10 m and 100 m wind, and irradiance at all
  nine generators from 2024-01-01, and its wind and global irradiance equal the `site_points/`
  extracts bit for bit on every shared row (checked for this plan, 54,438 and 141,696 rows). The
  station comparison needs Open-Meteo's UKV at the four stations inside the CEDA crop, so the plan
  adds one small fetch (section 7), about 4 stations by 16 months by 3 variables and far under the
  30 GBP limit. No other download is planned.
- **Beam and diffuse irradiance cannot enter any arm,** because CEDA's files hold one downward
  shortwave field. The archives share temperature, 10 m wind, and global horizontal irradiance.
- **Wind has a matched 10 m pair as well as the as-available pair,** because CEDA has no 100 m wind
  and a model cannot be scored on columns it was not trained on. The matched pair also carries the
  transfer contrast (P4), the one measurement of the issue's question about training history.

**Size: complex.** All five triggers from `plan-issue` step 3 are answered.

| Trigger | Answer |
|---|---|
| Changes what gets stored | Yes. A published page, a write-once results folder, a fetched station file, and changes to `packages/studies` |
| Touches the production serving path | No. Production never imports `studies` |
| Touches a degradation rule | No |
| More than one defensible design | Yes. The lead each contrast reads, how to match wind heights, the margins, the transfer design, and the station score |
| Callers not nameable without searching | Yes. `cross_validation.out_of_fold_losses` and `ukv_ceda_stores` serve several studies |

**Reviews: all four, plus the study skill's.** The study coordinator runs the two plan reviews. The
diff reviews include the mutation pass, because `packages/studies/` changes. The study skill adds a
fresh Opus review of each script before its first run, two Opus scientific-validity reviews, and the
persona and prose reviews. The PR stays a draft until the plan reviews are triaged.

## 2. Arms, variables, and reading each archive

**Which UKV lead stands in for each archive's "past weather".** Open-Meteo serves, for each hour,
the freshest run's analysis (lead 0; `verify_ukv_lineage.py` checked this for hours since
2024-08-12). CEDA's 6-hourly archive is read at each hour from the freshest run at or before the
hour (`studies.ukv_ceda_stores`), so its lead is 0 to 5 hours (hour of day modulo 6). **The two
archives can agree exactly only at the four lead-0 hours of the day (00, 06, 12, and 18 UTC).**
The planned contrasts read all hours, which is what a consumer of each archive gets, and the plan
adds a lead-0-only read of every contrast, so the page can separate a product difference from a
lead difference. The lead-0 read refits on the lead-0 rows (`restricted_frame`).

**Model-free comparison (question 1, no contrast, no margin).** For temperature, 10 m wind speed,
and global horizontal irradiance, at the nine generator sites and the four stations, the script
reports the mean difference, the mean absolute difference, the 99th percentile of the absolute
difference, and the correlation, by CEDA lead, hour of day, calendar month, and era, with intervals
that resample whole months (`bootstrap_row_difference`). Three diagnostics run first and gate the
rest:

- **Cell match.** At lead-0 hours, which of the nine CEDA cells around the nearest has the value
  closest to Open-Meteo's? If the nearest cell does not win in most hours, the archives read
  different cells, and the plan reads the CEDA cell the diagnostic picks.
- **Elevation.** Open-Meteo may lapse-rate-adjust 2 m temperature to the point's elevation, which
  shows as a near-constant offset at lead 0. Every XGBoost model is fitted per generator and every
  station score removes a per-station mean error, so a constant offset cannot move a planned
  contrast. The page reports the offsets.
- **Irradiance construction.** Open-Meteo builds its hourly irradiance from UKV's snapshot at the
  hour's end. The check compares that value with CEDA's snapshot at the label at lead-0 hours. The
  comparison also scores irradiance back to 2022-03-01, Open-Meteo's backfill from an unnamed
  source, as an exploratory row.

**Set B wind arms (three wind farms, W1 to W3).** Every arm has the shared columns `hour_of_day`,
`day_of_year`, and `era_code`, plus weather columns, built by one function per arm type that returns
a fixed-length tuple, printed into the report:

| Arm | Weather columns (all 4) | Total |
|---|---|---|
| `ceda_wind` (as available) | 10 m speed, sine and cosine of 10 m direction, 925 hPa speed | 7 |
| `om_wind` (as available) | 100 m speed, sine and cosine of 100 m direction, 10 m speed | 7 |
| `ceda_wind_10m` (matched) | 10 m speed, sine and cosine of 10 m direction | 6 |
| `om_wind_10m` (matched) | the same three columns from Open-Meteo | 6 |

CEDA's 1000 hPa wind is excluded because the files hold NaN where the level is below ground, and a
rule on NaN would filter by one product's values. Power is the hour centred on the label (stamps
shifted back 30 minutes), because both archives' wind is instantaneous. An hour-ending pair of the
as-available arms at the primary setting scans the offset for both archives.

**Set B solar arms (six solar farms, A to F).** The shared columns are `solar_zenith_deg`,
`solar_azimuth_deg`, `extraterrestrial_horizontal_w_m2`, `hour_of_day`, `day_of_year`, and
`era_code` (6). For both archives, temperature is the mean of the instants at the hour's two ends,
and global irradiance is the snapshot at the label:

| Arm | Columns after the shared 6 | Total |
|---|---|---|
| `cams_ceda_temp` | CAMS global, beam, and diffuse irradiance; CEDA temperature | 10 |
| `cams_om_temp` | CAMS global, beam, and diffuse irradiance; Open-Meteo temperature | 10 |
| `ceda_ghi_temp` | CEDA global irradiance; CEDA temperature | 8 |
| `om_ghi_temp` | Open-Meteo global irradiance; Open-Meteo temperature | 8 |

The `cams_*` pair tests temperature alone, as #1024's solar contrast did. The `*_ghi_temp` pair tests
a solar forecast built on each UKV archive alone. An exploratory pair averages the snapshots at
both ends of the hour for both archives (Open-Meteo's `ghi_instant_w_m2` through
`weather_products._without_sunrise_spikes`), because that construction cut UKV's error by 0.60
points in the weather-products study. A solar hour needs both instants' CEDA runs to be complete.

**Station arms (question 4, set A).** The four hourly Met Office stations inside the CEDA crop
(labelled S1 to S4; `studies.midas.select_nearest_stations`) are scored for 10 m wind speed and for
air temperature, over 2024-09 to 2025-12 (16 months, era 0 only, because the station archive ends
2025-12-31). Both archives are UKV at 2 km, so the 2 km cell against a point reading penalises
both alike and cancels in the contrast, which #1024's station contrast could not claim.
**The score is the mean absolute error after removing each (station, archive, hour of day) mean
error**, the same removal for both archives. #1024 removed a mean per calendar month and hour, which
over 16 months leaves one or two months in each cell and would absorb a large share of the error.
The primary score is therefore the per-hour removal, and the plan prints the per-month-and-hour
removal and the raw error beside it.

## 3. Planned contrasts, margins, and the decision rule, frozen before any result

**Each contrast is CEDA minus Open-Meteo (or the transfer penalty), and a negative value favours
CEDA.** Verdicts have three outcomes, applied identically to every contrast:

- **Interchangeable:** the 95% interval lies wholly inside the margin on both sides of zero.
- **Differ:** the interval excludes zero and the point estimate lies beyond the margin.
- **Unresolved:** everything else, including a statistically significant difference smaller than the
  margin, which the page reads as "small, not clear".

**Margins and what the design can resolve.** A margin is the smallest difference the project would
act on. Wind uses 0.16 points of capacity and solar 0.06, the margins #1024 froze, so the two pages
read alike. The solar temperature-only pair uses 0.02, which is 2.8 times the scaled standard error
below, rounded up. #1024's saved intervals (about 78 scored months) give a standard error of 0.047
for the wind contrast (half-width 0.0915) and 0.0033 for the solar temperature contrast (half-width
0.0065). A standard error scales with the square root of the number of months, so this study's 23
months scale them by 1.84, to about 0.086 and 0.0061. Those errors come from contrasts between
different products and probably overstate the error between two copies of one weather model.
**A 95% interval of half-width 1.96 x 0.086 = 0.17 is wider than the wind margin of 0.16, so the
wind contrasts can reach "interchangeable" only if the real standard error is below 0.082.** Where a
wind contrast reads "unresolved", the page gives the bound ("an effect as large as 0.17 points is
not excluded").

| Label | Contrast | Margin | Decides |
|---|---|---|---|
| P1 (planned) | Set B wind, as-available arms: `ceda_wind` minus `om_wind`, points of capacity, both settings | 0.16 | Whether wind power differs between the archives |
| P2 (planned) | Set B solar temperature alone: `cams_ceda_temp` minus `cams_om_temp`, both settings | 0.02 | Whether temperature differs for solar |
| P3 (planned) | Set B solar, global irradiance and temperature: `ceda_ghi_temp` minus `om_ghi_temp`, both settings | 0.06 | Whether solar power differs between the archives |
| P4 (planned) | Transfer penalty: (trained on CEDA, scored on Open-Meteo) minus (trained on Open-Meteo, scored on Open-Meteo), on the matched arms. Two reads, wind (`*_wind_10m`) and solar (`*_ghi_temp`), at both settings | 0.16 wind, 0.06 solar | The training-history recommendation (question 5) |
| P5 (planned) | Set A, stations: CEDA minus Open-Meteo bias-removed mean absolute error, two reads, air temperature (K) and 10 m wind speed (m/s) | 5% of Open-Meteo's score on the same rows | Which archive is closer to the weather that happened (question 4) |

**Five planned contrasts and seven planned reads.** P1 to P3 decide question 2, and P5 decides
question 4. P4 decides question 5. The model-free comparison, every year, season, farm, station, lead
read, and arm outside the table is exploratory, as is anything added after the first run.

**Rules, by question.**

- **Question 2 (does the difference matter for power).** P1, P2, and P3 each report their verdict at
  both hyperparameter settings, and a verdict stands only if both settings agree. Where the settings
  disagree the verdict is "unresolved" and the page shows both.
- **Question 4 (which archive is closer).** The archive is called closer only if P5 reads "differ"
  for that variable at all hours and also at CEDA lead-0 hours. Open-Meteo's analysis is always
  fresher than a CEDA lead of 3 to 5 hours, so a gap that appears only at leads 1 to 5 is a
  difference of served lead, and the page says so rather than naming a better product.
- **Question 3 (eras and years).** Every contrast is reported for era 0 and era 1 separately, for
  each half-year, and for each calendar year with at least 6 scored months. The change in a
  contrast across the 2026-01-21 boundary is exploratory, as is each year's.
- **Question 5 (training history).** The study states one default and what moves it, and commits the
  project to nothing:

| P4 reads (each variable separately) | The page recommends |
|---|---|
| Interchangeable | CEDA-trained models can be served live UKV for that variable at the precision tested, subject to the licence |
| Differ | For that variable, train on live-lineage UKV only (history from 2024-08), or let a calibrator absorb the CEDA-to-live difference, and do not mix |
| Unresolved (the default) | Treat as "differ": do not mix lineages until a longer overlap exists |

**The decision is subject to the licence.** CEDA's licence is Creative Commons
Attribution-NonCommercial-ShareAlike 4.0. The study is non-commercial research, publishes only
error scores and charts with no UKV values, and cites "Met Office (2016): NWP-UKV: Met Office UK
Atmospheric High Resolution Model data. Centre for Environmental Data Analysis". Whether the main
work's use is non-commercial is the maintainer's decision.

## 4. Row set, gaps, and eras

**Rows are the intersection across every arm, decided from the target and availability, never from
a value.** An hour stays only if the target exists, both archives have every column every arm reads,
and (solar) CAMS has its three columns. All solar arms share one row set, because the `cams_*` pair
needs CAMS and the other pair does not. Wind rows drop the hours `wind_product_frames.common_rows`
drops. Solar rows drop the hours `solar_product_frames.common_rows` drops, and carry the export-cap
`constrained` flag that `out_of_fold_losses` reads (those hours are scored and never trained on).
A CEDA hour whose freshest run is missing or partial is dropped for every arm and never filled from
an older run, because that would change its lead. Open-Meteo's 94-hour gap in November 2024 is
dropped for every arm. The build stops if any month loses more than 25% of its hours to either
archive (#1024's rule), prints the loss per month, and lists the dropped months before any fit.

**Eras.** The two eras are 2024-09 to 2025-12 and 2026-02 onward. `cerra_past_solar.with_covering_folds`
already cuts at `FIRST_MONTHS = (UKV_UPGRADE_MONTH,)`, so no new fold code is needed. A synthetic
check for this plan found that `search_fold_offsets` returns covering rotations for the 23-month set,
and the build re-runs it on the real rows and stops if it returns none. Every arm gets `era_code`.
January occurs only in 2025, so one scored month has no January in training. The coverage check
exempts such a month, and the page prints an exploratory row without it.

**Open-Meteo's backfill (2024-01-01 to 2024-08-11) comes from an unnamed source and is excluded from
every power and station arm,** because the study skill treats a change of source as an era
boundary. The model-free script reads it as an exploratory row.

## 5. Folds, intervals, settings, metric, hardware

- **Folds:** `with_covering_folds`: five blocks of whole months per era, rotated to cover every
  calendar month that occurs in more than one year.
- **Intervals:** `studies.bootstrap.bootstrap_difference` resamples whole calendar months, paired
  across arms, with one of three fitting seeds, 2,000 resamples. Set A uses
  `bootstrap_row_difference`. Both cover month-to-month weather, not differences between
  generators or stations.
- **Hyperparameters:** `PRIMARY_HYPER_PARAMETERS` and `SENSITIVITY_HYPER_PARAMETERS` on P1 to P4.
  Controls, lead-0 refits, and the hour-ending pair run at the primary setting only.
- **Metric and hardware:** mean absolute error as percent of capacity, each row divided by its own
  generator's `effective_capacity` before any mean, with every arm's absolute error reported. Check
  `uptime` for a spinning core, then `nvidia-smi`, and use `device="cuda"` with column
  subsampling off (`colsample_bytree` of 1). One arm refits on the CPU for the noise floor.

## 6. Controls

- **Negative control.** Each archive's weather columns shuffled within (site, year-month, hour of
  day), one joint permutation per stratum, at the primary setting (the `cams_*` pair shuffles only
  temperature). Shuffled-CEDA minus shuffled-Open-Meteo should differ by about zero.
- **Positive control.** `om_wind` and `om_ghi_temp` each plus independent Gaussian noise, with a
  standard deviation equal to the root-mean-square difference between the archives at lead-0 hours
  (from the model-free comparison). The control shows how much power error a weather difference of
  the observed size produces. A monotone rescaling would prove nothing, because trees ignore it. A
  null contrast is read as "no effect" only when the positive control passed or the interval bounds
  the effect.
- **Set A.** A lag scan of each archive against each station at -3 to +3 hours must put the minimum
  error at lag 0.

## 7. Scripts, files, and outputs

Scripts go in `studies/past_weather/` and import only that folder, `studies.*`, and reviewed
packages. They reuse the #1024 machinery by import where the machinery stays, and by a move into
`packages/studies` where two studies now need it.

| Script | Job |
|---|---|
| `fetch_open_meteo_ukv_stations.py` | The one fetch: Open-Meteo's UKV 2 m temperature and 10 m wind speed at the four stations, 2024-08-12 to 2025-12-31, one request per station, checkpointed per station, resumable, per the `data-download` skill. Writes only S1 to S4 labels and the elevation Open-Meteo reports. Output under `data/studies/downloads/NWP/OPEN-METEO-PREVIOUS-RUNS/UKV/`, with a README and lineage note |
| `ukv_ceda_vs_openmeteo_compare.py` | Model-free comparison and the three diagnostics; `direct_report.md` |
| `ukv_ceda_vs_openmeteo_build.py` | Station-hours, wind rows, and solar rows; arm-column functions; `--check-only` prints coverage and stops on a failed guard; `--dry-run` builds one month |
| `ukv_ceda_vs_openmeteo_station_scores.py` | P5, the lag scan, and the station tables; no fit |
| `ukv_ceda_vs_openmeteo_fit.py` | Set B fits, transfer scoring, controls; `--dry-run`, `--verified`, `--report-only`; writes `report.md` and `decision.md` |
| `ukv_ceda_vs_openmeteo_charts.py` | Figures; every shared number is read from the reports |

**New and changed code in `packages/studies/`, each with tests.**

- **`ukv_ceda_stores.py` gains the store readers** now in `ukv_ceda_vs_era5_build.py`
  (`UkvStores`, `open_ukv_stores`, `nearest_ukv_cells`, `ukv_at`), and the #1024 scripts import them
  from the package. The proof the move changed nothing: the one-month `--dry-run` frames before and
  after the move are identical, and `ukv_ceda_station_scores.py` rerun into a scratch directory
  reproduces `station_intervals.parquet` bit for bit.
- **`station_scoring.py` (new)** takes the bias removal and station scoring out of
  `ukv_ceda_station_scores.py`, generalised from the fixed ERA5-and-UKV pair to any two products.
- **`cross_validation.out_of_fold_losses` gains `scoring_site_rows`**, an optional frame with the
  same rows and the feature columns of another archive under the training columns' names. The fold's
  model trains on `site_rows` and predicts on `scoring_site_rows`. Rows align by `time` and an
  unequal set raises. This is the transfer scoring, and no model is refitted for it.
- **`sources.py` gains** the study's write-once output directory and the station extract's directory.

**Output.** `data/studies/per_study/ukv_ceda_vs_openmeteo/` is write-once (`refuse_to_overwrite`);
a re-run after a review moves the earlier output to `superseded/` first. The folder holds the
frames and build stamp, every arm's per-row losses and out-of-fold predictions (site, time, fold,
seed, arm, setting, actual value, scoring archive), the interval tables, `report.md`, and
`decision.md`. Stamps record each arm's columns, the XGBoost version, the device, and input hashes.
One session owns running the scripts, because every worktree shares the main checkout's `data/`.

## 8. Compute estimate

**The fits total 123 fit-sets, plus one CPU refit.** Planned arms take 72 (wind: 4 arms, 2 settings,
3 farms is 24; solar: 4 arms, 2 settings, 6 farms is 48). Lead-0 refits take 18, shuffled controls
18, positive controls 9, and the wind hour-ending pair 6. The transfer scoring adds none. The #1024
fit-sets took 8 to 10 seconds on 6,000 to 14,000 rows, and this study has about 16,000 wind rows and
9,000 daytime solar rows per farm, so plan **5 to 20 seconds per fit-set, 10 to 40 minutes in all**.

## 9. The page and the docs

**The page** is `docs/studies/past-weather/ukv-ceda-vs-open-meteo.md`, under Studies > Past weather,
following the study skill's twelve sections. Figure 1 is the five planned contrasts with 95%
intervals, the margins as bands, each row labelled with both archives' absolute errors, and the
second setting as a marker. Each results section has a chart: the XGBoost models work (out-of-fold
power against measured, A to F and W1 to W3, over weeks chosen by a stated rule); model-free
differences by lead and by month (a step at 2026-01-21 shows); station scores by lead; the transfer
penalty; and the controls. The title is written after the results.

**Cross-references:** `docs/studies/index.md`, `docs/studies/past-weather/index.md`,
`past-weather/methods.md` (row set and planned contrasts), `studies/README.md`,
`studies/past_weather/README.md`, `packages/studies/README.md`, and the `mkdocs.yml` nav line.
`docs/roadmap/training-history.md` and `docs/roadmap/data-sources.md` change when the result lands,
in a separate small PR, so the study PR stays inside `studies/`, `packages/studies/`,
`docs/studies/`, and `mkdocs.yml`.

## 10. Risks and open questions, with recommendations

1. **Two archives of one weather model differ by little, and the design may not resolve it.**
   Recommendation: keep the margins, report the bound, and read the positive control to say what a
   difference of the observed size costs in power.
2. **Four stations are few.** Recommendation: report each station's rows separately and pool
   distances only.
3. **Multiple comparisons.** Five planned contrasts, no correction, one deciding read each. About
   one in 20 exploratory rows with no real effect reaches the 5% level, and the page says so.
4. **The wind transfer reads 10 m wind only,** a poorer predictor than the as-available columns.
   Recommendation: report the matched arms' absolute errors beside the as-available ones.
5. **Moving the store readers could change #1024's outputs.** Recommendation: the bit-for-bit proof
   in section 7 before any new script runs.
6. **Licence.** The maintainer decides whether the main work's use of CEDA is non-commercial.
   Recommendation: keep stores and per-row predictions in the private data store.

The plan commits only to this study. It does not commit to serving live UKV, to training on either
archive, to fetching further data, or to a decision about the licence.

## 11. Verification

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/README.md
uv run mkdocs build --strict      # and read the rendered page and its figures
uv run python studies/past_weather/ukv_ceda_vs_openmeteo_build.py --check-only   # must exit 0
```

Also run `pydoclint` and the docs-link checker locally, and the mutation pass, because
`packages/studies/` changes. Each new test must fail on the bug it exists to catch:

- **`scoring_site_rows`:** swapping in other values changes the prediction, passing the training
  frame itself reproduces the baseline exactly, and a scoring frame with a different set of times
  raises. None of this runs on `main`, which lacks the argument.
- **The moved store readers:** 06 UTC reads run 06 at lead 0, never run 00 at lead 6 (`<` against
  `<=`); a missing run is never served by an older run; the nearest cell is chosen on a synthetic
  grid.
- **`station_scoring`:** the mean error is removed within each (station, hour) and not across
  hours, and a fixture with unequal station counts per archive catches an unweighted mean.
- **Unequal capacities:** the normalisation test gives the fixtures unequal capacities.

## 12. What this plan does not commit the project to

The plan commits only to this study. It does not commit to serving live UKV, to training on either
archive, to fetching any further data, or to a decision about the CEDA licence.
