# Plan: which IFS forecast variables help a day-ahead to nine-day-ahead solar PV forecast?

Status: draft, written before any IFS result exists. It follows
[PR #1097 (Plan: which ERA5 variables help explain solar PV output and CAMS
irradiance)](https://github.com/openclimatefix/nged-substation-forecast/pull/1097),
whose ERA5 screen asks the same question of a reanalysis. The IFS data is fetched and validated.

## Question

**If an XGBoost model (a gradient-boosted tree library) is given the extra variables that
Open-Meteo serves from ECMWF's high-resolution forecast (IFS), does it predict solar photovoltaic
(PV) output better than an XGBoost model given only the minimal IFS set, at forecast lead times of
1 to 9 days?** IFS is the Integrated Forecasting System of the European Centre for Medium-Range
Weather Forecasts (ECMWF). The study repeats, on real past forecasts, the question the ERA5 study
asks of a reanalysis. ERA5 is not a forecast, so a positive ERA5 result does not guarantee a gain
at a lead time, and a negative ERA5 result may reflect ERA5's age. The follow-up answers the
question where it matters: the forecast the NGED service would receive.

**Two decisions depend on the answer.** Which variables to ask Dynamical.org (the service that
supplies IFS ensemble forecasts to the production pipeline) to add, and whether the production
forecast should take a second IFS feed. The study does not test the 12 MARS-only variables, because
Open-Meteo does not serve them. The ERA5 study's planned contrast P4 stays the only evidence on
those.

## Why this design

- **Open-Meteo's Single Runs archive is the fastest source of past IFS forecasts that carry the
  variables.** It holds the 00 UTC run of every day from 2024-03-14, with hourly values out to 240
  hours, at one request per run. The fetch is done: 933 runs, six solar farms, 20 variables, in
  `data/studies/downloads/NWP/ECMWF-IFS-SINGLE-RUNS-SOLAR/`. Eight run days are missing (five in
  August 2025, three in June and September 2026).
- **The archive is 2.6 years, not the six of the ERA5 study.** Intervals are wider, so the planned
  contrasts name effects the sample can resolve, and the page reports what it cannot.
- **Open-Meteo builds some hourly values from coarser output.** IFS output steps coarsen beyond
  90 hours, so values at lead days 5 to 9 are interpolated, and `diffuse_radiation` is derived and
  not requested. The page says so.

## Arms: a shortened ladder from the free IFS variables

Every arm is shown the shared geometry and calendar columns of the ERA5 study
(`studies.era5_ladder.SHARED_FEATURES`), so a variable cannot win by supplying sun position. The
ladder is shorter than ERA5's, because IFS gives fewer variables. Each rung adds one physical idea.

| Rung | Adds | IFS variables (Open-Meteo names) |
|---|---|---|
| F0 minimal | what the ERA5 study's minimal set holds | `shortwave_radiation`, `temperature_2m` |
| F1 total cloud | cloud amount | `cloud_cover` |
| F2 cloud layers | where the cloud is | `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high` |
| F3 direct beam | the direct share of the sunlight | `direct_radiation` |
| F4 humidity and column water | haze and cloud-forming moisture | `dew_point_2m`, `total_column_integrated_water_vapour`, `boundary_layer_height` |
| F5 convection and visibility | convective cloud and haze | `cape`, `convective_inhibition`, `visibility` |
| F6 the rest | surface and weather state | `surface_temperature`, `snow_depth`, `snowfall`, `precipitation`, `surface_pressure`, `wind_speed_10m`, `wind_gusts_10m` |

F6 is the full set, with 20 variables. The ladder orders the rungs by the amount of information a
forecast service can supply cheaply, and drop-one-group runs (below) separate the order from each
group's own effect. Convective inhibition is missing wherever it is undefined (about 92% of
hours), and XGBoost reads a missing value natively.

**Every arm has the same number of feature columns where a contrast needs it, and column
subsampling is off** (`colsample_bytree=1`), as in the ERA5 study.

## Targets

- **PV output** (primary): the farm's hourly power divided by its effective capacity, from the
  same cleaned hourly series the ERA5 study uses, with the same filters (daylight, no outage runs,
  no commissioning ramp, export cap honoured).
- **CAMS clearness index** (second target): CAMS global horizontal irradiance over the
  top-of-atmosphere flux, so a gain that the PV target shows can be separated from curtailment.

Both targets are scored with `absolute_error_capped_fraction_of_capacity`, as in the ERA5 study.

## Rows, lead days, folds, and the metric

- **The row set comes from the ERA5 study's dataset.** The study reads
  `dataset_main_through_g9.parquet` for each (farm, valid hour), and joins the IFS forecast whose
  run began `L` days before the valid day, for each lead day `L`. Lead day `L` of a valid hour at
  hour `h` of its day is the run's lead of `24 L + h` hours. The lead days are 1, 2, 3, 5, 7, and
  9. Lead day 10 has one hour in the run and is not used.
- **Within one lead day, every arm is scored on the same rows.** An hour is kept only if every
  IFS variable of the full set is present (convective inhibition excepted), and a missing run day
  drops that valid day at that lead day only. Each lead day is its own model and its own sample.
- **Eras.** IFS cycle 49r1 went live on 2024-11-12 and cycle 50r1 on 2026-05-12, so the study cuts
  folds within each era, adds an era feature, and drops the two straddling part-months, per the
  study skill. The third era holds only 2026-06 and is dropped, because one month cannot form
  folds. The study window is therefore 2024-04 to 2024-10 and 2024-12 to 2026-04, 24 whole
  months (the exact span and the rows per farm are printed by the build).
- **Folds, seeds, intervals.** Contiguous blocks of whole months within each farm and era
  (`studies.cross_validation.assign_folds`), three seeds, and a paired bootstrap resampling whole
  months and one seed (`studies.bootstrap`), with 10,000 resamples for planned contrasts.
- **Smallest effects of interest:** 0.1 percentage points of capacity for PV and 0.01 for the
  clearness index, as in the ERA5 study.
- **Second hyperparameter setting** (`SENSITIVITY_HYPER_PARAMETERS`) on every planned arm.

## Planned contrasts (written before any result exists)

Three lead days (1, 3, and 7) and three contrasts on the PV target make nine planned contrasts,
each a treatment minus a reference, so a negative difference is a gain.

1. **P1:** F1 minus F0. Does total cloud help beyond the radiation and temperature a forecast
   already supplies?
2. **P2:** F2 minus F1. Do the cloud layers help beyond total cloud?
3. **P3:** F6 minus F2. Does any other free variable help beyond the cloud layers?

The family-wise level of 5% becomes 0.556% per contrast (a 99.44% interval). A contrast "rules out
a gain larger than the smallest effect" when its lower bound lies above minus the smallest effect.
Every other number is exploratory: the other lead days (2, 5, and 9), the rungs F3 to F5 one by
one, the CAMS target, the regime and season splits, and the importance. The page labels them
exploratory and does not correct them.

**Decision rule for the variables to request, fixed before any result.** A group of variables is
recommended for Dynamical.org when, at lead days 1 and 3 on the PV target at both hyperparameter
settings, its planned contrast lies wholly below minus the smallest effect. It is recommended
against when the contrast lies wholly above minus the smallest effect at both lead days. Otherwise
the page reports it unresolved.

## Controls

- **Negative control:** F2 plus a permuted copy of the F3 to F6 variables
  (`studies.blending.climatology_permutation`, over farm, month, and hour of day), at lead days 1,
  3, and 7. It shows what the extra columns produce from nothing. It is not strictly information
  free, because it keeps each month-and-hour mean.
- **Positive control:** F2 plus CAMS global horizontal irradiance at the valid hour, which is the
  observation. It must show a large gain, which shows that the instrument can detect one.
- **Drop-one-group runs** from F6 (exploratory), for lead days 1, 3, and 7.

## Comparison with the ERA5 study

The page draws the ERA5 contrast and the IFS contrast side by side for each rung that both studies
share (F1 against G1, F2 against G2 and so on), at lead day 1. This is the experiment that shows
whether the ERA5 screen transfers to a forecast. The page states that the two studies cover
different spans, and that an ERA5 contrast on the IFS study's span is also computed from the ERA5
losses so the comparison is on the same months.

## Page structure and figures

The page follows the study skill's figure-led form. It opens with the title, a bottom line of a few
sentences, and the headline figure above the fold. The take-home message is general (which IFS
variables help a solar PV forecast, and how far ahead), and it says what it means for the
Flexpectation project.

1. **Headline.** The planned contrasts (P1 to P3) at lead days 1, 3, and 7, with the adjusted and
   95% intervals, and the smallest effect marked.
2. **What the forecasts look like.** Three days at one farm: the IFS cloud covers and radiation at
   lead days 1 and 7 against the CAMS observation, with the same days' PV output.
3. **How skill falls with lead.** The error of F0 and F6 at every lead day, so a reader sees why
   far leads gain less.
4. **The models work.** Predictions against measured output on held-out months at lead day 1.
5. **Controls.** The negative and positive controls at lead days 1, 3, and 7, beside F0 and F6.
6. **The leaderboard.** Every rung's error by lead day, with intervals.
7. **ERA5 against IFS.** The contrasts that both studies share, side by side.
8. **Regimes and seasons.** The difference in error between F6 and F0 by ERA5 cloud regime (clear,
   broken, overcast) and by season, exploratory.
9. **Drop-one-group** and **importance**, exploratory.

Each figure carries the text it needs to be read alone, and its text is reviewed (a first-stumble
reader, then a one-rule-per-pass sweep) before the figure is first rendered.

## Data and code

- **Inputs.** The ERA5 study's dataset (rows, targets, geometry, export cap, ERA5 cloud regime),
  `ECMWF-IFS-SINGLE-RUNS-SOLAR.parquet` (validated: the four repeated variables equal the sibling
  product's exactly, no duplicate keys, no value outside its range), and the CAMS irradiance
  already in the dataset.
- **Code.** A new folder `studies/ifs_solar_variables/` holds `ifs_ladder_arms.py`,
  `ifs_ladder_build_dataset.py`, `ifs_ladder_fit.py`, `ifs_ladder_report.py`,
  `ifs_ladder_charts.py`, and a README. Code both studies need goes into
  `packages/studies/src/studies/` with tests (the matched-lead join, the era labelling, and the
  lead-day rows). Each script has at least one fresh Opus review before it runs.
- **Compute.** The fits run as the ERA5 fits do: a slot from the study coordinator, GPU, resumable
  checkpoints. A lead day is about 56,000 rows per farm-block, so the whole ladder is a fraction
  of the ERA5 fit.
- **Outputs.** `data/studies/per_study/ifs_solar_variables/`, and the page
  `docs/studies/forecasts/ifs-solar-variables.md`.

## What the study cannot say

- It does not test the 12 MARS-only IFS variables. Only the ERA5 study speaks to them.
- It uses one run a day (00 UTC) and the deterministic high-resolution forecast. It says nothing
  about the ensemble, which is what the production pipeline ingests, or about later runs.
- The history is 2.6 years with two model cycles, so intervals are wide and a result can change
  with a new cycle.
- Open-Meteo's hourly values beyond 90 hours are not native model output.
- Training on forecasts from the same archive that serves the test makes the result an upper
  bound on what a feed with different timing gives.

## The five complexity triggers (for sizing)

- **Changes what is stored:** yes. A published study page and its results.
- **Touches the production serving path:** no.
- **Touches a degradation rule:** no.
- **Admits more than one defensible design:** yes. The ladder, the lead days, and the eras can
  each be cut differently.
- **Spans code whose callers one could not name without searching:** yes, because it shares
  `packages/studies`.

The study is complex. It gets the plan and all four reviews: two Opus science reviews, the diff
review, the mutation pass (`packages/studies` changes), and the prose and persona reviews.

## Order of work

The IFS fetch is done and validated. Next: this plan and one Opus plan review, the build script
and its review, the fits (after the ERA5 fits release their slot), the report, the first Opus
science review, charts with their text reviewed before the first render, the page, the second
Opus science review, the diff review, the mutation pass, the first-stumble and one-rule-per-pass
prose reviews, the persona reviews, and merge. PR for this study stays a draft until the reviews
are done.
