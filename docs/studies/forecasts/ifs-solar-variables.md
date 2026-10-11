# A second forecast beats extra IFS variables, but not on top of the ensemble mean

**At six solar farms in Lincolnshire, none of the extra variables in ECMWF's 9 km IFS forecast cut
the error of day-ahead to three-day-ahead power forecasts by a resolved amount, but adding a second
forecast (AIFS Single, ECMWF's machine-learned forecast) cut it by about 1 percentage point of
capacity.** IFS is ECMWF's Integrated Forecasting System. The second forecast's gain is a gain over
a single deterministic forecast, which is noisy at long leads. The IFS ensemble mean, which the
production service reads, gives about the same gain, and adding AIFS Single to that ensemble mean
adds only about 0.1 points, which the study cannot resolve. The page therefore does not say that the
production service would gain 1 point from a second forecast. The study also does not recommend
asking Dynamical.org, the production service's source of IFS ensemble forecasts, for any extra
variable.

![Figure 1: Change in solar farm error from adding IFS variables, the five planned
contrasts](../assets/ifs_solar_variables_headline.svg)

**How to read the numbers.** Every number is a mean absolute error as a percentage of a farm's
capacity, and a difference of 0.1 is 0.1 percentage points of capacity, called "points". A negative
difference is a gain. A bracket holds a 95% interval from resampling whole months, unless the text
says 99%, the level adjusted for the five planned contrasts. A contrast is planned if the study plan
named it before any result existed, and every other number is exploratory. The smallest improvement
worth acting on is 0.1 points. The forecasts are the 00 UTC run of every day, and lead day L means
the run was issued L days before the day it describes. F0 is the minimal variable set (IFS radiation
and air temperature). F1 adds total cloud cover, F2 adds the three cloud layers, and F6 is all 20
IFS variables. FB and F6B add AIFS Single's radiation and temperature to F0 and to F6. P1 is F1
minus F0, P2 is F2 minus F1, P3 is F6 minus F2, P4 is FB minus F0, and P5 is F6B minus F6.

- **Do extra IFS variables help predict solar farm output at lead days 1 to 3?** Not by a resolved
  amount.
  Total cloud cover (P1) lowers the error by 0.12 points, the cloud layers (P2) by 0.19 points, and
  every other variable together (P3) by 0.14 points (pooled over lead days 1 to 3, 99% intervals
  [−0.19, −0.04], [−0.32, −0.06], and [−0.31, +0.03]). All three intervals straddle the smallest
  improvement worth acting on (0.1 points), so all three are unresolved. Twenty-five months of data
  cannot resolve effects this small, and a width preview from the ERA5 study's losses said so before
  any fit.
- **Does a second forecast help?** Yes, on top of a single deterministic forecast.
  Adding AIFS Single's radiation and temperature to the minimal IFS set lowers the error by 1.04
  points (P4, 99% interval [−1.42, −0.67]), and adding them to all 20 IFS variables lowers it by
  0.79 points (P5, [−1.25, −0.44]). Both are gains at both XGBoost tuning settings and against a
  control with the same number of columns. They use 14 months, from March 2025 to April 2026.
- **Does a second forecast help the production service?** Not by a resolved amount.
  The production service reads the IFS ensemble mean. On the same rows, adding the ensemble mean to
  the minimal set lowers the error by 0.96 points, about as much as AIFS Single does. Adding AIFS
  Single to the ensemble mean lowers the error by 0.13 points (post hoc, 99% interval [−0.29,
  +0.02]), with +0.07 at lead day 1 and about −0.2 at lead days 2 and 3. The reference in P4 and P5
  is weak: Open-Meteo's 9 km IFS radiation has a higher raw error against CAMS irradiance than AIFS
  Single or the ensemble mean.
- **Is it worth ingesting the 12 variables that only ECMWF's MARS archive serves, or another
  forecast?** The study does not separate the two routes for the production service.
  The pre-set rule returns "second forecast", because P5 is a gain. That rule compares P5 with the
  ERA5 study's P4, and neither is the gain the production service would see. The like-for-like
  comparison is AIFS Single on top of the ensemble mean (about 0.1 points) against the 12 MARS-only
  variables on top of 29 other ERA5 variables (0.13 points at the main setting, 99.5% interval
  [−0.18, −0.08]), and those two are of the same size.
- **Which variables should the production service ask Dynamical.org to add?** None, on this
  evidence.
  Total cloud cover ranks first: its gain over the minimal set is 0.12 points [0.04, 0.19], but its
  gain over the four variables the production feed already carries is −0.02 points [−0.16, +0.13],
  so it adds nothing detectable. No other group reaches the rule.
- **Does the production service need a second IFS feed, for the cloud layers or the direct beam,
  which ECMWF's free feed lacks?** No, by the pre-set rule.
  Removing the cloud layers from all 20 variables raises the error by 0.07 points at the main
  setting and 0.06 at the second, below the 0.1-point smallest effect. The cloud layers lower the
  error by 0.21 points [−0.35, −0.09] over the four production variables (exploratory, main setting
  only), and the production feed's long-wave radiation, a cloud signal, was not tested.
- **What can the study support?** Statements about 9 km IFS and AIFS Single forecasts at six farms
  in one county, at lead days 1 to 3, over 25 months (14 for the second forecast). Three limits
  matter most. Every gain is measured on top of a single deterministic 9 km forecast, the farms
  share weather, and 25 months give wide intervals.

> **How this page was made.** The research question came from a human. Everything else — the code
behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's AI
model (for this page, Claude Sonnet 5.5, Claude Sonnet 5, and Claude Opus 5.5). Several independent
Claude reviewers have checked the method, the evidence, and the prose adversarially.

## Key findings

- **The forecast's error rises from 10.1% of capacity at lead day 1 to 15.4% at lead day 9 for the
  minimal set, and every variable group's gain stays small** (F6 minus F0 is −0.39 points at lead
  day 1 and −0.29 at lead day 9) ([Figure
  3](#the-error-rises-with-lead-day-and-the-extra-variables-gain-little)).
- **The ERA5 study's screen overstated the gain at forecast lead by about three times.** At lead day
  1 on the same rows, ERA5's full set beats its minimal set by 1.20 points and the IFS full set
  beats its minimal set by 0.39 points ([Figure 7](#the-era5-screen-overstated-the-forecast-gain)).
- **A second forecast's gain grows with lead day because the reference degrades faster.**
  Open-Meteo's IFS radiation error against CAMS irradiance rises by 26% from lead day 1 to 3, and
  AIFS Single's by 18% ([Figure 10](#a-second-forecast-and-the-ensemble-mean)).
- **One winter fold drives the lead-day-3 gain.** The only winter months of the 14 are held out
  together, so their models train on no winter row. Without that fold, P5 at lead day 3 is −0.73
  points, not −1.02.
- **As in the ERA5 study, the full set is worse than the minimal set at low sun.** At 06, 07, 19,
  and 20 UTC, F6 is worse than F0 by 0.08 to 0.29 points ([Figure
  8](#the-gain-by-season-regime-and-hour)).
- **The shuffled-copy controls make the error worse.** Three shuffled cloud columns made F0's error
  0.02 points worse and the 14 shuffled columns of F6 made F2's 0.04 worse, so a wide variable set
  starts at a disadvantage ([Figure 5](#controls)).

## Introduction

**The question is which variables of ECMWF's IFS forecast, beyond its solar radiation and
temperature, help a day-ahead to three-day-ahead solar power forecast, and whether a second weather
forecast would do more.** The [ERA5 study](../past-weather/era5-solar-variables.md) asked which
variables of ERA5, a reanalysis, help estimate solar output. A reanalysis is not a forecast, and the
ERA5 study found signs that part of its gain came from ERA5's cloud fields having seen later
observations than its solar radiation. This study repeats the question on real past forecasts at the
lead times a forecast service would use. Three decisions depend on the answer: which variables to
ask Dynamical.org to add, whether the production service needs a second IFS feed, and whether to
ingest the 12 MARS-only variables or another forecast.

| Group | Adds | IFS variables (Open-Meteo names) | In ECMWF's free open-data feed |
|---|---|---|---|
| F0 | the minimal set | `shortwave_radiation`, `temperature_2m` | yes |
| F1 | total cloud | `cloud_cover` | yes |
| F2 | cloud layers | `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high` | no |
| F3 | direct radiation | `direct_radiation` | no |
| F4 | humidity, column water, boundary layer | `dew_point_2m`, `total_column_integrated_water_vapour`, `boundary_layer_height` | the first two |
| F5 | convection and visibility | `cape`, `convective_inhibition`, `visibility` | no |
| F6 | surface and weather state | `surface_temperature`, `snow_depth`, `snowfall`, `precipitation`, `surface_pressure`, `wind_speed_10m`, `wind_gusts_10m` | yes |
| FP | the production reference | F0 plus `dew_point_2m`, `surface_pressure`, `precipitation`, `wind_speed_10m` | yes |

Each group F1 to F6 includes every group above it. The free feed's availability was read from
ECMWF's open-data index files for the 2026-10-09 00 UTC run. The feed carries most-unstable CAPE,
and does not carry the surface-based CAPE that Open-Meteo serves. FP is the production feed's
current variables, and the study reads F1, F2, and F6 against it.

## Data and methods

**The rows are 43,921 daylight farm-hours at six farms, from March 2024 to April 2026, and the
forecasts are the 00 UTC IFS run of each day.** The rows, the targets, and the solar geometry come
from the ERA5 study, so the cleaning rules are the same: hours that hold a zero half-hour are
dropped, hours the network operator constrained are scored but not trained on, and output is a
fraction of each farm's capacity (the 99th percentile of its metered output). Lead day L of an hour
is read from the run issued L days before the hour's day, at lead 24L plus the hour of day.
Instantaneous variables are the mean of the values at the two ends of the hour, as the ERA5 study
averaged ERA5's snapshots, and radiation, precipitation, snowfall, and gusts are the hour-ending
means Open-Meteo serves. Lead days 1, 2, 3, 5, 7, and 9 are scored, and values beyond 90 hours are
interpolated by Open-Meteo.

**IFS changed cycle twice in the archive, so the study cuts folds inside two eras.** Cycle 49r1 went
live on 2024-11-12 and cycle 50r1 on 2026-05-12. The study drops 2024-11 and 2026-05 and everything
after, and the first era starts on 2024-03-15, so its first month is a 17-day part-month. The two
eras hold 25 months. Each farm's rows in each era are cut into five blocks of whole months, with the
same fold for the same hour at every lead day.

**One XGBoost model per farm, lead day, and variable set is fitted.** XGBoost is a gradient-boosted
tree model. Intervals come from resampling whole months, paired across variable sets, with one of
the three fitting seeds drawn in each resample, 10,000 resamples for the planned contrasts and 2,000
for the others. The planned contrasts (P1 to P5, on output only) are pooled over lead days 1 to 3
with equal weight, at the 99% level, and both XGBoost tuning settings must agree for a verdict. A
verdict is "gain" if the interval's upper bound is below −0.1 points, "no gain" if its lower bound
is above −0.1, and "unresolved" otherwise. Controls give the arm's columns shuffled copies of the
extra variables (shuffled within each farm, month, and hour of day), and a contrast must also be a
gain against its control. A positive control adds CAMS irradiance, which must help.

**The second forecast is AIFS Single, and P4 and P5 use 14 months.** AIFS Single became operational
on 2025-02-25, so the blend rows start in March 2025. AIFS Single's radiation is a 6-hourly mean
that is rebuilt hourly through a clear-sky index, and its temperature is interpolated to the hour.
The partner's columns come from the AIFS Single store only, with no CAMS or power value in them. F0
and F6 are refit on the blend rows, so each blend contrast compares variable sets on identical rows.

## Results

### The data and their issues

**IFS and CAMS agree on the shape of a day and disagree in detail, more at longer leads.** Figure 2
draws IFS's radiation and cloud covers at lead days 1 and 7 against CAMS irradiance on three days at
one farm: the clearest, most variable, and dullest by CAMS's clear-sky index.

![Figure 2: IFS forecasts a day and a week ahead, against CAMS, on three
days](../assets/ifs_solar_variables_weather.svg)

### The error rises with lead day, and the extra variables gain little

**The minimal set's error rises from 10.1% of capacity at lead day 1 to 15.4% at lead day 9, and the
full set lowers it by 0.4 to 0.6 points at most.** The full set (F6) beats the minimal set (F0) by
0.39 points at lead day 1, 0.50 at lead day 2, 0.48 at lead day 3, 0.45 at lead day 5, 0.54 at lead
day 7, and 0.29 at lead day 9. Those are exploratory single-lead-day numbers.

![Figure 3: Error of F0, FP, and F6 at each lead
day](../assets/ifs_solar_variables_lead_day_skill.svg)

### The models work

**Predictions follow the measured output on months the XGBoost models never saw.** Figure 4 draws
three days at every farm, chosen separately for each farm by CAMS's clear-sky index, with each
farm's error for four variable sets. The farms are labelled A to F.

![Figure 4: Day-ahead predictions on months the models never saw, three days at every
farm](../assets/ifs_solar_variables_models_work.svg)

### Controls

**The positive control lowers the error by 5 points, so the method detects a large effect, and the
shuffled controls make the error worse.** Adding CAMS irradiance to F2 gives an error of 5.0% at
lead day 1, against 9.7% for F2. Three shuffled cloud columns make F0 worse by 0.02 points pooled,
and 14 shuffled columns make F2 worse by 0.04 points. A planned gain must also beat its control, and
the controls do not change any verdict.

![Figure 5: Errors of the negative and positive controls beside the arms they
pad](../assets/ifs_solar_variables_controls.svg)

### Every variable set at every lead day

**The leaderboard gives each variable set's error at each lead day, with intervals.** Its intervals
overlap more than the paired differences of Figure 1 do, because every variable set's error rises
and falls with the weather.

![Figure 6: Error of every variable set at every lead
day](../assets/ifs_solar_variables_leaderboard.svg)

### The ERA5 screen overstated the forecast gain

**At lead day 1 on the same rows, ERA5's full set lowers the error by 1.20 points and IFS's full set
by 0.39 points.** ERA5's total and layered cloud cover give 0.67 points against IFS's 0.33 points.
The gap is consistent with ERA5's cloud fields having seen later observations than its solar
radiation, and also with the difference between a same-hour estimate and a forecast. The page does
not choose between them.

![Figure 7: Change in error from cloud and from every variable: IFS forecast against
ERA5](../assets/ifs_solar_variables_era5_against_ifs.svg)

### The gain by season, regime, and hour

**The full set is worse than the minimal set at low sun.** F6 is worse than F0 by 0.08 points at 06
UTC, 0.16 at 07, 0.29 at 19, and 0.13 at 20 UTC, the same pattern as the ERA5 study found, which
supports its reading that the penalty comes from the farms' low-sun output and not from the weather
data. Figure 8 shows the gain by ERA5 cloud regime, season, and hour of day.

![Figure 8: Change in error by sky regime, by season, and by hour of
day](../assets/ifs_solar_variables_regimes.svg)

### Removing one group

**Removing one group from all 20 variables raises the error by at most 0.07 points, and no group
clears the drop-one rule except at one setting.** The rule asks that removing a group raises the
error by more than a control with the same number of shuffled columns does. Removing the cloud
layers raises the error by 0.074 points at the main setting and 0.057 at the second, against
controls of 0.035 and 0.101. Removing humidity, column water, and boundary-layer height raises the
error by 0.036 points at the main setting, which passes the control by 0.002 points, and by 0.016
points at the second setting, which fails it. The raise is 0.118 points at lead day 3 and about zero
at lead days 1 and 2.

![Figure 9: Change in error when one group of variables is removed from all 20 IFS
variables](../assets/ifs_solar_variables_drop_one.svg)

### A second forecast and the ensemble mean

**Adding AIFS Single to Open-Meteo's single 9 km forecast lowers the error by about 1 point, and
adding the IFS ensemble mean does the same.** Panel 1 of Figure 10 shows the three gains the third
decision compares: P4, P5, and the ERA5 study's P4 for the MARS-only variables. Panel 2 shows the
post hoc contrasts: AIFS Single added to the ensemble mean (−0.13 points, unresolved) and the
ensemble mean added to F0 (−0.96 points, a gain).

**The gain grows with lead day because the reference degrades faster.** Against CAMS irradiance,
Open-Meteo's IFS 9 km radiation has a raw error of 65.7, 73.3, and 82.7 W m⁻² at lead days 1 to 3, a
rise of 26%. AIFS Single's is 62.0, 66.9, and 73.1 (a rise of 18%), and the ensemble mean's is 61.4,
67.6, and 74.0 (20%). A sharp point forecast is penalised twice under absolute error when cloud is
misplaced, and a smoother forecast is not. The blend models also train on 11 months with no row of
the same calendar month, and F0 refit on the blend rows is 0.4 to 0.7 points worse than the ladder's
F0 on the same rows.

**One winter fold holds the only winter months, so its models train on no winter row.** Without that
fold, P4 pooled over lead days 1 to 3 is −1.01 points (−1.04 with it) and P5 is −0.67 points (−0.79
with it). The direction holds in every farm and in 12 to 14 of the 14 months, and the size does not.

![Figure 10: A second forecast against the MARS-only variables: the gains the third decision
compares](../assets/ifs_solar_variables_variables_or_forecast.svg)

## Discussion: what to use

- **Do not ask Dynamical.org for extra variables on this evidence.** Total cloud cover is the only
  candidate, and it adds nothing detectable to the variables the production feed already carries
  (−0.02 points [−0.16, +0.13]). The page cannot say that total cloud cover has no value, because 25
  months cannot resolve a gain of 0.1 points.
- **Do not take a second IFS feed for the cloud layers on this evidence.** The cloud layers lower
  the error by 0.21 points over the production variables, but the drop-one test is below the
  smallest effect and the production feed's long-wave radiation, a cloud signal, was not tested.
- **Do not choose between the MARS-only variables and a second forecast from P5.** On the ensemble
  mean the production service reads, a second forecast adds about 0.1 points, which is unresolved
  and about the size of the 0.13 points the ERA5 study found for the MARS-only variables.
- **Do not read the 1-point gain as the value of AIFS Single.** The ensemble mean gives the same
  gain over the 9 km forecast, so the gain comes from replacing a noisy forecast with a smoother
  one.

## Limitations

- **Every gain is measured on top of a single deterministic 9 km point forecast.** A smoother
  reference, such as the ensemble mean, leaves less for any added input to fix, so every gain here
  is likely to be smaller in production.
- **The production service's 3-hourly ensemble is not tested.** The ensemble mean used here is on
  6-hourly steps, and the production feed's 3-hourly mean may do slightly better.
- **Twenty-five months are few.** They hold perhaps 150 to 250 independent weather episodes, the
  intervals are approximate, and the post hoc contrasts use 14 months.
- **The first era includes a 17-day part-month, and November is scored with no November in
  training.** The blend rows have ten calendar months with no training row of the same calendar
  month.
- **The newest model cycle is not scored.** Cycle 50r1 is in use from 2026-05-12, and the months
  after it are too few to cut into folds.
- **The planned contrasts were checked against controls with the same number of columns, but the
  exploratory ones were not.** The page does not correct exploratory intervals for multiple
  comparisons, and 33 near-the-line exploratory results have no second-setting run.
- **Some analyses are post hoc.** The ensemble-mean contrasts, the winter-fold reading, and the
  raw-error table were added after the first scientific review, and the report labels them.
- **ICON-EU was not tested as a second forecast.**

## Scope

**The page says nothing about wind, other regions, or leads beyond 9 days.** It tests Open-Meteo's 9
km IFS and AIFS Single only, and the production service's 0.25° ensemble differs from the 9 km run.

## Data and code availability

**IFS forecasts come from Open-Meteo's Single Runs archive, AIFS Single from the Dynamical.org
store, and CAMS irradiance from the Copernicus Atmosphere Data Store.** The page contains modified
Copernicus Atmosphere Monitoring Service information. The farms' output is private to National Grid
Electricity Distribution (NGED) and appears only as fractions of each farm's capacity under labels A
to F. The code is in `studies/ifs_solar_variables/` and `packages/studies/`. XGBoost runs on a
graphics processing unit with `colsample_bytree=1`.

## Reproducing the figures

```bash
uv run python studies/weather_downloads/fetch_open_meteo_ifs_solar_variables.py
uv run python studies/ifs_solar_variables/ifs_ladder_build_dataset.py
uv run python studies/ifs_solar_variables/ifs_ladder_build_dataset.py --ens-only
uv run python studies/ifs_solar_variables/ifs_ladder_fit.py --view ladder --target both
uv run python studies/ifs_solar_variables/ifs_ladder_fit.py --view blend
uv run python studies/ifs_solar_variables/ifs_ladder_fit.py --view ens --lead-days 1 2 3
uv run python studies/ifs_solar_variables/ifs_ladder_fit.py --view extra_sensitivity --lead-days 1 2 3
uv run python studies/ifs_solar_variables/ifs_ladder_report.py
uv run --with vl-convert-python python studies/ifs_solar_variables/ifs_ladder_charts.py
```
