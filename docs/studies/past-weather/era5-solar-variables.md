# Adding ERA5's cloud and other variables to its solar radiation cuts solar farm error by about 12%

**At six solar farms in Lincolnshire, an XGBoost model estimates each farm's output in an hour from
ERA5's weather for the same hour, and giving the model every ERA5 variable instead of only solar
radiation, temperature, and sun position cut its error by about 12%.** XGBoost is a gradient-boosted
tree model, and ERA5 is the European Centre for Medium-Range Weather Forecasts' (ECMWF) estimate of
past weather, called a reanalysis. The estimate is not a forecast, because the model sees weather
from the hour it estimates. Total and layered cloud cover give about half of the 12%. The cloud
groups are largely interchangeable with the other variables, so how the credit splits depends on the
order in which the study added the variables. ERA5 builds its solar radiation from short forecasts
and its cloud fields from the analysis, which has seen later observations, so part of the gain may
be a timing advantage that a real forecast lacks. The page therefore does not say that a forecast
would gain as much, and a follow-up study of IFS forecasts (ECMWF's operational forecasting system)
at matched lead times tests that.

![Figure 1: Change in error from adding ERA5 variables, the ten planned
contrasts](../assets/era5_solar_variables_headline.svg)

**How to read the numbers.** An error is a mean absolute error as a percentage of a farm's capacity,
and a difference of 0.1 is 0.1 percentage points of capacity (called "points"). A negative
difference is a gain. A bracket holds a 95% interval from resampling whole months unless the text
says 99.5%, the level adjusted for the ten planned contrasts. A contrast is planned if the study
plan named it before any result existed. The smallest improvement worth acting on is 0.1 points for
output and 0.01 for the CAMS clearness index (a second target, explained below). G0 is the minimal
variable set, G1 adds total cloud cover, G2 adds the three cloud layers, and G9 adds every ERA5
variable studied. P0 is G9 minus G0, P1 is G1 minus G0, P2 is G2 minus G0, P3 is G9 minus G2, and P4
is G9 minus G9 without 12 variables that only ECMWF's MARS archive serves.

- **Which ERA5 variables help predict solar farm output?** Cloud cover first, then the other groups
  in smaller steps.
  Adding total cloud cover (G1) lowers the error from 9.22% to 8.89% of capacity (−0.32 points,
  99.5% interval [−0.41, −0.25]). Adding the cloud layers as well (G2) lowers it by 0.59 points
  [−0.72, −0.48], or 6.4%. Every variable (G9) lowers it by 1.14 points [−1.31, −0.98], or 12.4%.
  All three planned contrasts beat the smallest improvement worth acting on (Figure 1).
- **Does anything beyond the cloud layers help?** Yes, but the credit cannot be split cleanly.
  Going from G2 to G9 lowers the error by a further 0.55 points (95% interval [−0.65, −0.46]). Cloud
  water and cloud base height give the largest single step (0.33 points). The other groups each
  change the error by between +0.03 points (snow and albedo, which make it worse) and −0.07 points.
  Removing any one group from the full set costs at most 0.07 points, probably because the groups
  carry overlapping information (Figures 7 and 10).
- **Is it worth fetching the 12 variables that the operational IFS forecast offers only through
  ECMWF's MARS archive?** Unresolved.
  The rule fixed before the results was to recommend a pilot fetch if the 99.5% interval of P4 lay
  wholly below −0.1 points at both settings, and to recommend against if it lay wholly above. The 12
  variables lower the error by 0.13 points (99.5% interval [−0.18, −0.08]) at the main XGBoost
  tuning setting and by 0.11 points ([−0.15, −0.06]) at the second. Both intervals straddle −0.1, so
  the rule gives no recommendation. The gain is about the size of the threshold, so a longer record
  would not settle the question. The 12 variables are not in ECMWF's free open-data forecast feed or
  in Open-Meteo's. The follow-up IFS study puts the gain beside the gain from adding a second
  forecast model.
- **Does the gain hold on the satellite irradiance target?** In relative terms, yes.
  The second target is the CAMS clearness index: irradiance from the Copernicus Atmosphere
  Monitoring Service's satellite service divided by the irradiance at the top of the atmosphere. The
  full set lowers its error by 14.5%, against 12.4% for output. The CAMS verdicts look weaker only
  because the CAMS smallest effect (0.01) is about 10% of its error, while the output's (0.1 points)
  is about 1% of its error.
- **Does aerosol help?** Not detectably. A gain of 0.1 points of capacity is ruled out, and the
  clear and dusty hours are too few to say.
- **Does the extra detail help the XGBoost model say how uncertain it is?** No.
  A second XGBoost model fits the 10% to 90% quantiles, giving an interval that should hold 80% of
  outcomes. At the same coverage, one fixed-width interval is no wider than the quantile fits'
  intervals.
- **What can the study support?** Statements about ERA5 at six farms in one county from September
  2019 to June 2026. Three limits matter most: ERA5 is not a forecast, the farms share weather, and
  the order in which variables were added affects how credit is split.

> **How this page was made.** The research question came from a human. Everything else — the code
behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's AI
model (for this page, Claude Sonnet 5.5, Claude Sonnet 5, and Claude Opus 5.5). Several independent
Claude reviewers have checked the method, the evidence, and the prose adversarially.

## Key findings

- **Under the order the variables were added in, cloud cover, cloud layers, and cloud water carry
  about 0.9 of the 1.14 points of gain.** The cloud groups are largely interchangeable with the
  other variables, so the split depends on the order. Humidity and haze (G7) add 0.04 points beyond
  the groups before them, and snow and albedo (G8) make the error 0.03 points worse [0.01, 0.04],
  about twice what 4 shuffled columns cost in the negative control ([Figure
  7](#every-variable-set-scored-on-the-same-hours)).
- **As a share of the error, the full set's gain is largest from 13 to 17 UTC.** The gain from cloud
  water and cloud base steps up between the labels 09 and 10 UTC, where ERA5's cloud fields move to
  the next analysis window, so part of the gain may be a timing advantage of a reanalysis. On the
  output target the full set is worse than the minimal set at low sun, and the penalty does not
  appear on the CAMS target ([Figure 11](#the-gain-changes-with-the-hour-of-the-day)).
- **Twenty-nine columns of shuffled weather made the error 0.11 points worse.** Wide variable sets
  start at a disadvantage, so their gains are probably understated ([Figure
  7](#every-variable-set-scored-on-the-same-hours)).
- **Adding CAMS irradiance to the output target lowers the error by 3.65 points [−3.84, −3.45].**
  This positive control shows that the instrument can detect a large effect, and that ERA5's other
  variables recover a minority of what its solar radiation misses ([Figure
  7](#every-variable-set-scored-on-the-same-hours)).
- **The gain is similar in every season and in every regime of ERA5's total cloud cover** (11% to
  14% of the error at the farms' output) ([Figure
  8](#the-gain-is-similar-in-every-season-and-cloud-regime)).
- **Aerosol from CAMS's reanalysis changes the error by −0.03 points [−0.05, +0.00].** A gain of 0.1
  points is ruled out at the 95% level ([Figure
  14](#aerosol-adds-little-and-dusty-hours-are-too-few-to-say)).
- **The quantile intervals are too narrow in both tails, and the extra variables narrow them
  further** ([Figure 13](#the-extra-variables-do-not-improve-the-uncertainty-estimates)).

## Introduction

**The question is which ERA5 variables, beyond solar radiation and temperature, help estimate how
much power a solar farm produces.** A forecast service chooses which weather variables to ingest,
and each costs engineering effort. ERA5 is the cheapest place to screen the variables, because it
has nearly seven years of hourly values for the study period for every one of them, whereas an
archive of past forecasts has far fewer. This study tests 32 ERA5 single-level variables beyond
solar radiation and temperature. It adds them in groups, in a fixed order (the "ladder"), and
credits each group with the drop in error when it is added (its "step").

| Group | Adds | ERA5 variables |
|---|---|---|
| G0 | the minimal set (9 columns) | downward solar radiation (`ssrd`), 2 m air temperature (`t2m`), sun position, two top-of-atmosphere fluxes, hour of day, day of year, and ERA5's clearness index (`ssrd` over the top-of-atmosphere flux) |
| G1 | total cloud | `tcc` |
| G2 | cloud layers | `lcc`, `mcc`, `hcc` |
| G3 | clear-sky irradiance | `ssrdc`, and the clear-sky index derived from it |
| G4 | cloud water and base | `tclw`, `tciw`, `tcslw`, `cbh` |
| G5 | direct beam | `fdir`, `cdir` |
| G6 | wind and thermal radiation | `u10`, `v10`, `strd` |
| G7 | humidity and haze | `d2m`, `tcwv`, `blh` |
| G8 | snow and albedo | `sd`, `sf`, `asn`, `fal` |
| G9 | all other ERA5 variables | `tp`, `tcrw`, `tcsw`, `cape`, `cin`, `skt`, `tco3`, `uvb`, `i10fg`, `sp`, `deg0l` |
| G10 | aerosol (not ERA5) | CAMS reanalysis aerosol optical depth, total and dust |

Each group includes every group above it. The 12 MARS-only variables are available for ERA5 from the
Climate Data Store, and for the operational IFS forecast only from ECMWF's MARS archive. They are
`ssrdc`, `cdir`, `tclw`, `tciw`, `tcslw`, `cbh`, `tcrw`, `tcsw`, `tco3`, `uvb`, `fal`, and `deg0l`.

## Data and methods

**The XGBoost model estimates output in an hour from ERA5's values for the same hour, so the page
measures how well ERA5 describes past sunshine at the farms, with no lead time.** The rows are
126,229 daylight hours at six farms, from September 2019 to June 2026. A daylight hour is one with
the sun's top-of-atmosphere horizontal flux above 50 W m⁻². Hours that hold a zero half-hour are
dropped, which also drops hours with a fully snow-covered panel. Rows per farm range from 5,539 to
27,012, so the pooled error weights each farm by its rows. Output is each farm's hourly output as a
fraction of its capacity, which is the 99th percentile of the farm's metered output and not its
nameplate rating. Predictions are held at the export cap. Hours that the network operator
constrained are scored but left out of training. The second target is the CAMS clearness index.
Every variable set is scored on exactly the same hours, except that the aerosol comparison uses only
the hours that CAMS's reanalysis covers.

**One XGBoost model per farm is fitted for each variable set.** Each farm's own span is cut into 5
contiguous blocks of whole months, so a month is never in training and test sets together, and each
model trains on the other four fifths. The error is the mean absolute error as a percentage of the
farm's capacity, from three fitting seeds. An interval comes from resampling whole months, paired
across variable sets, because neighbouring farms share their weather. Each resample draws one of the
three seeds, and there are 10,000 resamples for the planned contrasts and 2,000 for the exploratory
ones. A result is "statistically significant at the 5% level" when its 95% interval excludes zero.
The intervals cover the month-to-month weather and the fitting seed, not differences between farms.
For uncertainty, a second XGBoost model per variable set fits the 10%, 20%, and up to 90% quantiles
for G0, G2, G9, G9 without the MARS-only variables, the negative control, and the aerosol
comparison, at the main tuning setting only.

**Planned means written into the study plan before any result existed.** The plan named ten
contrasts (P0 to P4 on each target) and a decision rule for the MARS variables. They are shown at
the 99.5% level, adjusted for ten contrasts, and at 95%. A planned verdict needs both XGBoost tuning
settings to agree. Every other number is exploratory: an exploratory row with no real effect behind
it has a 5% chance of reaching statistical significance, the number of such rows is unknown, the
rows share months so spurious results cluster, and the page does not correct for multiple
comparisons.

**The controls are a negative control and a positive control, and a second XGBoost tuning setting
checks the result.** The negative control adds 29 shuffled copies of the extra weather to G2,
keeping each month's mean at each hour. The positive control adds CAMS irradiance to G2 on the
output target, a column that must help. The main tuning setting uses trees of depth 6, a learning
rate of 0.05, and 500 rounds. The second uses depth 4, a learning rate of 0.03, and 1,200 rounds,
with a stronger penalty on small leaves and on large weights. Neither is tuned, and every planned
contrast ran at both.

**ERA5 hour conventions.** Accumulations (solar radiation, thermal radiation, precipitation) are
means over the hour ending at the label. Instantaneous variables are averaged over the labels H−1
and H so the hour matches. All times are UTC.

## Results

### The data and their issues

**ERA5's cloud fields and CAMS's irradiance agree on the shape of a day and disagree in detail.**
Figure 2 draws ERA5's irradiance, cloud, and cloud water beside CAMS's irradiance on three days at
one farm (the clearest, most variable, and dullest by CAMS's clear-sky index).

![Figure 2: ERA5 variables and CAMS irradiance on three
days](../assets/era5_solar_variables_days.svg)

**ERA5's solar radiation differs from CAMS's irradiance in ways that depend on cloud.** On the
clearest day ERA5 is slightly darker than CAMS, and on the dullest day it is brighter. Binned by
cloud variables, the gap is mixed: in the hours with the most cloud liquid water, ERA5 is darker
than CAMS. Figures 3 to 5 show where ERA5's solar radiation differs from CAMS, and how CAMS's
clearness index falls with ERA5's cloud cover and cloud water.

![Figure 3: CAMS clearness index against each ERA5 cloud
cover](../assets/era5_solar_variables_cloud_against_clearness.svg)

![Figure 4: Where ERA5's downward solar radiation differs from
CAMS](../assets/era5_solar_variables_where_ssrd_misses.svg)

![Figure 5: CAMS clearness index against ERA5 cloud water, by low-cloud
cover](../assets/era5_solar_variables_cloud_water.svg)

### The XGBoost models work

**Predictions follow the measured output on months the models never saw.** Figure 6 draws three days
at every farm, chosen by a stated rule per farm, and each farm's error for four variable sets. The
farms are labelled A to F.

![Figure 6: Predictions on months the models never saw, three days at every
farm](../assets/era5_solar_variables_models_work.svg)

### Every variable set scored on the same hours

**The leaderboard shows the ladder, both controls, and the drop-one runs on the farms' output.** Its
intervals overlap more than the paired differences of Figure 1 do, because every variable set's
error rises and falls with the weather.

![Figure 7: Error and correlation of every variable set for the solar farms'
output](../assets/era5_solar_variables_leaderboard_pv.svg)

![Figure 7b: Error and correlation of every variable set for the CAMS clearness
index](../assets/era5_solar_variables_leaderboard_cams.svg)

### The gain is similar in every season and cloud regime

**The full set lowers the error by 11% to 14% of the farms' output error in every season and in
every regime of ERA5's total cloud cover.** Clear, broken, and overcast hours gain 13.7%, 11.2%, and
13.1% of their error. Regimes defined from CAMS's clear-sky index differ more: broken skies gain
9.0% and overcast skies 16.8%.

![Figure 8: Change in error by sky regime from ERA5's total cloud
cover](../assets/era5_solar_variables_regimes.svg)

![Figure 8b: Change in error by sky regime from CAMS's clear-sky
index](../assets/era5_solar_variables_regimes_cams.svg)

![Figure 9: Change in error by season](../assets/era5_solar_variables_seasons.svg)

![Figure 9b: Change in error by season and sky
regime](../assets/era5_solar_variables_regimes_by_season.svg)

### Removing one group costs little

**Removing any one group from the full set costs at most 0.07 points on output, probably because the
groups overlap.** Removing the last group (G9, 0.07), cloud water and base (G4, 0.06), or wind and
thermal radiation (G6, 0.06) costs the most. By the rule fixed in the plan, under which a group is
useful only if both the step in the ladder and the drop-one run show a gain, the useful groups at
the main tuning setting are the cloud layers, cloud water and base, wind and thermal radiation,
humidity and haze, and the remaining ERA5 variables. Total cloud, clear-sky irradiance, and the
direct beam are not useful once the other groups are present, although removing total cloud costs
0.02 points [0.01, 0.04] at the second setting. Snow and albedo make the error worse. The credit
moves if the order of the ladder changes, so only the planned contrasts and these variable-level
statements are safe.

![Figure 10: Change in error when one group of variables is removed from the full
set](../assets/era5_solar_variables_drop_one.svg)

### The gain changes with the hour of the day

**As a share of the error, the full set's gain is largest from 13 to 17 UTC, the gain from cloud
water and cloud base steps up between the labels 09 and 10 UTC, and the full set is worse than the
minimal set at low sun on the output target.** At the farms' output, the G4 step is 1.4% of the
error at 09 UTC and 3.8% at 10 UTC. On the CAMS target it is 2.7% at 09 UTC and 4.9% at 10 UTC. In
summer, when the sun is high at 09 UTC, the output step is from 1.9% to 4.2%. The full set lowers
the error by 8.1% at 09 UTC and by 14% to 15% from 13 to 17 UTC. It is worse than the minimal set by
0.13 to 0.17 points at 05, 06, 19, and 20 UTC. At the same hours, 29 shuffled columns cost 0.05 to
0.11 points, so extra columns explain part of that penalty but not all of it, and the cloud-water
step alone makes 05 UTC worse by 0.10 points. The penalty does not appear on the CAMS target, where
the full set is better at every hour from 06 to 20 UTC and indistinguishable at 05 UTC, so the
penalty is more likely to come from the farms' low-sun output than from ERA5. Each instantaneous
value at label H is the mean of the raw values at H−1 and H, so label 10 is the first label that
holds data from the new analysis window. ERA5 builds its cloud, humidity, and wind fields from the
analysis, whose window changes between 09 and 10 UTC, and builds solar radiation from the forecasts
started at 06 and 18 UTC. A step at that hour is what a timing advantage of the analysis fields
would produce, but a gain weighted to the afternoon could also come from cloud types that vary with
the time of day. The page therefore treats the timing as likely and unquantified. A refit with the
instantaneous fields taken at label H−1 only would test it, because the step should then move one
label later. That refit was not run.

![Figure 11: The gain from the extra variables by hour of
day](../assets/era5_solar_variables_hour_of_day.svg)

### Which variables the XGBoost model leans on

**ERA5's solar radiation takes about two thirds of the full set's importance, then the ultraviolet
flux (7.2%), the direct beam (5.2%), and rain water (1.9%).** Three of the top six are MARS-only:
the ultraviolet flux, rain water, and the forecast albedo. The ultraviolet flux and the forecast
albedo come from the same forecast as solar radiation, so those two have no timing advantage. The
importance of an XGBoost model is descriptive, because correlated variables share the credit. The
shuffled-copy noise line is 0.09% for the largest shuffled copy, so the top 12 variables are well
above noise.

![Figure 12: XGBoost importance of the full set, the solar farms'
output](../assets/era5_solar_variables_importance_pv.svg)

![Figure 12b: XGBoost importance of the full set, CAMS clearness
index](../assets/era5_solar_variables_importance_cams.svg)

### The extra variables do not improve the uncertainty estimates

**The quantile fits' 10% to 90% intervals contain 72% to 76% of outcomes, against the 80% they aim
for.** The full set's intervals are 17% narrower than the minimal set's, and contain 72.8% against
75.9%. At that coverage, one fixed-width interval from the same errors is 21.4 points of capacity
wide against 22.9 for the full set's quantile fit, which is 0.934 of its width (0.995 for the
minimal set). The fixed-width interval is fitted on the errors it is scored on, which flatters it
slightly, but the ratio falls as variables are added. The extra variables therefore do not help the
XGBoost model estimate its own uncertainty. The CRPS, a score for a whole forecast distribution,
falls from 6.32 to 5.65, which follows the better central estimate.

![Figure 13: Do any inputs help the XGBoost model say how uncertain it
is?](../assets/era5_solar_variables_probabilistic.svg)

![Figure 13b: Are the predicted quantiles
calibrated?](../assets/era5_solar_variables_reliability.svg)

![Figure 13c: Coverage against interval
width](../assets/era5_solar_variables_coverage_and_width.svg)

### Aerosol adds little, and dusty hours are too few to say

**On the hours that CAMS's reanalysis covers, aerosol changes the error by −0.03 points [−0.05,
+0.00].** A gain of 0.1 points is ruled out at the 95% level. In clear and dusty hours (55 days in
27 months), the interval is −0.26 to +0.29 points, so the study cannot say. The reading rule needed
an interval wholly below −0.1, and gives "not shown". In dusty hours, aerosol moves the mean signed
error from +0.83 to −0.08 points, so it removes an over-forecast without lowering the mean absolute
error. The CAMS target's gain of 0.001 is partly circular, because CAMS's irradiance is built with
CAMS aerosol.

![Figure 14: Whether CAMS aerosol helps in cloud-free and dusty
hours](../assets/era5_solar_variables_aerosol_conditions.svg)

## Why cloud variables can add to ERA5's solar radiation

**A literature search done for this study found three reasons that ERA5's other variables can add to
its own solar radiation.** ERA5 runs IFS cycle 41r2 ([Hersbach et al.
(2020)](https://doi.org/10.1002/qj.3803)), whose radiation scheme sees only a monthly aerosol
climatology ([Tegen et al. (1997)](https://doi.org/10.1029/97JD01864)) and corrects its fluxes
between hourly radiation calls for the sun's movement but not for changing cloud ([Hogan and Bozzo
(2015)](https://doi.org/10.1002/2015MS000455)). ERA5 overestimates irradiance under cloud and
slightly underestimates it under clear sky, and the bias grows when irradiance is turned into PV
output ([Urraca et al. (2018a)](https://doi.org/10.1016/j.solener.2018.02.059); [Urraca et al.
(2018b)](https://doi.org/10.1016/j.solener.2018.10.065)). ERA5's downloaded cloud, humidity, and
wind fields change at the analysis window boundaries, so they appear to come from the analysis,
which saw later observations than the 06 and 18 UTC forecasts that give solar radiation. The most
similar study the search found post-processed a regional weather model's irradiance forecasts at
Dutch stations, using the same weather model's cloud, cloud water, humidity, and direct radiation.
That study raised the root-mean-square-error skill by 2% to 3% in winter and by 7.5% to 10% in
summer and autumn ([Bakker et al. (2019)](https://doi.org/10.1016/j.solener.2019.08.044)). That
result is on forecasts, so it is evidence that the effect can survive without a reanalysis's timing
advantage. The search found no paper that corrected ERA5 irradiance over Europe with humidity
predictors.

## Discussion: what to use

- **Ingest total cloud cover first, and the cloud layers second, when choosing variables for a
  forecast of solar output.** Total cloud cover alone gives 0.32 points. The cloud layers give
  another 0.27 points. Once every other group is present, removing total cloud cover changes the
  error by only +0.01 to +0.02 points. Cloud water and cloud base height give a further 0.33 points
  but are MARS-only. Whether a forecast gains the same is untested here, and the follow-up IFS study
  tests it.
- **Do not decide on the 12 MARS-only variables from this page.** The gain of 0.11 to 0.13 points is
  real and may or may not exceed 0.1 points. The page gives no recommendation either way. The
  question is whether another forecast model would give more, and the follow-up IFS study measures
  that.
- **Check which groups a forecast supplier can deliver before ranking them.** On 2026-10-10, ECMWF's
  [free open-data page](https://www.ecmwf.int/en/forecasts/datasets/open-data) listed total cloud
  cover, 2 m dew point, total column water vapour, 10 m wind, skin temperature, snow depth,
  precipitation, and downward solar and thermal radiation, and did not list the cloud layers, the
  cloud water and cloud base height, the direct beam, the clear-sky radiation, or the boundary layer
  height. [Open-Meteo's IFS service](https://open-meteo.com/en/docs/ecmwf-api) lists total and
  layered cloud cover, direct radiation, boundary layer height, and total column water vapour, and
  does not list the clear-sky radiation, the cloud water, or the cloud base height. The "not listed"
  entries were not rechecked against ECMWF's full parameter table, and the page did not check every
  one of the 12 MARS-only variables against each service.
- **The study found no gain from reanalysis aerosol.** A forecast would see less of a dust plume
  than a reanalysis does.

## Limitations

- **ERA5 is not a forecast.** Part of the gain from cloud, humidity, and wind variables may come
  from those fields having seen later observations than solar radiation. The hour-of-day figure fits
  that reading but does not prove it.
- **The order in which variables were added sets how credit is split.** A group that comes later
  gets only what the earlier groups missed.
- **Extra columns have a cost.** Twenty-nine shuffled columns made the error 0.11 points worse, so
  the wide sets' gains are probably understated. Informative but correlated columns may cost less
  than shuffled ones.
- **Six farms in one county share weather.** The sample size is months of weather. Intervals cover
  that, not differences between farms or regions.
- **The snow variant was not run.** Hours with a zero half-hour are excluded, which drops hours with
  a fully snow-covered panel, so the snow and albedo variables are untested where they would matter.
- **Some worst days look like output faults.** Several of the worst days at one farm in March 2025
  are clear days with almost no output. They add the same noise to every variable set, so the
  contrasts stand, and the report lists them.
- **Some second-setting runs were added after the first results.** Every planned contrast ran at
  both settings, as planned. For exploratory rows, the second setting was run after the first
  results, for rows on either target whose 95% interval ended within a fifth of its width of zero, a
  rule written in an early version of the plan.
- **The aerosol rows end in December 2025,** when CAMS's reanalysis ends, so the aerosol comparison
  covers fewer months than the rest.
- **The correlation in the leaderboard is dominated by the daily cycle.** On the farms' output,
  every variable set except the positive control sits between 0.88 and 0.91, so the correlation is a
  weak guide.

## Scope

**The page says nothing about forecast skill, other regions, wind, or any lead time.** It tests ERA5
only, and CAMS reanalysis aerosol only for the hours that reanalysis covers.

## Data and code availability

**ERA5 comes from Google's ARCO-ERA5 copy, which agreed with the Copernicus Climate Data Store to
floating-point rounding on the variables and months checked.** CAMS irradiance and the CAMS
reanalysis aerosol come from the Copernicus Atmosphere Data Store. The farms' output is private to
NGED and appears only as fractions of each farm's capacity under labels A to F. The code is in
`studies/era5_solar_variables/` and `packages/studies/`. XGBoost runs on a GPU with
`colsample_bytree=1` and the settings in "Data and methods".

## Reproducing the figures

```bash
uv run --with netCDF4 python studies/weather_downloads/fetch_era5_solar_variables.py --tier tier1a
uv run --with zarr --with gcsfs python studies/weather_downloads/fetch_era5_solar_arco.py --tier tier1b
uv run --with zarr --with gcsfs python studies/weather_downloads/fetch_era5_solar_arco.py --tier tier2
uv run --with zarr --with gcsfs python studies/weather_downloads/fetch_era5_solar_arco.py --tier tier3
uv run python studies/weather_downloads/fetch_cams_eac4_aod.py
uv run --with netcdf4 python studies/era5_solar_variables/era5_ladder_build_dataset.py
uv run python studies/era5_solar_variables/era5_ladder_fit.py --view both --max-workers 8
uv run python studies/era5_solar_variables/era5_ladder_fit.py --view extra_sensitivity \
  --sensitivity-arms drop_g1 drop_g2 drop_g3 drop_g7 drop_g8 g7 g8 --max-workers 8
uv run python studies/era5_solar_variables/era5_ladder_importance.py --through-rung g9
uv run python studies/era5_solar_variables/era5_ladder_report.py
uv run --with vl-convert-python python studies/era5_solar_variables/era5_ladder_charts.py
```
