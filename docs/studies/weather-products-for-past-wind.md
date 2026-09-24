# Which weather product best describes past wind?

> **This study uses only Flexpectation's own data, and its purpose is to inform Flexpectation's
> choices.** The study scores each weather product only at the wind farms in Flexpectation's trial
> area in Lincolnshire. The study does not compare results across many regions or climates, so a
> result on this page may not hold elsewhere.

**At three wind farms in Lincolnshire, the German weather service's ICON-D2 and the Met Office's UK
variable-resolution model (UKV) describe past wind best of the five products tested.** ICON-D2
belongs to the German weather service's (DWD's) Icosahedral Nonhydrostatic (ICON) model family. For
each weather product, an XGBoost model, a gradient-boosted tree, was fitted per generator to predict
hourly output from that product's wind, and scored by its mean absolute error as a percentage of the
generator's capacity (its 99th percentile of metered output). Each bracketed pair below is a 95%
interval. Across the window of this study, August 2024 to September 2026, the XGBoost
model's error given UKV's wind is 0.44 percentage points of capacity [0.24, 0.63] lower than given
the wind of ERA5, the reanalysis of the European Centre for Medium-Range Weather Forecasts (ECMWF).
Given ICON-D2's wind, the error is 0.58 points [0.40, 0.75] lower than given ERA5's, in a comparison
chosen after the results were seen. The two gaps are 6% and 8% of ERA5's error.

**For historical features in the live service the page recommends UKV, and for training history
ICON-D2 where ICON-D2 covers, with ICON-EU or ERA5 elsewhere.** Historical features and training
history are two of the parts of this project that read past weather, described in the
[introduction](#introduction). For the other two, capacity estimation and disaggregation, this study
makes no recommendation. The evidence is three wind farms in flat Lincolnshire, and 50,734
generator-hours from August 2024 to September 2026.

![Figure 1: ICON-D2 and UKV have the lowest errors of the five products tested, and ICON global the highest](assets/wind_leaderboard.svg)

![Figure 2: UKV, ICON-D2, and ICON-EU each beat ERA5 by a margin statistically significant at the
5% level](assets/wind_headline.svg)

**Figure 1's intervals are wide mainly because every product's error rises and falls together from
month to month.** Some months are harder to describe than others for every product, and resampling
whole months carries that shared swing into each product's own interval. The three generators also
share their weather, so each interval rests on 26 months rather than on thousands of independent
hours. Figure 2 pairs two products on the same hours, which cancels the shared swing. Two products
whose intervals overlap in Figure 1, or in the top panel of Figure 2, can therefore still differ by
a margin that is statistically significant at the 5% level. Only a contrast pairing those two
products tests them directly, and the bottom panel of Figure 2 holds the four planned ones.

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Opus 5.5 and Claude Sonnet 5, reusing the data-preparation and
> model-fitting code that Claude Opus 5 wrote for the [beam/diffuse
> study](beam-diffuse-split.md)). Several independent Claude reviewers have checked the method, the
> evidence, and the prose adversarially.

## Key findings

**A difference between two errors is in percentage points of capacity, written "points", and a
bracketed pair after a figure is its 95% interval.** Where a product "beats" another, the difference
is statistically significant at the 5% level. A weather model's value for an hour, as served, comes
from the freshest run of that weather model that Open-Meteo's archive holds for the hour.

- **UKV's and ICON-D2's advantage over ERA5 is larger from April to September, and from October to
  March only ICON-D2's advantage over ERA5 is statistically significant at the 5% level.** See [UKV
  and ICON-D2 describe past wind
  best](#ukv-and-icon-d2-describe-past-wind-best-of-the-five-products-tested).
- **In an exploratory comparison, ICON-D2 leads UKV across the window, but not since the Met Office
  upgraded UKV in January 2026.** Across the window ICON-D2 leads by 0.14 points [0.03, 0.25]. Since
  the upgrade UKV is 0.06 points ahead [−0.03, +0.19]. See [ICON-D2 leads UKV across the window, but
  not since UKV's upgrade](#icon-d2-leads-ukv-across-the-window-but-not-since-ukvs-upgrade).
- **An XGBoost model given the 80 m wind of ICON-EU, the European ICON model, beats an XGBoost
  model given ERA5's 100 m wind.** An XGBoost model given the 100 m wind Open-Meteo serves for
  ICON-EU also beats ERA5. From October to March, the difference between ICON-EU at 80 m and ERA5
  is not statistically significant at the 5% level. See [ICON-EU beats ERA5 at 80 m and at 100
  m](#icon-eu-beats-era5-at-80-m-and-at-100-m-mostly-from-april-to-september).
- **UKV's lead over ICON-EU is small, 0.13 points [0.01, 0.23], and depends on the XGBoost model's
  settings.** See [ICON-EU beats ERA5 at 80 m and at 100
  m](#icon-eu-beats-era5-at-80-m-and-at-100-m-mostly-from-april-to-september).
- **ICON global, DWD's global model, had the largest error of the five as served here, and about
  half of its gap to ICON-EU is a pair of steps in the wind Open-Meteo's archive serves for ICON
  global at one generator.** With the XGBoost models for ICON global and ERA5 both told when the
  steps fall, the difference between ICON global and ERA5 is not statistically significant at the 5%
  level. See [About half of ICON global's gap to ICON-EU is a pair of steps in its served
  wind](#about-half-of-icon-globals-gap-to-icon-eu-is-a-pair-of-steps-in-its-served-wind).
- **In an exploratory comparison on January to September of each year, UKV's lead over ERA5 grew
  from 0.46 points in 2025 to 0.75 points in 2026, a change of 0.30 points [0.04, 0.58] after
  rounding, while ICON-EU's and ICON-D2's leads did not change by a margin statistically
  significant at the 5% level.** See [ERA5's deficit, year by year](#era5s-deficit-year-by-year).
- **ICON-DREAM-EU, DWD's newer reanalysis, does not beat ERA5, and trails ICON-EU, the operational
  model it is built on, by 0.34 points [0.27, 0.41] at both hyperparameter settings.** See
  [ICON-DREAM-EU does not beat ERA5, and trails
  ICON-EU](#icon-dream-eu-does-not-beat-era5-and-trails-icon-eu).

## Introduction

This page is the wind counterpart of [Which weather product best describes past
sunshine?](weather-products-for-past-solar.md). Several parts of this project read an estimate of
weather that has already happened: capacity estimation, training history, historical features in the
live service, and disaggregation. The solar page describes each of these consumers of past weather,
which are parts of the project, not electricity customers. Two of the four can use a result about
wind: training history, the years of past weather that pre-training a forecasting model needs, and
historical features, which give the live forecasting model the weather of hours already past. This
page measures how well five weather products describe past hub-height wind at the three metered wind
farms in the trial area, and says which product those two consumers should read.

**Each product publishes wind at different heights, and each archive serves a different lead, so the
comparison has to state the height and the lead each product is scored at.** The served lead is how
many hours after a run started its archived value was forecast. A lead of zero, written T+0, is the
run's analysis: the weather model's best estimate of the weather at the moment the run starts. CAMS,
the Copernicus Atmosphere Monitoring Service's satellite retrieval, describes past sunshine best of
the eight products on the solar page but publishes no wind, so it is not compared here.
ICON-DREAM-EU, DWD's reanalysis, is scored separately in [ICON-DREAM-EU does not beat ERA5, and
trails ICON-EU](#icon-dream-eu-does-not-beat-era5-and-trails-icon-eu), on its own row set.

| Product | What it is | Grid spacing, native and as served | Wind heights served | Served lead | Covers all of Great Britain? | Start of the hub-height wind archive read here | Available after |
|---|---|---|---|---|---|---|---|
| ERA5 | ECMWF reanalysis, a consistent record back to 1940 | about 31 km; served on a 0.25° grid | 10 m and 100 m | an hourly analysis | yes | 1940 | about 5 days |
| UKV | Met Office model for the UK | 1.5 km over the UK, coarsening to 4 km at the domain's edges; served at 2 km | 100 m (Open-Meteo also serves 50 m and 80 m) | T+0, the analysis | yes | August 2024 | about 4 hours |
| ICON-D2 | DWD model for Germany and neighbouring countries | 2.2 km; served at about 2 km | 80 m and 120 m | 0 to 2 hours | no: its western edge runs from about 2°W on the south coast to about 2.5°W in the Midlands | November 2022 | about 1.5 hours |
| ICON-EU | DWD model for Europe, nested inside ICON global | 6.5 km; served at about 7 km | 80 m and 120 m | 0 to 2 hours | yes | November 2022 | about 3.5 hours |
| ICON global | DWD global model | 13 km; served at about 11 km | 80 m and 120 m | 0 to 5 hours | yes | November 2022 | about 3.5 hours |

![Figure 3: ICON-D2 has no data west of a line running from 1.8°W at 49.9°N to 3.9°W at 57.3°N.
The map also draws AROME France, which this page does not test](../roadmap/assets/weather_product_domains.svg)

## Data and methods

### What each error measures

**Every error on this page is a mean absolute error as a percentage of each generator's own 99th
percentile of output, which the page calls its capacity.** That capacity is a statistic of the
metered output, not the farm's registered or export capacity. No wind measurement is used. Each
error is that of an XGBoost model fitted per generator to predict hourly output from one product's
wind, so the figures rank how much each product's wind says about the output, not how close its
speeds are to the true wind.

### Where each product comes from, and its height, lead, and history

**ERA5 comes from Open-Meteo's copy of the Copernicus archive, and the other four from Open-Meteo's
historical-forecast archive.** `verify_era5_sources.py` checked Open-Meteo's ERA5 radiation and
temperature against the Copernicus original, but not its wind. ERA5 is built as a consistent
multi-decade record from one fixed weather model, not for local wind at a single farm, and two of
the three farms here share one ERA5 grid cell.

- **Hub heights.** The three wind farms' hub heights are not known. For each ICON product,
  Open-Meteo serves a 100 m wind that is the product's 120 m speed multiplied by 0.98, so an XGBoost
  model cannot tell the served 100 m wind apart from the 120 m speed. The XGBoost model is therefore
  given each ICON product's 80 m wind, which is not a fixed multiple of another height, and ERA5's
  and UKV's 100 m wind. A check gives the XGBoost model UKV's 80 m wind instead, and scores 0.02
  points worse (6.85% against 6.83%).
- **Leads.** ERA5's hourly wind is an analysis at every hour, not a forecast. ERA5's largest
  hour-to-hour jumps fall between 09 and 10 UTC and between 21 and 22 UTC, the boundaries of the
  12-hour windows in which ERA5 absorbs observations, not at the start of its forecasts. The served
  leads for ICON-D2 and ICON global come from where their hour-to-hour jumps fall, because a jump
  marks the hour at which the archive switches from one run to the next. ICON-EU's jumps show its
  3-hourly pattern only weakly, so ICON-EU's lead rests on its 3-hourly run cycle and on the lineage
  check described on the solar page. The ICON leads here are an hour shorter than the solar page's 1
  to 3 hours, because wind is an instantaneous value at the hour's timestamp while radiation is a
  mean over the hour before it.
- **History.** Open-Meteo's archive of UKV hub-height wind starts in August 2024. This study starts
  on 12 August 2024, when Open-Meteo's own UKV downloader started, so the comparison covers only 2
  years. ERA5 starts in 1940, and Open-Meteo's archive of the ICON products' 80 m wind starts in
  November 2022. DWD has run ICON global and ICON-EU since 2015, and ICON-D2 since February 2021.

### How the comparison was made

**The method is the solar study's, with three differences: each hour of power is centred on its
timestamp, hours holding an exactly-zero half-hour are dropped, and every product is read from its
nearest grid cell over land.**

- **Same hours, same XGBoost model, same number of inputs.** An hour is kept only if all five
  products cover it. A gradient-boosted tree (XGBoost) is fitted per generator on one product's
  hub-height speed, that height's direction, and its 10 m speed. The tree is also given the time of
  day, the season, and which side of the Met Office's PS47 upgrade of UKV, on 21 January 2026, the
  hour falls on. No hour is removed for curtailment, because National Grid Electricity Distribution
  (NGED), the distribution network operator this project forecasts for, has confirmed that no wind
  farm in the trial area is under active network management.
- **The power hour is centred on its timestamp.** Open-Meteo's wind is an instantaneous value at the
  timestamp, where its radiation is a mean over the hour before. So the hour of power labelled T is
  the hour from 30 minutes before T to 30 minutes after. Every product scores best with the hour
  centred this way. The solar study's hour, ending at T, raises UKV's error by 0.28 points, in a
  check added after the first run.
- **Every hour holding an exactly-zero half-hour is dropped.** From April 2026 NGED's telemetry feed
  publishes no exact zeros at two of the generators. Their calm half-hours are missing instead, and
  the build already drops an hour with a missing half-hour. Dropping every hour holding an exact
  zero, before April 2026 as well, makes the two periods match. Most dropped hours are calm, so
  behaviour near the turbines' cut-in speed is under-sampled. Dropping no hours at all moves no
  contrast by more than 0.031 points, in a check added after the first run.
- **Every product is read from its nearest grid cell over land.** At one generator ICON global's
  nearest cell is influenced by the sea, and outside June 2025 to June 2026 its 10 m speed runs
  about 30% above that of the nearest land cell. At the other two generators ICON global's nearest
  cell is the land cell.
- **Folds, normalisation, and intervals as in the solar study.** Each XGBoost model is trained on
  some blocks of whole months and scored on the others; each held-out block is called a fold. The
  folds are cut separately before and after the UKV upgrade, and the rest of January 2026 after the
  upgrade is dropped. Each error is divided by its own generator's capacity. Each 95% interval comes
  from resampling whole calendar months, each time also drawing one of three XGBoost fits that
  differ only in their random seed. This page calls a difference statistically significant at the 5%
  level when its 95% interval from resampling whole months lies wholly on one side of zero, and not
  statistically significant at the 5% level when the interval includes zero. The test covers
  month-to-month variation in the weather and the fitting seed only, not variation between
  generators. The intervals are not corrected for the number of comparisons, so among the many
  exploratory rows some will reach significance by chance.
- **Four planned contrasts:** ICON-EU against ERA5, UKV against ERA5, ICON-EU against UKV, and
  ICON-D2 against ICON-EU. A contrast is the difference between two products' errors on the same
  hours. A comparison is planned when it was written into the study plan before any result existed;
  every other figure is exploratory, chosen or added after results were seen. The distinction
  matters because with many comparisons, about 1 in 20 exploratory rows reaches significance at the
  5% level by chance, so an exploratory result is a lead to follow up rather than a finding. A chart
  holding both kinds marks each planned row "(planned)". A chart whose rows are all one kind says so
  once, in its subtitle. Every contrast is also rerun with a second hyperparameter setting for the
  tree. The plan written before the run gave the XGBoost model each product's served 100 m wind.
  After the first run, each ICON product was switched to its 80 m wind. The results with the served
  100 m wind are reported alongside and change no ranking.

## Results

### The XGBoost models work

**Given either ICON-D2 or ERA5, the XGBoost model follows the shape of measured power at every
generator, in a windy, a variable, and a calm week.** The two predictions run close together,
because ICON-D2's advantage over ERA5 is a small fraction of the error. Figure 4 plots both XGBoost
models' out-of-fold predictions against measured power. At Generator W3 both predictions run well
above measured power for days at a time, most visibly in the windiest week. Generator W3's output
swings for months at a time with turbine availability that no product records, as described in
[UKV and ICON-D2 describe past wind
best](#ukv-and-icon-d2-describe-past-wind-best-of-the-five-products-tested).

**The three weeks are chosen by a rule that reads measured power alone, so the choice cannot favour
ICON-D2 or ERA5.** The rule pools the three generators. The windiest week is the one in which the
generators' mean output was highest relative to their own capacity, and the calmest week the one in
which it was lowest. The most variable week is the one whose daily mean output swings the most from
day to day. The windiest week fell in December 2024, the most variable week in February 2025, and
the calmest week ran from July into August 2026.

**Figure 4 shows each week as days 1 to 7, with no calendar dates, because a dated hourly series
would identify the wind farm.** With only 3 wind farms in the study, a generator's hourly
output on known dates could be matched against publicly available generation data. That match
would undo the anonymisation that names the generators only as Generator W1 to W3. The page
therefore gives each week's month and year in the text and no finer date.

![Figure 4: An XGBoost model given ICON-D2 follows measured power at every generator, across a
windy, a variable, and a calm week](assets/wind_models_work_timeseries.svg)

**ICON-D2, UKV, ICON-EU, and ERA5 rank in the same order at each of the three generators.** Figure 5
plots each product's mean absolute error at each generator separately, one dot per product per
generator. ICON global is left out of Figure 5, because its dots would show which generator carries
the steps in ICON global's served wind, described in [About half of ICON global's gap to ICON-EU is
a pair of steps in its served
wind](#about-half-of-icon-globals-gap-to-icon-eu-is-a-pair-of-steps-in-its-served-wind), and this
page does not name that generator.

![Figure 5: ICON-D2, UKV, ICON-EU, and ERA5 rank in the same order at each of the three generators](assets/wind_models_work_error.svg)

### UKV and ICON-D2 describe past wind best of the five products tested

**UKV beats ERA5 by 0.44 points [0.24, 0.63] across the window, and ICON-D2 beats ERA5 by 0.58
points [0.40, 0.75].** ICON-D2 also beats ICON-EU at every generator, by 0.26 points [0.19, 0.33]
across the three. ERA5 trails every product except ICON global, and [Why ERA5 describes past
sunshine and wind worse than most current weather
products](../roadmap/data-sources.md#why-era5-describes-past-sunshine-and-wind-worse-than-most-current-weather-products)
sets out ERA5's documented weaknesses.

| Product | Mean absolute error, % of capacity |
|---|---|
| ICON-D2 | 6.69 |
| UKV | 6.83 |
| ICON-EU | 6.96 |
| ERA5 | 7.27 |
| ICON global | 7.65 |

**Both leaders' advantage over ERA5 is larger from April to September, and only ICON-D2's is
statistically significant at the 5% level from October to March.** UKV beats ERA5 by 0.71 points
[0.58, 0.85] from April to September, and is 0.13 points ahead [−0.17, +0.41] from October to March.
ICON-D2 beats ERA5 by 0.77 points [0.63, 0.90] from April to September and by 0.36 points [0.08,
0.64] from October to March. The seasonal split rests on parts of three summers and two winters, and
its cause was not examined.

![Figure 6: UKV's and ICON-D2's advantage over ERA5 is larger from April to September](assets/wind_half_years.svg)

**UKV's advantage over ERA5 is larger since its upgrade, and ICON-D2's is not.** Since the upgrade
UKV beats ERA5 by 0.79 points [0.63, 0.97], against 0.53 points [0.29, 0.72] over the same months of
the year before the upgrade. Over the same two periods ICON-D2's advantage over ERA5 is 0.73 and
0.79 points, so the change sits with UKV rather than with ERA5. The post-upgrade figures rest on 8
months, so their intervals are likely too narrow.

**UKV's advantage over ERA5 is statistically significant at the 5% level at two of the three
generators.** At the third generator the advantage is not statistically significant at the 5% level.
That generator's output swings for months at a time against every product's wind, which this page
attributes to turbine availability the feed does not record.

![Figure 7: UKV's advantage over ERA5 is statistically significant at the 5% level at two of the three
generators](assets/wind_per_generator.svg)

### ICON-D2 leads UKV across the window, but not since UKV's upgrade

**ICON-D2 leads UKV across the window, but not since UKV's upgrade.** Across the window ICON-D2
leads by 0.14 points [0.03, 0.25], an exploratory comparison. Before the upgrade ICON-D2
led by 0.22 points [0.08, 0.36]. Since the upgrade UKV is 0.06 points ahead [−0.03, +0.19], an
interval resting on 8 months. ICON-D2's lead is also statistically significant at the 5% level from
October to March, under the second hyperparameter setting, and with the XGBoost model given UKV's
80 m wind. ICON-D2's lead is not statistically significant at the 5% level from April to September,
and with the served 100 m wind the plan specified, where ICON-D2 leads by 0.07 points [−0.03,
+0.17].

**ICON-D2's lead over UKV is largest at the start of each ICON-D2 run.** ICON-D2 runs every 3 hours,
so at each run hour it is served at T+0, like UKV. On those hours ICON-D2 is 0.30 points ahead
[0.19, 0.41]. An hour into the run ICON-D2 is 0.15 points ahead [0.02, 0.27], and 2 hours in the
difference between ICON-D2 and UKV is not statistically significant at the 5% level (ICON-D2 0.04
points behind [−0.13, +0.20]). ICON-D2 at T+0 is ahead both before the upgrade, by 0.34 points
[0.20, 0.48], and after it, by 0.21 points [0.07, 0.33]. Since the upgrade, 2 hours into each
ICON-D2 run, UKV is 0.37 points ahead [0.25, 0.51]. The post-upgrade intervals rest on 8 months, and
every comparison in this paragraph was chosen after the first run.

**The rate at which ICON-D2's lead shrinks may differ nearer ICON-D2's edge.** ICON-D2 takes the
weather at its boundaries from ICON-EU, and the three farms sit about 150 to 200 km east of
ICON-D2's western boundary.

![Figure 8: ICON-D2 leads UKV across the window, but not since UKV's upgrade](assets/wind_icon_d2_against_ukv.svg)

### ICON-EU beats ERA5 at 80 m and at 100 m, mostly from April to September

**An XGBoost model given ICON-EU's 80 m wind beats ERA5 by 0.31 points [0.17, 0.45].** An XGBoost
model given the served 100 m wind, which the plan specified before the first run, beats ERA5 by 0.22
points [0.08, 0.35]. Both XGBoost models are also given the 10 m speed. The 80 m wind improves
ICON-EU by 0.10 points and ICON-D2 by 0.07 points over the served 100 m wind. ICON-EU's advantage
over ERA5 depends on the season, as UKV's does. From April to September ICON-EU is 0.48 points ahead
[0.35, 0.59], and from October to March the difference between ICON-EU and ERA5 is not
statistically significant at the 5% level (ICON-EU 0.13 points ahead [−0.08, +0.35]).

**ICON-EU is 0.13 points behind UKV across the window [0.01, 0.23], a gap that depends on the
XGBoost model's settings and has widened since UKV's upgrade.** The gap is not statistically
significant at the 5% level when the tree is refitted with the second hyperparameter setting, and
when the XGBoost model is given UKV's 80 m wind. The difference between ICON-EU and UKV is not
statistically significant at the 5% level before the upgrade or from October to March. Since the
upgrade UKV is 0.41 points ahead of ICON-EU [0.33, 0.49], with all 5 folds agreeing, though that
interval rests on 8 months and is likely too narrow.

### About half of ICON global's gap to ICON-EU is a pair of steps in its served wind

**ICON global is 0.70 points behind ICON-EU [0.48, 0.93], and about half of that gap is a pair of
steps in ICON global's served wind at one generator.** At that generator, ICON global's wind
relative to ICON-EU's falls by about 15% at 10 m and 8% at 80 m in early June 2025, and rises by
about as much in early June 2026. No other product shows the steps. Telling the XGBoost model which
of the three periods each hour falls in cuts ICON global's deficit to ICON-EU at that generator from
1.39 to 0.44 points, and the deficit across the three generators to 0.38 points [0.27, 0.50]. The
same flag lowers ICON-EU's error by 0.06 points and ERA5's by 0.07, so every comparison with the
flag gives the flag to both products.

![Figure 9: About half of ICON global's gap to ICON-EU is a pair of steps in its served wind at one
generator](assets/wind_icon_global_steps.svg)

**The steps may belong to the archive rather than to ICON global, and their cause is not
identified.** The steps do not come from this study's choice of land cell, because at the generator
with the steps ICON global's nearest cell is the land cell. At the generator where ICON global's
nearest cell is influenced by the sea, the same two dates change how that cell's 10 m speed compares
with the nearest land cell's: about 30% faster before June 2025 and after June 2026, and about 3%
faster between. A change that appears in two neighbouring cells on the same dates and reverses a
year later could come from how the archive serves ICON global's grid cells as well as from the
weather model.

**Against ERA5, ICON global is behind only from October to March and at the generator with the
steps.** Across the window ICON global is 0.38 points behind ERA5 [0.09, 0.70], with only 3 of 5
folds agreeing. From October to March ICON global is 0.85 points behind [0.38, 1.31], and from April
to September the difference is not statistically significant at the 5% level (ICON global 0.03
points ahead [−0.18, +0.21]). At the two generators without the steps, ICON global's difference from
ERA5 is not statistically significant at the 5% level either. With both XGBoost models told when the
steps fall, ICON global is 0.07 points behind ERA5 [−0.09, +0.23].

**ICON global's longer served lead may explain a smaller part of its gap to ICON-EU.** ICON global
runs every 6 hours, so half its hours are served 3 to 5 hours after the run, where ICON-EU's are 0
to 2 hours. On the hours where the two leads are equal ICON global is 0.63 points behind ICON-EU
[0.41, 0.86], against 0.76 points [0.55, 1.00] on the others. This study did not test that
difference, and the split also divides the hours of the day.

**At the two generators without the steps ICON global is still 0.33 and 0.40 points behind
ICON-EU.** ICON-EU is a 6.5 km weather model nested inside ICON global's own 13 km run, so grid
spacing is the most obvious difference between the two, though this study does not isolate it. At
one of those two generators ICON global is read from a land cell further away than its nearest cell,
which may handicap ICON global there.

### ERA5's deficit, year by year

**In an exploratory comparison on January to September of each year, UKV's lead over ERA5 grew
from 0.46 points in 2025 to 0.75 points in 2026, a change of 0.30 points [0.04, 0.58] after
rounding, while ICON-EU's and ICON-D2's leads did not change by a margin statistically significant
at the 5% level.** Each comparison below is exploratory, and each year's interval comes from
resampling that year's months alone, restricted to January to September so a partial 2026 compares
against the same months of the earlier years rather than against their full 12 months. Each year's
interval rests on 9 months, so the intervals are likely too narrow.

**UKV, ICON-EU, and ICON-D2 each beat ERA5 in both years.** UKV beat ERA5 by 0.46 points [0.22,
0.64] in 2025 and by 0.75 points [0.59, 0.93] in 2026. The larger 2026 figure is in step with
UKV's January 2026 upgrade, and lines up with the post-upgrade contrast in [UKV and ICON-D2
describe past wind best](#ukv-and-icon-d2-describe-past-wind-best-of-the-five-products-tested),
0.79 points [0.63, 0.97] since the upgrade against 0.53 points [0.29, 0.72] over the same months a
year earlier. ICON-EU beat ERA5 by 0.49 points [0.33, 0.66] in 2025 and by 0.36 points [0.20,
0.52] in 2026. ICON-D2 beat ERA5 by 0.77 points [0.60, 0.97] in 2025 and by 0.71 points [0.59,
0.83] in 2026. Resampling each year's months independently of the other's, so the test covers the
change itself rather than each year on its own, UKV's lead over ERA5 grew by 0.30 points [0.04,
0.58] from 2025 to 2026; ICON-EU's changed by −0.12 points [−0.36, +0.10]; ICON-D2's by −0.06
points [−0.29, +0.15]. Only UKV's change is statistically significant at the 5% level. August to
December 2024 holds 5 months, too few for an interval even before the January-to-September
restriction, and is left out.

**ICON global's two years are not comparable with each other.** The pair of steps in ICON global's
served wind at one generator, described in [About half of ICON global's gap to ICON-EU is a pair of
steps in its served
wind](#about-half-of-icon-globals-gap-to-icon-eu-is-a-pair-of-steps-in-its-served-wind), falls in
early June 2025 and early June 2026, so each year holds part of the stepped period. In neither year
is ICON global's difference from ERA5 statistically significant at the 5% level.

![Figure 10: On January to September of each year, UKV's lead over ERA5 grew in 2026; ICON-EU's and
ICON-D2's did not](assets/wind_era5_by_year.svg)

### ICON-DREAM-EU does not beat ERA5, and trails ICON-EU

**ICON-DREAM-EU, the German Weather Service's reanalysis, does not beat ERA5, and trails ICON-EU,
the operational European ICON run it is built on, by 0.34 points [0.27, 0.41] at both
hyperparameter settings.** ICON-DREAM-EU is a newer reanalysis than ERA5, built on the same ICON
model family as ICON-EU, ICON-D2 and ICON global, the three operational ICON runs already scored
above. This section asks whether a reanalysis built on a newer model beats ERA5, and whether it
improves on the operational run it draws its physics from.

**ICON-DREAM-EU is read from a gridded download, at DWD's model level 72, about 96 m, rather than
fetched at each generator's coordinates.** Its direction comes from its `U` and `V` wind-component
fields at that level, and its 10 m speed from its own surface field. Each generator reads the
nearest ICON-DREAM-EU grid cell: 1.9 km for Generator W1, 3.4 km for Generator W2, and 1.3 km for
Generator W3.

**Every arm, including the five products already scored above, is refitted on ICON-DREAM-EU's own
row set: 50,041 site-hours from August 2024 to August 2026.** ICON-DREAM-EU's record stops a month
short of the other five products', so a saved loss from earlier on this page cannot be reused
without silently mixing two different row sets. This section therefore rests on its own, shorter
row set than the rest of the page, though the row-build rules — the same power hour, the same
zero-half-hour drop, the same fold boundary at the UKV upgrade — are otherwise identical.

![Figure 11: ICON-DREAM-EU ties ERA5 and beats only ICON global of the six products
tested](assets/wind_icon_dream_leaderboard.svg)

**ICON-DREAM-EU's error is statistically indistinguishable from ERA5's.** Across the window
ICON-DREAM-EU is 0.001 points behind ERA5 [−0.119, +0.131], not statistically significant at the 5%
level. The sign reverses at the second hyperparameter setting, where ICON-DREAM-EU is 0.047 points
ahead of ERA5 [−0.163, +0.080], also not statistically significant.

**ICON-DREAM-EU trails ICON-EU by 0.34 points at both hyperparameter settings.** At the primary
setting the gap is 0.340 points [0.266, 0.405], and at the second setting 0.338 points [0.282,
0.391]; both are statistically significant at the 5% level, with every fold agreeing.

![Figure 12: ICON-DREAM-EU does not beat ERA5, and trails ICON-EU by 0.34
points](assets/wind_icon_dream_planned_contrasts.svg)

**The rest of this section is exploratory: chosen after the results were seen, not named in the
plan.** Giving the XGBoost model two more height levels — level 73, about 42 m, and level 71, about
167 m — alongside level 72 improves on the primary arm by 0.044 points [0.016, 0.074], statistically
significant at the 5% level. That arm carries two more feature columns than its reference, and this
repository's own measurement puts the effect of extra columns alone at up to about 0.4% of mean
absolute error even at this study's column-subsampling setting, so part of this gain may belong to
the extra columns rather than to the extra heights.

**Direction adds skill for ICON-DREAM-EU too, in line with every other product tested.** Giving the
XGBoost model ICON-DREAM-EU's direction and 10 m speed alongside its hub-height speed, rather than
the hub-height speed alone, cuts its error by 0.305 points [0.204, 0.415]. The same addition cuts
ERA5's error by 0.616 points [0.454, 0.789], UKV's by 0.388 points [0.280, 0.501], ICON-D2's by
0.239 points [0.137, 0.344], ICON-EU's by 0.367 points [0.245, 0.496], and ICON global's by 0.541
points [0.377, 0.744].

**ICON-DREAM-EU's gap to ICON-EU is consistent across all three generators, but its comparison with
ERA5 is not.** Against ICON-EU, ICON-DREAM-EU trails by 0.403 points [0.324, 0.484] at Generator
W1, 0.293 points [0.144, 0.438] at Generator W2, and 0.326 points [0.160, 0.462] at Generator W3,
each statistically significant at the 5% level. Against ERA5, ICON-DREAM-EU trails by 0.164 points
[0.049, 0.285] at Generator W1, a gap statistically significant at the 5% level, but neither
Generator W2 (0.225 points ahead [−0.002, +0.449], not statistically significant) nor Generator W3
(0.074 points behind [−0.083, +0.238], not statistically significant) shows a difference
distinguishable from zero.

**Neither planned contrast's size changed from 2025 to 2026, in a comparison restricted to
January to August of each year so a partial 2026 compares against the same months of 2025.** Against
ERA5, the difference was −0.075 points [−0.273, +0.097] in 2025 and −0.036 points [−0.217, +0.132]
in 2026, neither statistically significant, a change of +0.039 points [−0.212, +0.297], also not
statistically significant. Against ICON-EU, the gap was 0.416 points [0.312, 0.498] in 2025 and
0.376 points [0.309, 0.430] in 2026, both statistically significant, a change of −0.041 points
[−0.145, +0.080], not statistically significant.

**The pre-fit checks passed before any arm was fitted.** ICON-DREAM-EU's speed from its `U` and `V`
wind components agrees with its own served scalar speed almost exactly: a median absolute
difference of 0.0003 m/s at level 72 and 0.0002 m/s at 10 m. ICON-DREAM-EU's direction disagrees
with ERA5's own 100 m direction by 10.4 degrees mean absolute, consistent across all three
generators (10.1 to 10.5 degrees), the size of disagreement expected between two different products
rather than a sign of a wrong meteorological convention. The hour-to-hour correlation between
ICON-DREAM-EU's and ERA5's speed changes peaks at zero offset (0.436), clearly higher than at plus
or minus one hour (0.345 and 0.322) or plus or minus two hours (0.194 and 0.181), confirming
ICON-DREAM-EU's served value carries no timestamp offset relative to ERA5.

## What to use

**These recommendations rest on three wind farms in one flat part of Lincolnshire, over 2 years.**

- **Historical features in the live service: UKV, with ICON-D2 an alternative where ICON-D2
  reaches.** UKV covers all of Great Britain and has been 0.41 points ahead of ICON-EU [0.33, 0.49]
  since its upgrade. ICON-D2's average advantage over UKV has not been statistically significant at
  the 5% level since that upgrade, so reading ICON-D2 inside its domain adds a seam near ICON-D2's
  western edge for no measured gain. ERA5 arrives about 5 days late and cannot supply the last few
  hours at run time. Every figure here scores the archive's freshest run for each hour. At run time
  the last few hours come from an older run, at a longer lead than any scored here.
- **Training history: ICON-D2 where ICON-D2 covers, from November 2022, with ICON-EU or ERA5
  elsewhere. This study scores none of them before August 2024.** ICON-D2 is the only product of the
  five that beats ERA5 in both halves of the year. ICON-EU's advantage over ERA5 is statistically
  significant at the 5% level only from April to September, so west of ICON-D2's edge this study
  cannot choose between ICON-EU and ERA5 for October to March. ERA5 is a reanalysis built with one
  fixed version of its weather model, so it has no weather-model upgrade by design, and its record
  goes back to 1940. UKV is not recommended, because Open-Meteo's archive of its hub-height wind
  starts only in August 2024. A training history that switches from ERA5 to an ICON product part-way
  through has to tell the forecasting model which product each hour comes from, as this study tells
  the XGBoost model which side of the UKV upgrade each hour falls on. Open-Meteo's ICON archives
  before August 2024 have not been screened for steps like the pair in ICON global's served wind.
- **Capacity estimation and disaggregation: no recommendation.** Capacity estimation infers a farm's
  size from how its output tracks the wind, and disaggregation separates hidden generation from
  demand at a substation. Both have to read a product's wind without a fit to the farm's own metered
  output, and every figure here comes after such a fit.

**Giving an XGBoost model all five products at once beats giving it UKV alone, with UKV's
neighbouring hours, by 0.48 points [0.40, 0.56], a post hoc comparison written up in [Does blending
weather products beat the best single weather
product?](blending-weather-products.md#wind-a-blend-beats-ukv-given-its-neighbouring-hours)**

## Limitations

- **Terrain.** [Issue #826](https://github.com/openclimatefix/nged-substation-forecast/issues/826)
  expected a finer grid to matter most for wind, because terrain shapes hub-height wind. The three
  generators are onshore in flat country, and two of them share one ERA5 grid cell, so this study
  cannot speak to terrain, offshore wind, or any other region.
- **The intervals describe these three farms only.** The intervals resample months, not farms, so
  they say nothing about how a fourth farm would rank the products.
- **Raw accuracy.** The products' mean speeds at the served 100 m differ by up to 0.7 m/s, and the
  ICON products' served 100 m value is Open-Meteo's rescaling of their 120 m wind. The per-generator
  XGBoost model absorbs such differences.
- **Any period before August 2024,** and behaviour near the turbines' cut-in speed, which the zero
  rule under-samples.
- **Turbine availability.** One generator's output swings for months at a time with availability no
  product can see, which weakens every contrast there.
- **The figures rest on the capacity table as rebuilt in September 2026.** Each generator's capacity
  is its 99th percentile of output from the `effective_capacity` table, so a rebuilt table would
  move every figure.
- **ICON-DREAM-EU's own row set.** [ICON-DREAM-EU does not beat ERA5, and trails
  ICON-EU](#icon-dream-eu-does-not-beat-era5-and-trails-icon-eu) refits every arm, including the
  five products above, on ICON-DREAM-EU's own, shorter row set (August 2024 to August 2026, a
  month short of the rest of the page), so its numbers are not directly comparable to the rest of
  this page's.

## Reproducing the figures

```bash
uv run python studies/beam_diffuse_split/fetch_wind_point.py
uv run python studies/beam_diffuse_split/wind_products.py
uv run python studies/beam_diffuse_split/wind_products.py --era5-by-year
uv run python studies/beam_diffuse_split/wind_product_charts.py
uv run python studies/beam_diffuse_split/wind_icon_dream.py
uv run python studies/beam_diffuse_split/wind_icon_dream_charts.py
```

The report lands in `data/studies/beam_diffuse_split/beam_diffuse_wind_products/report.md`.
`wind_products.py --fit-missing` keeps the losses already saved and fits only the XGBoost models
they lack. The report also prints the check with the solar study's power hour, the run that keeps
the zero hours, the step ratios, and the distances between the farms and to ICON-D2's edge. `uv run
python studies/beam_diffuse_split/check_served_wind.py` writes `served_wind_checks.md` beside it:
the grid-cell check, the 100 m rescaling, and when the ICON 80 m wind starts. The hour-to-hour jump
diagnostics behind the served leads were one-off checks during review, and are not in either report.
`wind_products.py --era5-by-year` reads the saved losses, fits nothing, and writes Figure 10's table
to `data/studies/beam_diffuse_split/past_weather_v2/wind/era5_by_year.md`.
`wind_icon_dream.py`'s report lands separately, in
`data/studies/beam_diffuse_split/past_weather_v2/wind_icon_dream/report.md`; `--report-only`
rebuilds it from a saved `losses.parquet` alone, fitting nothing.
