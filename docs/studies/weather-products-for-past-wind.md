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
chosen after the results were seen. The two gaps are 6% and 8% of ERA5's error. A sixth product,
ICON-DREAM-EU, DWD's reanalysis, scored on a slightly shorter row set, is statistically
indistinguishable from ERA5 and trails ICON-EU by 0.34 points [0.27, 0.41].

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
- **[ICON-DREAM-EU](https://doi.org/10.5676/dwd/icon-dream_v1), DWD's ICON-based reanalysis, does
  not beat ERA5, and trails ICON-EU, DWD's operational ICON model over Europe, by 0.34 points [0.27,
  0.41] at the primary hyperparameter setting and 0.34 points [0.28, 0.39] at the second.**
  ICON-DREAM-EU's hourly wind, read from DWD rather than Open-Meteo, is a 1 to 3 hour forecast,
  never an analysis. In an exploratory comparison restricted to hours where its served lead matches
  ICON-EU's, the gap narrows to 0.28 points [0.20, 0.36]. See [ICON-DREAM-EU does not beat ERA5, and
  trails ICON-EU](#icon-dream-eu-does-not-beat-era5-and-trails-icon-eu).
- **On 43,555 farm-hours from December 2024 to September 2026 at three wind farms, UKV beats
  ECMWF's HRES and ENS day 0, and HRES beats ERA5, in the three planned contrasts.** HRES is the
  high-resolution forecast of ECMWF's Integrated Forecasting System (IFS), and ENS day 0 is the
  mean of the 51 members of ECMWF's ensemble forecast (ENS) for the hours 00 to 23 UTC of the 00 UTC
  run's own day. Each difference below is the first-named product's error minus the second's, so a
  positive difference means the first-named product's error is larger. HRES minus UKV is +0.20
  points [+0.06, +0.33], ENS day 0 minus UKV is +0.41 points [+0.25, +0.58], and HRES minus ERA5 is
  −0.27 points [−0.40, −0.12]. See [UKV describes past wind better than ECMWF's HRES and ENS day 0
  on these rows](#ukv-describes-past-wind-better-than-ecmwfs-hres-and-ens-day-0-on-these-rows).
- **The HRES result against ERA5 depends on how the folds treat ECMWF's IFS Cycle 49r1, and holds
  mainly at one of the three farms, and ENS day 0 against ERA5 is unresolved.** ENS day 0 minus
  ERA5 is −0.07 points [−0.19, +0.06], an interval that bounds the difference and does not show
  that the two products are equal. The ENS horizons page's figure for the same contrast, +0.170
  points [+0.021, +0.322], is not like for like. See [UKV describes past wind better than ECMWF's
  HRES and ENS day 0 on these rows](#ukv-describes-past-wind-better-than-ecmwfs-hres-and-ens-day-0-on-these-rows).

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
trails ICON-EU](#icon-dream-eu-does-not-beat-era5-and-trails-icon-eu), on its own row set. The
table below includes ICON-DREAM-EU's own grid, lead, and history alongside the five, for
reference.

| Product | What it is | Grid spacing, native and as served | Wind heights served | Served lead | Covers all of Great Britain? | Start of the hub-height wind archive read here | Available after |
|---|---|---|---|---|---|---|---|
| ERA5 | ECMWF reanalysis, a consistent record back to 1940 | about 31 km; served on a 0.25° grid | 10 m and 100 m | an hourly analysis | yes | 1940 | about 5 days |
| UKV | Met Office model for the UK | 1.5 km over the UK, coarsening to 4 km at the domain's edges; served at 2 km | 100 m (Open-Meteo also serves 50 m and 80 m) | T+0, the analysis | yes | August 2024 | about 4 hours |
| ICON-D2 | DWD model for Germany and neighbouring countries | 2.2 km; served at about 2 km | 80 m and 120 m | 0 to 2 hours | no: its western edge runs from about 2°W on the south coast to about 2.5°W in the Midlands | November 2022 | about 1.5 hours |
| ICON-EU | DWD model for Europe, nested inside ICON global | 6.5 km; served at about 7 km | 80 m and 120 m | 0 to 2 hours | yes | November 2022 | about 3.5 hours |
| ICON global | DWD global model | 13 km; served at about 11 km | 80 m and 120 m | 0 to 5 hours | yes | November 2022 | about 3.5 hours |
| ICON-DREAM-EU | DWD reanalysis run of the ICON model, over Europe, with its own data assimilation | 6.5 km; served by DWD on ICON's native triangular grid, not by Open-Meteo; read at the nearest cell | every model level (levels 65 to 74 downloaded here); read at level 72 (about 96 m) and 10 m, with levels 71 and 73 added in one exploratory arm | 1 to 3 hours | yes | January 2010 | after each month ends, a month at a time; DWD's readme states a delay of about 2 to 3 months, and August 2026 was on DWD's server by 23 September 2026 |
| ECMWF-IFS-HRES | ECMWF's single high-resolution forecast, read as Open-Meteo's `ecmwf_ifs` | about 9 km native (O1280 grid); the served grid before 2025 is not established | 10 m and 100 m | 1 to 12 hours before 1 October 2025 and 0 to 5 hours from it, inferred from where hour-to-hour jumps fall; Open-Meteo does not document it | yes | 1 December 2024 (the first whole month after IFS Cycle 49r1); the archive's own start is not established here | not established |
| ECMWF ENS day 0 | mean of the 51 members of ECMWF's ensemble forecast, hours 00 to 23 UTC of the 00 UTC run's own day | about 9 km native (O1280 grid) since IFS Cycle 48r1 on 27 June 2023; served on a 0.25° grid from ECMWF's open data, in 3-hourly steps rebuilt to hourly here | 10 m and 100 m | 0 to 23 hours | yes | April 2024 (Dynamical.org's archive of the 00 UTC run starts on 1 April 2024); this study reads it from 1 December 2024 | about 09:00 UTC for a 00 UTC run, this repository's assumption; ECMWF disseminates ENS day 0 at about 06:40 UTC |

**The table's two ECMWF rows describe different products with different sources of limits, so the
limits are attributed to the source that sets them.** ENS runs four times a day. The single 00 UTC
run, the 3-hourly steps that follow ECMWF's hourly steps to 90 hours, and the roughly 09:00 UTC read
time belong to ECMWF's open-data subset, Dynamical.org's archive, and this repository's
`NWP_PUBLICATION_DELAY_HOURS` assumption, not to ENS itself. HRES is read through Open-Meteo's
Previous Runs interface at each farm's point, with Open-Meteo's default land-cell selection, where
ERA5 is read at the nearest cell. Open-Meteo's documentation says only that each run's first few
hours are stitched into a continuous hourly series, so HRES's served lead is inferred, and the
source of HRES's archive changed on 1 October 2025, when ECMWF moved to full open data. IFS Cycle
49r1 went live on 12 November 2024, and Cycle 50r1 with the 06 UTC run of 12 May 2026.

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
  ICON-D2 against ICON-EU. The ICON-DREAM-EU section has two planned contrasts of its own,
  ICON-DREAM-EU against ERA5 and ICON-DREAM-EU against ICON-EU, written into that section's plan
  before ICON-DREAM-EU was fitted but after the five products above had been scored. The ECMWF
  section has three planned contrasts of its own, HRES against UKV, ENS day 0 against UKV, and HRES
  against ERA5, written into a plan committed before that section's first fit. A contrast is
  the difference between two products' errors on the same hours. A comparison is planned when it
  was written into the study plan before any result existed; every other figure is exploratory,
  chosen or added after results were seen. The distinction
  matters because with many comparisons, about 1 in 20 exploratory rows reaches significance at the
  5% level by chance, so an exploratory result is a lead to follow up rather than a finding. A chart
  holding both kinds marks each planned row "(planned)". A chart whose rows are all one kind says so
  once, in its subtitle. Every contrast is also rerun with a second hyperparameter setting for the
  tree. The plan written before the run gave the XGBoost model each product's served 100 m wind.
  After the first run, each ICON product was switched to its 80 m wind. The results with the served
  100 m wind are reported alongside and change no ranking.

### How ECMWF's ENS and HRES were added

**The ECMWF section rests on a plan committed before its first fit, and on choices that differ from
the original request.** The plan, `plans/ens-hres-past-wind.md`, holds the three planned contrasts.
It was committed as `9bb0a6f7`, and its last revision before any fit as `715681a7`. The script that
fitted every model, `ens_hres_past_wind.py`, was committed as `83dbbad3` before its first fit, and
`script_commit.txt` beside the results records that commit. The section differs from the original
request in two ways. HRES is read from Open-Meteo's Previous Runs file, because that file carries
wind direction, where the 9 km grid file holds wind speed only, in kilometres per hour. The grid
file serves one cross-check: at each farm, one of the
five nearest grid points reproduced the Previous Runs speeds on every hour (71,712 farm-hours), which
shows that two Open-Meteo downloads agree and does not show that the grid or the served lead is
right. ENS day 0 is read from the saved inputs of the [ENS forecast horizons
study](ens-forecast-horizons.md), because the `T+3` band the request named covers leads of 3 to 21
hours only, and its download script is not in this repository.

**Wind is an instantaneous value, so the section keeps direction out of every average and
interpolation.** Each ENS member's speed and direction become eastward and northward components at
each 3-hourly step, the components are interpolated linearly to hourly values, and the hourly speed
and direction come from the interpolated vector. The mean of the 51 members' speeds is the ENS
speed, and the direction of the mean wind vector is the ENS direction. The horizons study's own
combination interpolates direction in degrees, which crosses north the wrong way on about 1% of
day-0 hours, and runs here as an exploratory comparison. Each XGBoost model in this section is given
the page's seven columns: the hour of the day, the day of the year, an era code, the hub-height
speed, that height's direction as sine and cosine, and the 10 m speed. HRES, ENS day 0, ERA5, and
UKV are read at 100 m.

**The row set starts on 1 December 2024, and the folds are cut inside three eras.** December 2024 is
the first whole month after IFS Cycle 49r1, and HRES's served grid before 2025 is not established.
Every product is refitted on the same 43,555 farm-hours (W1 14,489, W2 14,994, W3 14,072) in 22
calendar months. The three eras are before 1 October 2025, when the source of HRES's archive
changed; 1 October 2025 to 20 January 2026; and from 1 February 2026, after the UKV upgrade. The
era code takes three values for every XGBoost model. Cutting folds inside each era ranks each era's
months separately, so one calendar month can be held out of every era at once and leave no training
data for that season ([issue #868](https://github.com/openclimatefix/nged-substation-forecast/issues/868)).
A check before any fit found six such cases across the three farms, for July and September, so the
third era's fold numbers are rotated by 2, which leaves no failing case. October and November occur
in one year only, 2025, so whichever fold holds either month out has no training row for that
calendar month under any fold design. The second hyperparameter setting is rerun for HRES, ENS day
0, UKV, and ERA5.

**Everything after the first results is exploratory.** After the first results and a scientific
review, five analyses were added: the same products refitted on the longer row set from 12 August
2024, under the ENS horizons page's folds and under an extra era cut; a table of six fold designs;
a table of the hour-to-hour jumps that show HRES's served lead; a split of the scored hours by label
hour; and corrections to the wording of product facts. The plan lists each of them before it was
fitted, and the results below label them.

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

**ICON-DREAM-EU, DWD's ICON-based reanalysis, does not beat ERA5, and trails ICON-EU, DWD's
operational ICON model over Europe at the same 6.5 km grid spacing, by 0.34 points [0.27, 0.41] at
the primary hyperparameter setting and 0.34 points [0.28, 0.39] at the second.** ICON-DREAM-EU does
not consume ICON-EU's output: it is DWD's own reanalysis run of the ICON model family, with its own
data assimilation, nested inside a 13 km global run. Like ERA5, ICON-DREAM-EU runs one fixed version
of its weather model over its whole record so that the record is consistent over time, while
ICON-EU is DWD's operational run. This section's two planned contrasts ask whether a reanalysis
built on a newer model beats ERA5, and whether it improves on ICON-EU, the operational run sharing
its grid spacing.

**ICON-DREAM-EU is read from a gridded download, at DWD's model level 72, about 96 m, rather than
fetched at each generator's coordinates.** Its direction comes from its `U` and `V` wind-component
fields at that level, and its 10 m speed from its own surface field. Each generator reads the
nearest ICON-DREAM-EU grid cell: 1.9 km for Generator W1, 3.4 km for Generator W2, and 1.3 km for
Generator W3.

**ICON-DREAM-EU's hourly wind is a 1 to 3 hour forecast, never an analysis.** DWD's readme does not
say how the hourly series is built, but the files read here hold short forecasts from runs every 3
hours. At 00, 03, 06 UTC and every third hour after, ICON-EU is served here as a T+0 analysis, but
ICON-DREAM-EU at the same hour is a 3 hour forecast from the previous run, the longest lead in its
own cycle. The padding hours that cfgrib, the library that decodes DWD's GRIB files, fills in at
each month boundary establish that ICON-DREAM-EU's hourly series is a forecast, not an analysis:
they fall at 22:00 and 23:00 before the month and 00:00 of the next one, which fits a run every 3
hours starting from 00 UTC, giving steps 1 to 3. A second, independent check agrees: the mean
absolute hour-to-hour change in level 72's speed, over every cell in the download, is 0.604 m/s at
the hour a new run's first step arrives, against 0.561 and 0.562 m/s within a run, because a fresh
run, started from a new analysis, replaces the previous run's 3-hour forecast rather than continuing
it.

**Every arm — one XGBoost model fitted on one product's set of feature columns — including the five
products already scored above, is refitted on ICON-DREAM-EU's own row set: 50,041 generator-hours
from August 2024 to August 2026.** ICON-DREAM-EU's record stops 10 days short of the other five
products', ending 31 August 2026 against their 10 September 2026, so every arm is refitted on
ICON-DREAM-EU's months, giving every arm the same folds, rather than setting ICON-DREAM-EU's fits
beside fits whose folds were cut over a longer window. This section therefore rests on its own,
shorter row set than the rest of the page, though the row-build rules — the same power hour, the
same zero-half-hour drop, the same fold boundary at the UKV upgrade — are otherwise identical.

**In exploratory contrasts, the five original products still rank much as [UKV and ICON-D2 describe
past wind best](#ukv-and-icon-d2-describe-past-wind-best-of-the-five-products-tested) found, on this
shorter row set, with one exception.** ICON-EU still beats ERA5, by 0.340 points [0.193, 0.479],
close to the 0.31 points found there. ICON-D2 still beats ICON-EU, by 0.240 points [0.164, 0.317],
close to the 0.26 points found there. UKV's small lead over ICON-EU found there (0.13 points [0.01,
0.23], statistically significant at the 5% level) shrinks here to 0.086 points [−0.028, +0.193], not
statistically significant at the 5% level. `wind_products.py`'s own published fits, not refit,
scored on these same 50,041 rows, give 0.119 points [0.000, 0.228] for ICON-EU against UKV, −0.316
points [−0.451, −0.171] for ICON-EU against ERA5, and −0.263 points [−0.334, −0.183] for ICON-D2
against ICON-EU. Refitting on this row set's own folds, rather than reusing the published fit, moves
each of those three contrasts by 0.02 to 0.03 points, and moves the ICON-EU-against-UKV contrast
across the significance threshold. The month-resampled intervals do not capture that fold-layout
variation, so an effect of a few hundredths of a point on this page — the three-level arm's gain
below, the 0.06 attributed to the lead mismatch, and the column-count caveat — should be read with
that in mind.

**Every check below passed before any arm was fitted.** ICON-DREAM-EU's speed from its `U` and `V`
wind components agrees with its own served scalar speed almost exactly: a median absolute
difference of 0.0003 m/s at level 72 and 0.0002 m/s at 10 m. ICON-DREAM-EU's direction disagrees
with ERA5's own 100 m direction by 10.4 degrees mean absolute, consistent across all three
generators (10.1 to 10.5 degrees). As a same-method baseline, over the same row set ICON-EU
disagrees with ERA5's direction by 9.2 degrees, ICON-D2 by 10.5 degrees, and UKV by 10.6 degrees, so
ICON-DREAM-EU's 10.4 degrees sits inside the range already seen between products already trusted,
not a sign of a wrong meteorological convention. The hour-to-hour correlation between
ICON-DREAM-EU's and ERA5's speed changes peaks at zero offset (0.436), clearly higher than at plus
or minus one hour (0.345 and 0.322) or plus or minus two hours (0.194 and 0.181), confirming
ICON-DREAM-EU's served value carries no whole-hour offset relative to ERA5.

![Figure 11: ICON-DREAM-EU is statistically indistinguishable from ERA5, and beats only ICON global
of the other five products](assets/wind_icon_dream_leaderboard.svg)

**ICON-DREAM-EU's error is statistically indistinguishable from ERA5's.** Across the window
ICON-DREAM-EU is 0.001 points behind ERA5 [−0.119, +0.131], not statistically significant at the 5%
level. At the second hyperparameter setting ICON-DREAM-EU is 0.047 points ahead of ERA5 [−0.080,
+0.163], also not statistically significant; both estimates sit close to zero. In an exploratory
paired contrast, ICON-DREAM-EU beats ICON global, the lowest-ranked of the other five products, by
0.351 points [0.137, 0.592].

**Over the window the two pages share, neither finds ICON-DREAM-EU statistically distinguishable
from ERA5, and both find it behind ICON-EU.** The [past-solar
page](weather-products-for-past-solar.md#icon-dream-eu-beats-era5-but-not-the-icon-weather-models)
finds ICON-DREAM-EU ahead of ERA5 by 0.32 points [0.12, 0.52] over its longer window, back to
September 2019, but by 0.16 points [−0.09, +0.38] since August 2024, the window shared with the
past-solar page, not statistically significant at the 5% level. On both pages ICON-DREAM-EU trails
ICON-EU: by 0.38 points for solar, and, as the next section below finds, by 0.28 points [0.20, 0.36]
here.

**ICON-DREAM-EU trails ICON-EU by 0.34 points at both hyperparameter settings.** At the primary
setting the gap is 0.340 points [0.266, 0.405], and at the second setting 0.338 points [0.282,
0.391]; both are statistically significant at the 5% level, with every fold agreeing.

![Figure 12: ICON-DREAM-EU does not beat ERA5, and trails ICON-EU by 0.34
points](assets/wind_icon_dream_planned_contrasts.svg)

**The rest of this section is exploratory: chosen after the results were seen, not named in the
plan.**

**Part of ICON-DREAM-EU's gap to ICON-EU comes from its 3-hour forecasts: dropping the hours where
ICON-DREAM-EU is at step 3 and ICON-EU at T+0 narrows the gap to 0.28 points [0.20, 0.36].**
ICON-EU's own error against ERA5 is flat by lead: −0.341 points at T+0, −0.346 points at 1 hour, and
−0.332 points at 2 hours. ICON-DREAM-EU's error against ERA5 is not: it moves from −0.108 points at
step 1 to +0.121 points at step 3 (the next paragraph below sets out every step). So restricting to
equal-lead hours narrows the gap between ICON-DREAM-EU and ICON-EU by dropping ICON-DREAM-EU's
worst step, not by dropping one of ICON-EU's better leads. On the two hours in three where the two
products' served leads match (33,344 rows), ICON-DREAM-EU trails ICON-EU by 0.279 points [0.197,
0.356] at the primary setting and 0.271 points [0.208, 0.333] at the second, against 0.340 and 0.338
points on every hour. So dropping the step-3 hours removes about 0.06 of the 0.34-point gap (no
interval computed for that difference, and a change of that size sits within the refit noise
described above), and ICON-DREAM-EU still trails ICON-EU at every matched lead. On the same
equal-lead hours ICON-DREAM-EU is 0.059 points ahead of ERA5 [−0.091, +0.194] at the primary setting
and 0.113 points ahead [−0.036, +0.246] at the second, neither statistically significant at the 5%
level.

**ICON-DREAM-EU's own gap grows with its served lead.** Split by step, ICON-DREAM-EU trails
ICON-EU by 0.238 points [0.151, 0.325] at step 1, 0.321 points [0.226, 0.409] at step 2, and 0.462
points [0.371, 0.547] at step 3, each statistically significant at the 5% level. Against ERA5 the
pattern is similar: ICON-DREAM-EU is 0.108 points ahead [−0.055, +0.243] at step 1, not
statistically significant, 0.011 points ahead [−0.144, +0.162] at step 2, also not statistically
significant, and 0.121 points behind [+0.009, +0.230] at step 3, statistically significant at the
5% level. Step 3 is ICON-DREAM-EU's longest lead in its own cycle. Against ICON-EU the step-1 and
step-3 intervals do not overlap; against ERA5 they do, and no test of the step-1-to-step-3 change
was run, so the trend against ERA5 is suggestive only.

**Giving the XGBoost model two more ICON-DREAM-EU heights does not change either planned answer: the
three-level arm still trails ICON-EU, and still does not beat ERA5.** Giving the XGBoost model two
more height levels — about 42 m and 167 m — alongside level 72 improves on the primary, single-level
arm by 0.044 points [0.016, 0.074], statistically significant at the 5% level. Compared directly,
the three-level arm still trails ICON-EU, by 0.296 points [0.222, 0.365], statistically significant
at the 5% level with every fold agreeing, and is 0.043 points ahead of ERA5 [−0.079, +0.157], not
statistically significant, with only 2 of 5 folds agreeing.

**Part of the three-level arm's apparent gain may belong to its extra feature columns rather than
its extra heights.** Every arm here that includes the three-level arm carries two more feature
columns than the arm it is compared against. This repository's own measurement puts the effect of
extra columns alone at up to about 0.4% of mean absolute error even at this study's
column-subsampling setting — about 0.03 points on this page's error scale, the same size as the
refit noise described above. So part of the three-level arm's apparent gain, in every comparison
above, may belong to the extra columns rather than to the extra heights, and the benefit of the
extra heights themselves is not established.

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

**At Generator W2, ICON-DREAM-EU's lead over ERA5 is not statistically significant at the 5% level,
and it is no larger than the shift every leading product shows there.** At Generator W2 every one of
the three leading original products beats ERA5 by more than it does across all three generators on
this section's row set: ICON-EU by 0.518 points, UKV by 0.775 points, and ICON-D2 by 0.803 points,
each statistically significant at the 5% level, against 0.340 points [0.193, 0.479], 0.425 points
[0.226, 0.600], and 0.579 points [0.405, 0.742] across all three generators on this section's row
set.

**Neither planned contrast's size changed by a margin statistically significant at the 5% level
from 2025 to 2026, in a comparison restricted to January to August of each year so a partial 2026
compares against the same months of 2025.** Against ERA5, ICON-DREAM-EU was 0.075 points ahead
[−0.097, +0.273] in 2025 and 0.036 points ahead [−0.132, +0.217] in 2026 (signs flipped so ahead is
positive), a change of +0.039 points [−0.212, +0.297], not statistically significant. Against
ICON-EU, the gap was 0.416 points [0.312, 0.498] in 2025 and 0.376 points [0.309, 0.430] in 2026, both
statistically significant, a change of −0.041 points [−0.145, +0.080], not statistically
significant.

### UKV describes past wind better than ECMWF's HRES and ENS day 0 on these rows

**UKV beats HRES by 0.20 points and ENS day 0 by 0.41 points, and HRES beats ERA5 by 0.27 points,
on 43,555 farm-hours from December 2024, in the section's three planned contrasts.** A difference
below is the first-named product's error minus the second's, in points of capacity, so a positive
difference means the first-named product's error is larger. HRES minus UKV is +0.20 points [+0.06,
+0.33], with 4 of 5 folds agreeing. ENS day 0 minus UKV is +0.41 points [+0.25, +0.58], with all 5
folds agreeing. HRES minus ERA5 is −0.27 points [−0.40, −0.12], with all 5 folds agreeing. The
three contrasts share their 22 calendar months, and three wind farms are few independent sites, so
each interval describes these farms and this window only.

**Each product's own error, in order, is ICON-D2 6.67%, UKV 6.88%, ICON-EU 7.04%, HRES 7.08%, ENS
day 0 7.29%, ERA5 7.35%, and ICON global 7.68% of capacity.** Every product is refitted on the same
43,555 farm-hours, so these figures are not the page's earlier ones. Own intervals overlap widely,
for example HRES's [6.28, 8.04] and ENS day 0's [6.52, 8.21], because every product's error rises
and falls with the month. The paired differences are the test. No paired contrast between ICON-D2
and either ECMWF product was computed, so this section ranks ICON-D2 against them by own error
alone. In exploratory contrasts on these rows, UKV minus ERA5 is −0.48 points [−0.64, −0.29], and
ENS day 0 minus HRES is +0.21 points [+0.07, +0.33].

![Figure 13: UKV beats ECMWF's HRES by 0.20 points and ENS day 0 by 0.41, and HRES beats ERA5 by
0.27](assets/ens_hres_wind_leaderboard.svg)

**ENS day 0 minus ERA5 is −0.07 points [−0.19, +0.06], so this section does not resolve whether
ENS day 0 differs from ERA5.** The interval bounds the difference. An ENS error from 0.19 points
below ERA5's to 0.06 points above it is not excluded, and the interval does not show that the two
are equal. The contrast is exploratory.

**The XGBoost models given HRES's or ENS day 0's wind follow the shape of measured power at every
farm, in a windy, a variable, and a calm week.** Figure 14 plots out-of-fold predictions against
measured power. The weeks are chosen by the rule the [first results section](#the-xgboost-models-work)
uses, from measured power alone: the windiest week fell in December 2024, the most variable in
February 2025, and the calmest ran from July into August 2026. At Generator W3 both predictions run
above measured power in the windiest week and below it in the variable week, which fits the turbine
availability no product records. Figure 14 shows days 1 to 7 with no calendar date, for the reason
given there.

![Figure 14: XGBoost models given HRES's or ENS day 0's wind follow the shape of measured power at
every farm, in a windy, a variable, and a calm week](assets/ens_hres_wind_models_work.svg)

**ICON-D2 has the lowest own error at each of the three farms, and HRES is ahead of ENS day 0 at
each.** HRES's error is 5.94% against ENS day 0's 6.06% at Generator W1, 6.96% against 7.18% at
Generator W2, and 8.38% against 8.66% at Generator W3, and ICON-D2's is 5.54%, 6.58%, and 7.92%.
Figure 15 leaves ICON global out, as Figure 5 does, because its dots would show which generator
carries the steps in ICON global's served wind.

![Figure 15: ICON-D2 has the lowest error at each of the three farms, and ECMWF HRES is ahead of
ENS day 0 at each](assets/ens_hres_wind_per_farm_error.svg)

**The three planned contrasts keep their sign and stay statistically significant at the 5% level
under six fold designs and at the second hyperparameter setting.** The six designs, added after the
first results and so exploratory, are the study's design, the same losses without May 2026, three
eras with no fold rotation, two UKV eras as the rest of the page, the study's folds with an era
code of two values, and an extra era cut at IFS Cycle 50r1 with May 2026 dropped. Across them, HRES
minus UKV runs from +0.15 to +0.21 points, ENS day 0 minus UKV from +0.34 to +0.43, and HRES minus
ERA5 from −0.24 to −0.29. The smallest lower bound is HRES minus UKV's under the two-valued era
code, +0.15 points [+0.02, +0.28]. At the second hyperparameter setting the three differences are
+0.21 points [+0.07, +0.36], +0.43 points [+0.28, +0.58], and −0.29 points [−0.42, −0.13]. Intervals
adjusted for the three contrasts, at the 98.33% level, are [+0.03, +0.36], [+0.21, +0.61], and
[−0.42, −0.09], and none includes zero.

![Figure 16: Each planned contrast keeps its sign and stays statistically significant at the 5%
level under all six fold designs](assets/ens_hres_wind_robustness.svg)

**The ENS horizons page's figure for ENS day 0 against ERA5 is not like for like with this
section's, because the two differ in row set and in how the folds treat IFS Cycle 49r1.** The
horizons page reports ENS day 0 minus ERA5 as +0.170 points [+0.021, +0.322] on 50,268 rows from 12
August 2024. This section finds −0.07 points [−0.19, +0.06] on 43,555 rows from 1 December 2024.
Refitting on the page's own rows from 12 August 2024 (50,734 farm-hours in 26 calendar months) with
the horizons page's folds, which do not cut at IFS Cycle 49r1, gives +0.16 points [+0.01, +0.31] for
this study's ENS wind, and +0.17 points [+0.03, +0.32] for the horizons page's own combination, so
the +0.170 is reproduced. Scoring only the rows from 1 December 2024 under those folds gives +0.10
points [−0.05, +0.25]. With one extra era cut at 1 December 2024, ENS day 0 minus ERA5 is −0.03
points [−0.18, +0.11] on all rows and −0.05 points [−0.21, +0.12] from December. ENS day 0 trails
ERA5 by a margin statistically significant at the 5% level only when the folds ignore IFS Cycle 49r1
and August to November 2024 is scored. These refits are exploratory, added after the first results.
[Issue #892](https://github.com/openclimatefix/nged-substation-forecast/issues/892) tracks the
horizons page's treatment of the cycle change.

**HRES's lead over ERA5 depends on the same choice.** HRES minus ERA5 is +0.06 points [−0.14,
+0.27] on all rows from 12 August 2024 with the horizons page's folds, and −0.06 points [−0.24,
+0.15] on the rows from December. With the extra cut it is −0.21 points [−0.35, −0.04] on all rows
and −0.26 points [−0.41, −0.07] from December. A training history that reads HRES across a cycle
change without telling the forecasting model which side each hour is on therefore risks losing
HRES's advantage over ERA5 on these rows.

![Figure 17: How the folds treat IFS Cycle 49r1 decides whether ENS day 0 trails ERA5 and whether
HRES beats ERA5](assets/ens_hres_wind_reconciliation.svg)

**The mean 10 m wind speeds of ENS day 0 and HRES fall against ERA5's between September and
December 2024, and UKV's does not.** ENS day 0's ratio to ERA5 is 0.97 in August and September 2024
and 0.95 in October, and it stays between 0.90 and 0.93 in every month from November 2024. HRES's is
0.89 in August 2024, 0.86 in September, and 0.85 in October, and it stays between 0.78 and 0.83
from November 2024. UKV's ratio is 0.77 in August 2024, 0.75 in September, 0.71 in October, and
0.77 in November. ECMWF's description of IFS Cycle 49r1 lists an improvement to 10 m wind speed
forecasts. The step is consistent with that change, and this study does not establish that the
change is its cause.

![Figure 18: ENS's and HRES's 10 m wind speeds fall against ERA5's between September and December
2024, and UKV's does not](assets/ens_hres_wind_monthly_ratio.svg)

**ENS day 0 trails UKV and HRES by a statistically significant margin only in the later hours of the
day, which are also its longer leads.** Splitting the scored hours by label hour, an exploratory
comparison, ENS day 0 minus UKV is +0.11 points [−0.06, +0.27] for labels 00 to 08 UTC (16,140
rows) and +0.60 points [+0.38, +0.84] for labels 10 to 23 UTC (25,604 rows). ENS day 0 minus HRES is
+0.02 points [−0.10, +0.13] for labels 00 to 08 UTC and +0.34 points [+0.16, +0.52] for labels 10 to
23 UTC. Label 09 is dropped. The split mixes ENS lead, 0 to 8 hours against 10 to 23 hours from the
00 UTC run, with time of day, and HRES's lead before 1 October 2025 is mixed into it. The split is
not a test of whether ENS could be read in time.

![Figure 19: ENS day 0 trails UKV and HRES by a statistically significant margin only in the later
hours of the day](assets/ens_hres_wind_split.svg)

**HRES's lead over ERA5 is statistically significant at one farm of three, Generator W2.** HRES
minus ERA5 is −0.54 points [−0.72, −0.35] at Generator W2, −0.05 points [−0.21, +0.11] at
Generator W1, and −0.22 points [−0.51, +0.10] at Generator W3. HRES minus UKV is +0.26 points
[+0.15, +0.37] at Generator W1, +0.02 points [−0.23, +0.26] at Generator W2, and +0.33 points
[+0.05, +0.61] at Generator W3. ENS day 0 minus UKV is +0.38 points [+0.16, +0.60] at Generator
W1, +0.24 points [−0.04, +0.50] at Generator W2, and +0.61 points [+0.43, +0.79] at Generator W3.
Per-farm rows are exploratory, and the three farms share their weather, so they are not
independent replications.

![Figure 20: HRES beats ERA5 by a statistically significant margin at one farm of
three](assets/ens_hres_wind_by_farm.svg)

**No planned contrast separates the differences between the two ECMWF products and the others.**
Each contrast mixes served lead, step width, native and served grid, IFS cycle, the source of HRES's
archive, and how each value is read. ENS day 0's lead is 0 to 23 hours from the 00 UTC run, its
steps are 3-hourly and rebuilt to hourly, and each value is the area-weighted mean of the 0.25°
cells that a farm's H3 resolution-5 cell overlaps, with its speed the magnitude of the cell-mean
wind vector. HRES is a single land cell that Open-Meteo picks. UKV is read at T+0, and ERA5 is an
analysis. The ENS-against-HRES contrast also mixes ensemble averaging with a single run, so it
cannot be attributed to ensemble averaging.

**HRES's served lead is inferred from where hour-to-hour jumps fall, and the jumps support 1 to 12
hours before 1 October 2025 and 0 to 5 hours from it.** Before 1 October 2025 the 100 m wind speed's
hour-to-hour change at 01 UTC is 1.32 times the mean of its two neighbouring hours' changes, and at
13 UTC 1.29 times, the arrival hours of the 00 and 12 UTC runs. An unexplained jump of 1.19 falls at
07 UTC. From 1 October 2025 the jumps fall at 00 UTC (1.20), 06 UTC (1.15), and 18 UTC (1.14), and
the jump at 12 UTC (1.08) is below the report's threshold of 1.10 for wind. An hour above the
threshold is evidence of a handover, not proof of one.

**The period splits do not settle which difference drives the gap between HRES and UKV.** HRES
minus UKV is +0.11 points [−0.13, +0.31] before 1 October 2025 (10 calendar months, 20,703 rows)
and +0.29 points [+0.14, +0.44] from that date (12 calendar months, 22,852 rows), although HRES's
lead is shorter from that date. A shorter lead would be expected to narrow the gap, and the two
periods also differ in season and archive source. The split at IFS Cycle 50r1 leaves 5
calendar-month labels from 12 May 2026, so its intervals under-cover.

**The choice of ENS interpolation does not move ENS day 0's error by a margin statistically
significant at the 5% level.** Each of the ENS horizons page's three alternative interpolations
moves ENS day 0's error by no more than 0.02 points from the planned combination: +0.02 points
[−0.01, +0.04], −0.01 points [−0.05, +0.03], and +0.00 points [−0.03, +0.04]. These are exploratory
contrasts.

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
  ICON-DREAM-EU reaches back to January 2010, the only product on this page besides ERA5 whose
  archive reaches before November 2022, but from August 2024 this study cannot distinguish
  ICON-DREAM-EU from ERA5, so this study gives no reason to prefer ICON-DREAM-EU for wind training
  history. ICON-DREAM-EU's extra record, from 2010 to 2022, covers years this wind study never
  scores, so whether ICON-DREAM-EU would beat ERA5 there is untested; the solar page finds
  ICON-DREAM-EU ahead of ERA5 over its own longer window, back to 2019. DWD publishes ICON-DREAM-EU
  a month at a time, after each month ends — about 2 to 3 months — so whatever its accuracy it
  cannot supply the last few weeks a live service needs.
- **ECMWF's HRES and ENS day 0: on these rows UKV described past wind better than either.** The
  evidence is 43,555 farm-hours over 22 calendar months at three farms, and UKV beats both in the
  planned contrasts of [the ECMWF results section][ecmwf-results]. ENS day 0 was compared only with
  the products scored on the same rows, so this page says nothing about ENS day 0 on other rows.
  HRES beat ERA5 pooled, mainly at one farm, and only when the folds treat IFS Cycle 49r1 as a
  break, so a training history that uses HRES needs era cuts at IFS cycle changes. The page commits
  the project to no work.
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
  five products above, on ICON-DREAM-EU's own, shorter row set (August 2024 to August 2026, 10 days
  short of the rest of the page), so that section's numbers are not directly comparable to the rest
  of this page's.
- **ICON-DREAM-EU's equal-lead and served-lead comparisons are exploratory.** Both split the row set
  by the hour of the day after the first run, not before it, so neither was named in the plan.
- **The ECMWF section's row set and folds.** The section scores 43,555 farm-hours from 1 December
  2024, 22 calendar months, against the page's 50,734 from 12 August 2024, so its errors are not
  comparable with the rest of the page's. October and November occur in one year, 2025, so a fold
  that holds either month out has no training row for that calendar month. The third era's fold
  numbers are rotated to cover every other calendar month.
- **ECMWF's products are compared over three farms, with different leads, grids, and archive
  sources.** HRES's archive source changed on 1 October 2025 and IFS Cycle 49r1 and 50r1 fall in or
  before the window, so an XGBoost model given HRES or ENS day 0 trains across cycle changes that
  only the era cuts described above partly separate. ENS day 0 is past weather delivered late, most
  of its hours end before the 00 UTC run can be read, so its scores are a best case for a live
  service.
- **The post-review ECMWF rows are exploratory.** The longer row set, the six fold designs, the
  label-hour split, the per-farm rows, and the period splits were added after the first results, and
  about 1 in 20 exploratory rows reaches significance at the 5% level by chance. The ENS horizons
  page's figure for ENS day 0 against ERA5 is not like for like with this section's, as the section
  explains.

## Reproducing the figures

```bash
uv run python studies/beam_diffuse_split/fetch_wind_point.py
uv run python studies/beam_diffuse_split/wind_products.py
uv run python studies/beam_diffuse_split/wind_products.py --era5-by-year
uv run python studies/beam_diffuse_split/wind_product_charts.py
```

`wind_icon_dream.py` needs ICON-DREAM-EU's own gridded download already on disk in
`data/studies/weather/ICON-DREAM-EU/`. No committed script reproduces that download in this
repository: it was a one-off backfill for [issue
#841](https://github.com/openclimatefix/nged-substation-forecast/issues/841), documented in that
directory's own `README_*.md` files and their `lineage_*.json` siblings, which give the exact
request and any licence prerequisite. With the download in place:

```bash
uv run python studies/beam_diffuse_split/wind_icon_dream.py
uv run python studies/beam_diffuse_split/wind_icon_dream_charts.py
```

The ECMWF section has its own scripts, which need the saved ENS horizons inputs and the Open-Meteo
Previous Runs file for HRES already on disk:

```bash
uv run python studies/beam_diffuse_split/ens_hres_past_wind.py
uv run python studies/beam_diffuse_split/ens_hres_past_wind.py --extra-fits
uv run python studies/beam_diffuse_split/ens_hres_past_wind.py --report-only
uv run python studies/beam_diffuse_split/ens_hres_past_wind_charts.py
uv run python studies/beam_diffuse_split/check_page_numbers.py \
    docs/studies/weather-products-for-past-wind.md \
    data/studies/beam_diffuse_split/past_weather_v2/ens_hres_past_wind/report.md \
    "### UKV describes past wind better than ECMWF's HRES and ENS day 0 on these rows"
```

That section's report lands in
`data/studies/beam_diffuse_split/past_weather_v2/ens_hres_past_wind/report.md`. `--extra-fits` adds
the post-review fits and leaves the first fit's losses alone, and `--report-only` rebuilds the
report from the saved losses without fitting.

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

[ecmwf-results]: #ukv-describes-past-wind-better-than-ecmwfs-hres-and-ens-day-0-on-these-rows
