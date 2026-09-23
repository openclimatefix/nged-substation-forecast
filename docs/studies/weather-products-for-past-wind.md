# Which weather product best describes past wind?

This page is the wind counterpart of [Which weather product best describes past
sunshine?](weather-products-for-the-past.md). It measures how well five weather products describe
past hub-height wind at the three metered wind farms in the trial area, and says which product the
two consumers that can use the result should read: training history, and historical features in the
live service.

**The Met Office's UK variable-resolution model (UKV) and the German weather service's ICON-D2
describe past wind best, and both beat the reanalysis by about half a point of capacity.** ICON-D2
and UKV cannot be separated robustly: ICON-D2 leads across the window, but not since the Met Office
upgraded UKV in January 2026, and not when both are shown only their served 100 m wind. **ICON-EU,
the European ICON model, beats the reanalysis only when shown its own 80 m wind, and mostly in the
summer half of the year.** **ICON global is the weakest product here, and two defects in how it is
served explain more of its deficit than its coarser grid can.** Every error on this page is a mean
absolute error as a percentage of each generator's own 99th percentile of output, which the page
calls its capacity. The evidence is three wind farms in flat Lincolnshire, and 50,734
generator-hours from August 2024 to September 2026.

## The five products

**Each product publishes wind at different heights, and each archive serves a different lead, so the
comparison has to fix both.** CAMS, the satellite retrieval that describes past sunshine best,
publishes no wind, so it has no arm here.

| Product | Hub-height wind it publishes | Served lead | Covers all of Great Britain? |
|---|---|---|---|
| ERA5, the European Centre for Medium-Range Weather Forecasts' (ECMWF's) reanalysis | 100 m | an hourly analysis | yes |
| UKV | 100 m | T+0, the analysis | yes |
| ICON-D2 | 80 m and 120 m | 0 to 2 hours | no: stops near 2.5°W |
| ICON-EU | 80 m and 120 m | 0 to 2 hours | yes |
| ICON global | 80 m and 120 m | 0 to 5 hours | yes |

- **Hub heights.** Open-Meteo serves a 100 m wind for the ICON products too, but that value is the
  120 m speed multiplied by 0.98, so a model cannot tell the two apart. Each ICON product is
  therefore shown its native 80 m wind, and ERA5 and UKV their native 100 m wind. A sensitivity arm
  shows every product only the 100 m wind it serves.
- **Leads.** ERA5's hourly wind is an analysis at every hour: its largest hour-to-hour jumps fall at
  the boundaries of its data assimilation windows, 09 and 21 UTC, not at the start of its forecasts.
  The ICON leads come from where their hour-to-hour jumps fall, which marks each run's switch.
- **History.** UKV's hub-height wind starts on 12 August 2024, when Open-Meteo's own UKV downloader
  started, which is why the comparison covers only two years. ERA5 starts in 1940 and the ICON
  products in November 2022.

## How the comparison was made

**The method is the solar study's, with two differences that wind requires.**

- **Same rows, same model, same number of columns.** An hour is kept only if all five products cover
  it. A gradient-boosted tree (XGBoost) is fitted per generator on one product's hub-height speed,
  that height's direction, and its 10 m speed, plus the time of day, the season, and which side of
  the UKV upgrade the hour falls. Every product is read from its nearest land grid cell.
- **The power hour is centred on the label.** Open-Meteo's wind is an instantaneous value at the
  timestamp, where its radiation is a mean over the hour before. So the hour of power labelled T is
  the hour from 30 minutes before T to 30 minutes after. Every product scores best with the hour
  centred this way; the solar study's hour, ending at T, would handicap UKV by 0.27 points.
- **Every hour holding an exactly-zero half-hour is dropped.** From April 2026 the feed publishes no
  exact zeros at two of the generators, and their calm half-hours are missing instead, which the
  build already drops. Dropping the zeros makes the earlier period match. Most dropped hours are
  calm, so behaviour near cut-in is under-sampled; dropping no rows at all moves every contrast by
  0.03 points or less.
- **Folds, normalisation and intervals as in the solar study.** Blocks of whole months are cut
  separately before and after the UKV upgrade, each error is divided by its own generator's
  capacity, and each 95% interval comes from resampling whole calendar months.
- **Four contrasts named before the run:** ICON-EU against ERA5, UKV against ERA5, ICON-EU against
  UKV, and ICON-D2 against ICON-EU. Every other figure is exploratory.

## UKV and ICON-D2 describe past wind best

**UKV beats ERA5 by 0.44 points [0.24, 0.63], and ICON-D2 beats ICON-EU at every generator, by 0.26
points [0.19, 0.33].**

| Product | Mean absolute error, % of capacity |
|---|---|
| ICON-D2 | 6.69 |
| UKV | 6.83 |
| ICON-EU | 6.96 |
| ERA5 | 7.27 |
| ICON global | 7.65 |

UKV's advantage over ERA5 is larger since its upgrade, 0.79 points against 0.53 on the same calendar
months before it, and it holds at two of the three generators; at the third, whose output swings
with months of unrecorded turbine availability, the interval includes zero.

**ICON-D2 and UKV cannot be separated robustly.** Across the window ICON-D2 leads by 0.14 points
[0.03, 0.25]. Before the UKV upgrade it led by 0.22, and since the upgrade UKV is 0.06 points ahead
[−0.03, +0.19]. Showing both products only their served 100 m wind puts UKV ahead by 0.12 points. So
either product is a defensible choice, and the choice between them should not rest on this study.

## ICON-EU beats the reanalysis only with its own 80 m wind

**ICON-EU beats ERA5 by 0.31 points [0.17, 0.45] when shown its native 80 m wind, and loses to ERA5
by 0.27 points when shown only the 100 m wind Open-Meteo serves for it.** The served 100 m value is
the 120 m speed rescaled, so the 80 m wind is closer to the hub. ICON-EU's advantage also depends on
the season: 0.48 points from April to September, and no detectable difference from October to March.
ICON-EU is 0.13 points behind UKV [0.01, 0.23], and the two cannot be told apart before the upgrade
or in winter.

## ICON global is the weakest, for reasons that are not its grid

**ICON global is 0.70 points behind ICON-EU [0.48, 0.93], and two defects in how Open-Meteo serves
it explain more of that than its coarser grid can.** Its nearest grid cell at one generator lies
partly over the sea, with 10 m winds 16% faster than the land cell's. Reading the land cell instead,
as this study does, cut that generator's deficit from 1.38 to 0.36 points in the first run. And its
10 m wind changes in steps at two generators, on single days in June 2025 and June 2026, which no
other product shows. This page does not attribute the deficit to grid resolution.

## Which product each consumer should read

These recommendations rest on three wind farms in one flat part of Lincolnshire, over two years.

- **Training history: ICON-D2 or ICON-EU from November 2022, ERA5 before that.** UKV's hub-height
  wind starts only in August 2024. ICON-D2 does not cover generators west of about 2.5°W, where
  ICON-EU is the option, shown its 80 m wind.
- **Historical features in the live service: UKV or ICON-D2, with ICON-EU at its 80 m wind where
  ICON-D2 does not reach.** ERA5 arrives about five days late and cannot supply the last few hours
  at run time.
- **Capacity estimation and disaggregation: no recommendation.** Every figure here is after a
  per-generator recalibration those consumers cannot make.

## What this does not show

- **Terrain.** The issue expected resolution to matter most for wind because terrain shapes
  hub-height wind. The three generators are onshore in flat country, and two of them share one ERA5
  grid point, so this study cannot speak to terrain, offshore wind, or any other region.
- **Raw accuracy.** Every figure is after per-generator recalibration by a tree, so the figures rank
  how much each product's wind says about the output, not how close its speeds are: the products'
  mean hub-height speeds differ by up to 0.5 m/s.
- **Anything before August 2024,** and behaviour near the turbines' cut-in speed, which the zero
  rule under-samples.
- **Turbine availability.** One generator's output swings for months at a time with availability no
  product can see, which weakens every contrast there.

## Reproducing the figures

```bash
uv run python studies/beam_diffuse_split/fetch_wind_point.py
uv run python studies/beam_diffuse_split/wind_products.py
```

The report lands in `data/studies/beam_diffuse_split/beam_diffuse_wind_products/report.md`.

