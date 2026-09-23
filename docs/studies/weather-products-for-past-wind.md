# Which weather product best describes past wind?

This page is the wind counterpart of [Which weather product best describes past
sunshine?](weather-products-for-the-past.md). It measures how well five weather products describe
past hub-height wind at the three metered wind farms in the trial area. It also says which product
the two consumers that can use the result should read: training history, and historical features in
the live service.

**The Met Office's UK variable-resolution model (UKV) and the German weather service's ICON-D2
describe past wind best.** Across the window both beat the reanalysis by about half a point of
capacity, but from October to March only ICON-D2's advantage is detectable. ICON-D2 and UKV cannot
be separated robustly. **ICON-EU, the European ICON model, beats the reanalysis at either height it is
shown.** **ICON global is the weakest product, and about half of its gap is a pair of steps in the
wind it is served at one generator.** Every error on this page is a mean absolute error as a
percentage of each generator's own 99th percentile of output, which the page calls its capacity. The
evidence is three wind farms in flat Lincolnshire, and 50,734 generator-hours from August 2024 to
September 2026.

## The five products

**Each product publishes wind at different heights, and each archive serves a different lead, so the
comparison has to fix both.** CAMS, the satellite retrieval that describes past sunshine best,
publishes no wind, so it has no arm here.

| Product | Wind heights served | Served lead | Covers all of Great Britain? |
|---|---|---|---|
| ERA5, the European Centre for Medium-Range Weather Forecasts' (ECMWF's) reanalysis | 10 m and 100 m | an hourly analysis | yes |
| UKV | 100 m (Open-Meteo also serves 50 m and 80 m) | T+0, the analysis | yes |
| ICON-D2 | 80 m and 120 m | 0 to 2 hours | no: stops near 2.5°W |
| ICON-EU | 80 m and 120 m | 0 to 2 hours | yes |
| ICON global | 80 m and 120 m | 0 to 5 hours | yes |

- **Hub heights.** The three wind farms' hub heights are not known. Open-Meteo serves each ICON
  product a 100 m wind that is its 120 m speed multiplied by about 0.98, which a model cannot tell
  apart from the 120 m speed. Each ICON product is shown its native 80 m wind, and ERA5 and UKV
  their 100 m wind. A check arm shows UKV its 80 m wind, which scores 0.02 points worse than its
  100 m wind.
- **Leads.** ERA5's hourly wind is an analysis at every hour: its largest hour-to-hour jumps fall at
  the boundaries of its data assimilation windows, 09 and 21 UTC, not at the start of its forecasts.
  ICON-D2's and ICON global's leads come from where their hour-to-hour jumps fall, which marks each
  run's switch. ICON-EU's jumps show the 3-hourly pattern only weakly, so its lead rests on its
  3-hourly cycle and the solar study's lineage check.
- **History.** UKV's hub-height wind starts on 12 August 2024, when Open-Meteo's own UKV downloader
  started, which is why the comparison covers only two years. ERA5 starts in 1940, and the ICON
  products' 80 m wind is complete from December 2022.

## How the comparison was made

**The method is the solar study's, with two differences that wind requires.**

- **Same rows, same model, same number of columns.** An hour is kept only if all five products cover
  it. A gradient-boosted tree (XGBoost) is fitted per generator on one product's hub-height speed,
  that height's direction, and its 10 m speed. It is also given the time of day, the season, and
  which side of the UKV upgrade the hour falls. Every product is read from its nearest land grid
  cell.
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
  UKV, and ICON-D2 against ICON-EU. Every other figure is exploratory. The plan showed every product
  its served 100 m wind; after the first run each ICON product was switched to its native 80 m wind.
  The pre-run arm's results are reported alongside and change no ranking.

## UKV and ICON-D2 describe past wind best

**UKV beats ERA5 by 0.44 points [0.24, 0.63] across the window, and ICON-D2 beats ICON-EU at every
generator, by 0.26 points [0.19, 0.33].**

| Product | Mean absolute error, % of capacity |
|---|---|
| ICON-D2 | 6.69 |
| UKV | 6.83 |
| ICON-EU | 6.96 |
| ERA5 | 7.27 |
| ICON global | 7.65 |

**Both leaders' advantage over ERA5 is mostly a summer result.** UKV beats ERA5 by 0.71 points
[0.58, 0.85] from April to September and by 0.13 points [−0.17, +0.41] from October to March.
ICON-D2 beats ERA5 by 0.77 and 0.36 points [0.08, 0.64]. UKV's advantage over ERA5 is also larger
since its upgrade: 0.79 points, against 0.53 on the same calendar months before it. It holds at two
of the three generators. At the third, whose output swings with months of unrecorded turbine
availability, the interval includes zero.

**ICON-D2 and UKV cannot be separated robustly.** Across the window ICON-D2 leads by 0.14 points
[0.03, 0.25]. Before the UKV upgrade it led by 0.22, and since the upgrade UKV is 0.06 points ahead
[−0.03, +0.19]. With the served 100 m wind and the 10 m speed, as planned before the run, ICON-D2
leads by 0.07 points [−0.03, +0.17].

**At equal lead ICON-D2 beats UKV, and its advantage fades within hours of each run.** ICON-D2 runs
every 3 hours, so at each run hour it is served at T+0, like UKV. On those hours ICON-D2 is 0.30
points ahead [0.19, 0.41]. One hour into the run it is 0.15 points ahead [0.02, 0.27], and two hours
in the two are level. This comparison was made after the run.

## ICON-EU beats the reanalysis at either height

**ICON-EU beats ERA5 by 0.31 points [0.17, 0.45] when shown its native 80 m wind, and by 0.22
points [0.08, 0.35] when shown the served 100 m wind the plan specified before the first run.**
Both arms carry the 10 m speed. The 80 m wind improves ICON-EU by 0.10 points and ICON-D2 by 0.07
points over the served 100 m wind. ICON-EU's advantage over ERA5 depends on the season, as UKV's
does: from April to September ICON-EU is 0.48 points ahead, and from October to March the two cannot
be told apart. ICON-EU is 0.13 points behind UKV [0.01, 0.23]. That interval includes zero under the
second hyperparameter setting and with UKV at 80 m, and the two cannot be told apart before the
upgrade or in winter. Since the upgrade UKV is 0.41 points ahead of ICON-EU [0.33, 0.49].

## ICON global is the weakest product

**ICON global is 0.70 points behind ICON-EU [0.48, 0.93]. About half of that gap is a pair of steps
in ICON global's served wind at one generator, and a smaller part is its longer served lead.** At
that generator, ICON global's wind relative to ICON-EU's falls by roughly 12% in early June 2025 and
rises by about as much in early June 2026, at 10 m and at 80 m alike. No other product shows the
steps, and their cause is not identified. Telling the model which side of the steps each hour falls
cuts ICON global's deficit at that generator from 1.39 to 0.44 points, and the pooled deficit to
0.38 points [0.27, 0.50].

ICON global runs every 6 hours, so half its hours are served 3 to 5 hours after the run, where
ICON-EU's are 0 to 2 hours. On the hours where the two leads are equal ICON global is 0.63 points
behind [0.41, 0.86], against 0.76 on the others. At the two generators without the steps it is still
0.33 and 0.40 points behind, which is consistent with its coarser grid, though this study does not
separate the grid from other differences between the two models. At one generator the nearest
ICON global grid cell is influenced by the sea; every figure here reads the nearest land cell
instead.

## Which product each consumer should read

**These recommendations rest on three wind farms in one flat part of Lincolnshire, over two years.**

- **Historical features in the live service: UKV, or ICON-D2 where ICON-D2 reaches.** UKV covers all
  of Great Britain and has been 0.41 points ahead of ICON-EU [0.33, 0.49] since its upgrade. ERA5
  arrives about five days late and cannot supply the last few hours at run time.
- **Training history: ICON-D2 where it covers, from December 2022, with ICON-EU or ERA5 elsewhere.
  This study scores none of them before August 2024.** ICON-D2 is the only product here that beats
  ERA5 in both halves of the year; ICON-EU's advantage is detectable only from April to September. A
  history that switches from ERA5 to an ICON product mid-record changes product, so a model trained
  on it has to be told which product each hour comes from, as it is told the UKV era here. The ICON
  archives before August 2024 have not been screened for steps like the one in ICON global.
- **Capacity estimation and disaggregation: no recommendation.** Every figure here is after a
  per-generator recalibration those consumers cannot make.

## What this does not show

- **Terrain.** The issue expected resolution to matter most for wind because terrain shapes
  hub-height wind. The three generators are onshore in flat country, and two of them share one ERA5
  grid point, so this study cannot speak to terrain, offshore wind, or any other region.
- **Raw accuracy.** Every figure is after per-generator recalibration by a tree, so the figures rank
  how much each product's wind says about the output, not how close its speeds are: the products'
  mean speeds at the served 100 m differ by up to 0.7 m/s.
- **Anything before August 2024,** and behaviour near the turbines' cut-in speed, which the zero rule
  under-samples.
- **Turbine availability.** One generator's output swings for months at a time with availability no
  product can see, which weakens every contrast there.

## Reproducing the figures

```bash
uv run python studies/beam_diffuse_split/fetch_wind_point.py
uv run python studies/beam_diffuse_split/wind_products.py
```

The report lands in `data/studies/beam_diffuse_split/beam_diffuse_wind_products/report.md`.
