# Demand inputs

**At Primary substations, four extra inputs lowered the demand forecast error by 0.1 points of
capacity or more on the rows they target: the hour and weekday, a Christmas column limited to the 14
days either side of 25 December, an explicit forecast of the unmetered solar output, and (in cold
hours only) a heating term.** The electricity price, school terms, statistics of installed solar
capacity, substation fingerprints, and the PV_Live estimate of solar generation each gave no gain of
0.1 points that held up. The experiments scored 16
[Primary substations](../../glossary.md#primary-substation) and, in some cases, 4 supply points,
over one or two validation years.

Each experiment adds one kind of information to an XGBoost demand forecast and scores it against the
same XGBoost model without that information. Errors are in [points of
capacity](../../glossary.md#capacity-and-points-of-capacity), and a negative change is an
improvement. Each number below names its metric: the error of the median (the mean absolute error of
the middle forecast) or the [quantile score](../../glossary.md#quantile-score-and-pinball-loss).
Several gains in a selection year vanished in a confirmation year or against a control.

![Paired contrasts of the input sets on the confirmation year, by substation type](../../00006-stage1a-demand-input-feature-sweep/figure_2.svg)

**For Primary substations on the confirmation year (July 2025 to June 2026), adding the hour and
weekday to the weather lowered the error of the median by 0.46 points (99.4% interval -0.61 to
-0.30), the largest single input.** Source: [experiment
6](../../00006-stage1a-demand-input-feature-sweep/result.md), an ERA5-trained [stage
1a](../../glossary.md#stages-0-1a-and-2) XGBoost model on 14 Primaries.

## Does it work?

**The hour and weekday, and a windowed Christmas column, help; school terms do not.** In experiment
6 the hour and weekday lowered the error of the median by 0.55 points on the selection year (July
2024 to June 2025), a third of the 1.5 predicted. The Christmas column lowered the Primary error by
1.78 points from 15 December to 7 January (95% interval -2.30 to -1.26) and changed it by 0.00
elsewhere, in a post hoc check on three Christmases. [Experiment
11](../../00011-school-term-and-return-to-work-columns-stage1a-demand/result.md) found no gain from
return-to-work or school-term columns on Lincolnshire Primaries: +0.20 at Easter (99.2% interval
-0.19 to +0.59) and -0.29 in half-terms (-0.68 to +0.07). Its loosely worded rule says `refuted`,
but Easter and half-terms are inconclusive.

![Change in error inside each special-day window, Primary substations](../../00006-stage1a-demand-input-feature-sweep/figure_12.svg)

**The Christmas columns cut the Christmas error and leave other days unchanged.** Source:
[experiment 6](../../00006-stage1a-demand-input-feature-sweep/result.md), post hoc windows, three
folds.

**An explicit forecast of the unmetered solar output lowers the daylight error; same-time analogues
may too, and the data could not rank the two.** In [experiment
74](../../00074-an1-same-time-analogues-vs-explicit-solar-forecast-demand/result.md), five Primaries
refitted monthly and scored on daylight April-to-September rows, the explicit offset lowered the
quantile score by 0.149 (99.17% interval -0.201 to -0.104) and 3 analogues by 0.095 (-0.185 to
-0.007), against a monthly refit with neither. An offset built from another date's sunshine scored
0.246 worse than the real offset. Statistics of installed capacity did not help 16 Primaries
(+0.006, -0.022 to +0.033; [experiment 31](../../00031-embedded-pv-features-primary-demand/result.md)).

![Quantile-score contrasts of the solar arms against the monthly refit](../../00074-an1-same-time-analogues-vs-explicit-solar-forecast-demand/figure_1_headline.svg)

**Both ways of telling the XGBoost model about sunshine lowered the daylight error, and the intervals
overlap.** Source: [experiment
74](../../00074-an1-same-time-analogues-vs-explicit-solar-forecast-demand/result.md), 5 Primaries,
one validation year.

**Satellite sunshine as a training input gave a small gain, and a correction of the forecast
sunshine a further one.** Training on [CAMS](../../glossary.md#ens-era5-and-cams) irradiance lowered
the Primary quantile score by 0.035 (97.5% interval -0.070 to +0.001), `refuted` against 0.2
planned ([experiment 9](../../00009-cams-trained-stage1a-vs-era5-trained-demand/result.md)). Mapping
the ENS shortwave onto CAMS lowered it by a further 0.067 (-0.094 to -0.039), but the monthly ratio
to CAMS still ran from 0.81 to 1.13, so the verdict is `inconclusive` ([experiment
14](../../00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md)). PV_Live,
Sheffield Solar's estimate of solar generation, gave no resolved gain at the two grid supply points
Walpole and Bicker Fen ([experiment 27](../../00027-pvlive-lags-in-stage2-gsp-demand/result.md),
-0.03; [experiment 28](../../00028-pvlive-forecast-in-stage2-gsp-demand/result.md), -0.14 with
interval -0.34 to +0.04).

**The electricity price gave no gain that held against a control.** The price at the target
half-hour lowered the pooled error of the median by 0.52 in the selection year (97.5% interval -0.88
to -0.11) and 0.18 in the confirmation year (-0.50 to +0.08), and a price from another week matched
about 0.45 of the 0.52 ([experiment 15](../../00015-same-time-price-stage1-demand/result.md)). Price
changes beat the price level by only 0.037 and 0.050 on Primaries, and national demand and wind
scored 0.22 and 0.40 below the price-change arm ([experiment
25](../../00025-price-change-features-stage1-demand/result.md)). The price from 7 days earlier did
not help the largest-battery substation in the confirmation year ([experiment
10](../../00010-elexon-price-lags-stage1a-demand/result.md)).

**Lagged power, pooling, and fingerprints.** Giving [N1](../../glossary.md#n1) the 13 lags of
[`manual_heuristic`](../../glossary.md#manual_heuristic) changed the quantile score by -0.01 (99.375%
interval -0.45 to +0.43, seed 0), and the [14-day correction](../../glossary.md#14-day-correction)
removed the gain of the weekly lags ([experiment
30](../../00030-lagged-power-feature-sets-primary-demand/result.md)). One XGBoost model per
substation type raised the score by 0.15 (99.44% interval +0.03 to +0.26) against one per series
([experiment 24](../../00024-pooling-groupings-demand-xgboost/result.md)). Fingerprints changed it by
+0.02 (-0.05 to +0.10) ([experiment 32](../../00032-substation-fingerprints-pooled-demand/result.md)).

**An explicit heating term lowered the cold-hour error a little and did not fix the cold tail.** On
16 Primaries over one winter, the quantile score fell by 0.143 in cold hours at leads up to 48 hours
(98.75% interval -0.203 to -0.034), under the 0.3 threshold, and by 0.022 over all rows (-0.042 to
+0.000). The q95 forecast was still exceeded in 16.2% of those hours ([experiment
94](../../00094-explicit-heating-offset-primary-demand/result.md)).

## Best design so far

**No experiment in this theme tested every part together, so the best design is the set of parts that
each beat their control, among the recipes tested here.** Give the weather-to-power XGBoost model the
hour and weekday and the days to Christmas inside a 14-day window ([experiment
6](../../00006-stage1a-demand-input-feature-sweep/result.md)). Where a clean solar capacity estimate
exists, add the forecast unmetered solar output as a base margin ([experiment
74](../../00074-an1-same-time-analogues-vs-explicit-solar-forecast-demand/result.md), using the
offset of [experiment
64](../../00064-pvdis-5-net-demand-as-demand-minus-forecast-solar/result.md)). In
[BG2](../../glossary.md#bg2) the offset lowered the Primary score by only 0.010 over the whole year,
because only 5 Primaries carry one and it acts in daylight from April to September ([experiment
87](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)).

## Candidates for production

**Five experiments here are marked `candidate`, and each rests on one validation year or an
unconfirmed input.**

- **Stage 1a inputs ([experiment 6](../../00006-stage1a-demand-input-feature-sweep/result.md)):**
  the clock, the windowed Christmas column, and recency weights (half-life 2 years) for the
  ERA5-trained stage 1a XGBoost model. A training start of 2023-09 stays unconfirmed (0.86 better on
  the confirmation year, 0.05 on the selection year).
- **Stage 0 ([experiment
  14](../../00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md)):** a quantile
  mapping of ENS shortwave onto CAMS for a CAMS-trained stage 1a, worth 0.067 points.
- **Pooling ([experiment 24](../../00024-pooling-groupings-demand-xgboost/result.md)):** 3 or 4
  models instead of 20, at a cost of 0.15 points.
- **Weekly mean and median ([experiment
  30](../../00030-lagged-power-feature-sets-primary-demand/result.md)):** two extra columns in N1
  with the correction, 0.05 points better (-0.08 to -0.02). The arm is post hoc.
- **Heating term ([experiment
  94](../../00094-explicit-heating-offset-primary-demand/result.md)):** a per-series base margin. The
  experiment advises a second winter before porting.

Experiment 74 is `not-for-production`.

## Gotchas and implementation details

- **Add a placebo twin of every added input.** A price from another week matched 0.45 of the
  0.52-point price gain ([experiment 15](../../00015-same-time-price-stage1-demand/result.md)), the
  shuffled price matched most of the gain at the battery site ([experiment
  10](../../00010-elexon-price-lags-stage1a-demand/result.md)), an offset from another date scored
  0.246 worse ([experiment
  74](../../00074-an1-same-time-analogues-vs-explicit-solar-forecast-demand/result.md)), and a
  shuffled capacity control matched the register products ([experiment
  31](../../00031-embedded-pv-features-primary-demand/result.md)). Without the twin, a column that
  only carries a level is credited. Re-check with a different data source.
- **Check a calendar column against a half-year flag and score it inside the special days.** The
  unwindowed days-to-Christmas column flips sign every late June, so it acts as a half-year flag, and
  a time-of-year harmonic matched the Christmas gain but worsened summer ([experiment
  6](../../00006-stage1a-demand-input-feature-sweep/result.md)). A column can also help a few days a
  year and barely move the yearly mean. Add a placebo calendar shifted by a fixed number of days
  ([experiment
  11](../../00011-school-term-and-return-to-work-columns-stage1a-demand/result.md)). Re-check with
  more years than 3.
- **Fix `colsample_bytree` at 1.0 in every arm of a column ablation.** Experiments
  [6](../../00006-stage1a-demand-input-feature-sweep/result.md),
  [30](../../00030-lagged-power-feature-sets-primary-demand/result.md),
  [31](../../00031-embedded-pv-features-primary-demand/result.md), and
  [32](../../00032-substation-fingerprints-pooled-demand/result.md) did so because column
  subsampling hands the wider arm a free win. None measured the size of that effect. Re-check if the
  arms are ever given equal column counts.
- **Give every arm the same starting level.** Experiments 76 and 78 found that XGBoost's estimated
  base score raised the (worse) score by 0.07 to 0.10 points against a start at zero, so experiment
  74 started every arm at the median of the training labels ([experiment
  74](../../00074-an1-same-time-analogues-vs-explicit-solar-forecast-demand/result.md)). Re-check
  with a different pipeline.
- **Judge a lag set against N1 with the 14-day correction.** The weekly lags gained 0.30 without the
  correction and lost 0.09 with it ([experiment
  30](../../00030-lagged-power-feature-sets-primary-demand/result.md)). Re-check if the correction
  changes.
- **Read a selection-year gain only after a confirmation year.** The price gain at the battery site
  fell from -0.92 to -0.06 ([experiment
  10](../../00010-elexon-price-lags-stage1a-demand/result.md)), and a 2023-09 training start gained
  0.05 and then 0.86 ([experiment
  6](../../00006-stage1a-demand-input-feature-sweep/result.md)). Re-check with a third year.
- **Test an input on the weather it will be served with.** A CAMS-trained XGBoost model lowered the
  daylight error by 0.88 reading CAMS and by 0.04 reading ENS ([experiment
  9](../../00009-cams-trained-stage1a-vs-era5-trained-demand/result.md)), and an ERA5-trained solar
  forecast ran 1.8 to 3.0 points too high on ENS members ([experiment
  28](../../00028-pvlive-forecast-in-stage2-gsp-demand/result.md)). Re-check with a different weather
  product.
- **Report a pooled share with and without the largest-residual substation.** One substation set the
  pooled share in [experiment 7](../../00007-cams-static-correction-stage1a-residual/result.md).
  Re-check on new series.
- **Give every pooled XGBoost model the series identifier, and no fixed-window statistics for a
  held-out series.** Dropping the identifier raised the score by 0.82 for by-type pools ([experiment
  24](../../00024-pooling-groupings-demand-xgboost/result.md)), and training-window statistics
  worsened a held-out substation by 1.39, probably because 7 series learn them as identifiers
  ([experiment 32](../../00032-substation-fingerprints-pooled-demand/result.md)). Re-check with a
  pool of hundreds of series.
- **Score cold-hour and special-day contrasts on more than one winter.** Three ISO weeks held 66% of
  the cold rows at leads up to 48 hours ([experiment
  94](../../00094-explicit-heating-offset-primary-demand/result.md)), and the Christmas windows rest
  on three Christmases ([experiment
  6](../../00006-stage1a-demand-input-feature-sweep/result.md)). Re-check when a second winter
  exists.

## What we have not settled

- **A second validation year.** The analogues on top of the offset (-0.066, -0.149 to +0.015) and the
  heating term need one.
- **Battery response to price.** Only the largest battery site was testable, and national demand and
  wind scored at least as well as the price, so the price may be a proxy for them.
- **Stage 0 on more than 14 months of ENS and all members**, and half-term effects with more than 9
  half-terms.

The theme's [research plan](plan.md) lists the proposals still waiting.
