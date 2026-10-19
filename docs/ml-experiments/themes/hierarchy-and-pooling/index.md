# Hierarchy and pooling

**One XGBoost model shared by a group of Primary substations forecasts almost as well as one model
per substation (0.15 points of capacity worse, 99.44% interval +0.03 to +0.26), but nothing tested
lets a substation outside the pool be forecast, and forecasts of the series beneath a supply point
added nothing.**

Pooling means training one XGBoost model on the rows of several series, with the series identifier
as a column, so that 20 models become 3 or 4. The pooling results are from one validation year (July
2025 to June 2026) on 16 [Primary substations](../../glossary.md#primary-substation) and 4
[supply points](../../glossary.md#bulk-supply-point-and-grid-supply-point), with
[ENS](../../glossary.md#ens-era5-and-cams) weather. The hierarchy results cover 2 bulk supply points
and 2 grid supply points, too few to rank any design.

![Pooling costs 0.15 to 0.29 points with the series identifier and 0.8 to 1.0 points without it](../../00024-pooling-groupings-demand-xgboost/figure_1.svg)

**Every pooled arm scores worse than one model per series (positive is worse), by 0.15 to 0.29
points with the series identifier and 0.8 to 1.0 without it.** Source: [experiment
24](../../00024-pooling-groupings-demand-xgboost/result.md); C1 to C3 and C5 to C7 use 16 Primary
series, and C4, C8, and C9 use all 20 series.

## Does it work?

**Three or four pooled models score 0.15 points of capacity worse than 20 per-series models on the
16 Primaries, inside the pre-registered margin of 0.30.** The scores are the
[quantile score](../../glossary.md#quantile-score-and-pinball-loss) over all leads after the
[14-day correction](../../glossary.md#14-day-correction) and a spread factor of 1.15 (the
[stack](../../glossary.md#stack) of experiment 19, with a half-life of 0.5 years). Per-series models
score 5.32, one model per substation type 5.47 (C1, +0.15, 99.44% interval +0.03 to +0.26), and one
model per capacity band 5.47 (C3, +0.15, +0.05 to +0.26). Both beat
[`manual_heuristic`](../../glossary.md#manual_heuristic) (5.89), the type pools by 0.42 (interval
-0.68 to -0.19, level not stated in the write-up). The verdict is supported ([experiment
24](../../00024-pooling-groupings-demand-xgboost/result.md)).

**One model for all 20 series falls outside the margin.** It scores 5.61 on the 16 Primaries (C2,
+0.29, +0.19 to +0.42). On the 2 bulk supply points alone it loses 0.80 (0.64 to 1.00, exploratory,
95%), and the 4 supply points are too few to separate the type pools from per-series models (+0.03,
-0.15 to +0.15; experiment 24).

**For Primaries, the number of models matters more than which series share one.** Random groups of 5
score 5.47 on the 16 Primaries, the same as the type pools. On all 20 series the type pools beat
random groups by 0.06 (C9, -0.13 to -0.01), and on the 4 supply points by 0.29 (exploratory)
(experiment 24).

**The series identifier is essential.** Dropping it raises the score of the type pools by 0.82 (C7,
+0.46 to +1.23) and of the all-series pool by 0.85 (C8, +0.72 to +1.00) (experiment 24).

**Describing each substation with numbers instead of an identifier added nothing for series seen in
training.** Adding a substation's location, its mean level and shape over the training window, or
its level over the previous 14 days to the identifier moved the score by at most 0.02 points (C1 to
C4; C4, +0.02, 99.375% interval -0.05 to +0.10). The verdict is inconclusive by the pre-registered
rule, though the data exclude a gain above 0.05. Training-window statistics replaced the identifier
at no measurable loss (C5, +0.01, +0.00 to +0.02; [experiment
32](../../00032-substation-fingerprints-pooled-demand/result.md)).

![Quantile score of the cold-start arms](../../00032-substation-fingerprints-pooled-demand/figure_2.svg)

**A substation held out of the pool was forecast worse with training-window statistics than with no
fingerprint (C7, +1.39, +0.99 to +1.91), and better with recent power statistics alone (post hoc,
-0.23, -0.29 to -0.16, 95%).** The test used 8 held-out Primaries, one seed, and a pool of 7 other
series. The write-up suggests, without testing it, that 7 distinct fingerprints act as 7 identifiers
(experiment 32).

**Forecasts of the series beneath a supply point did not lower its score.** Adding the summed median
forecasts of the Primaries and generators beneath a bulk supply point changed its score by +0.04
(99.17% interval -0.18 to +0.25) at Boston and +0.19 (-0.10 to +0.73) at Skegness. At the 2 grid
supply points, generic wind and solar columns changed it by -0.02 (-0.06 to +0.03) and -0.04 (-0.13
to +0.07). The experiment's verdict is inconclusive ([experiment
36](../../00036-hierarchy-aware-bsp-gsp-forecasts/result.md)).

![Contrasts for forecast columns at four supply points: none excludes zero](../../00036-hierarchy-aware-bsp-gsp-forecasts/figure_1.svg)

**An N1 trained on ERA5 and applied to ENS members lowered the grid supply point scores by 0.36 and
0.40, but its intervals span -1.8 to +1.0.** [N1](../../glossary.md#n1) is the single-stage XGBoost
quantile model. Only Boston resolved a gain, of 0.41 (0.15 to 0.66, exploratory, 95%) (experiment
36).

## Best design so far

**Of the pooled designs tested, one model per capacity band or per substation type scored best, and
none beat one model per series.**

1. Train an XGBoost quantile model (the N1 recipe of experiment 24: 9 quantile levels, ENS control
   member, lead hour, calendar columns, three power lags, power scaled by each series' effective
   capacity) on 3 capacity bands (terciles) or 4 substation types.
2. Give every XGBoost model the series identifier as a categorical column.
3. Apply the 14-day correction and a spread factor of 1.15 to each series' quantiles.

The capacity-band and type pools score the same to two decimals (experiment 24). The pre-registered
rule chose capacity bands because they have fewer models.

## Candidates for production

**Experiment 24 is `candidate`; experiments 32 and 36 are `not-for-production`.**

- **Pooled XGBoost models for Primary demand (experiment 24).** The XGBoost forecaster would train 3
  or 4 models instead of 20 and store them at 73 to 94 MB instead of 435 MB, for a loss of 0.15
  points. Each model would need the series identifier, and a new series would need a retrain. The
  test covered 20 series, so it says nothing about 2,500.
- **Nothing from experiments 32 or 36 is ready.** No fingerprint helped a series seen in training,
  and no forecast column lowered a supply point's score.

## Gotchas and implementation details

- **Keep the series identifier in every pooled model.** Without it, the type pools lost 0.82 and the
  all-series pool 0.85 ([experiment 24](../../00024-pooling-groupings-demand-xgboost/result.md)).
  Re-check with a different model type or a different pool of substations.
- **Score pooled arms with and without the 14-day correction.** The correction lowers per-series
  models by 1.0 and type pools by 0.6, so type pools beat per-series models by 0.30 uncorrected
  (exploratory, 95%, -0.56 to -0.03) and lose 0.15 corrected (experiment 24). A ranking measured on
  one stack may not carry to another. Re-check if the correction changes.
- **Scale boosting rounds with the pool size, or say they were not scaled.** Experiment 24 fixed 500
  rounds and depth 6, so the all-series loss may be partly lower capacity per series. Re-check with
  rounds scaled to rows.
- **Do not compare fit seconds across arms.** Other sessions loaded the machine during the
  capacity-band and random-group fits of experiment 24, so only model counts and saved sizes are
  reliable. Re-check with timings on an idle machine.
- **Match a held-out substation's fingerprint window to the pool's.** The held-out substation's
  28-day statistics differed from the pool's 15-month ones, and using the whole window removed about
  two thirds of the C7 damage (post hoc; [experiment
  32](../../00032-substation-fingerprints-pooled-demand/result.md)). Re-check with a pool of
  hundreds of series.
- **Do not read the oracle's gain as headroom for forecasts.** Giving the XGBoost model the observed
  values of the series beneath a bulk supply point lowered its score by 5.6 and 3.4, while their
  forecasts lowered it by none ([experiment
  36](../../00036-hierarchy-aware-bsp-gsp-forecasts/result.md)). Re-check if the child forecasts
  come from a different weather source.
- **Note how child-forecast columns were built.** Experiment 36 cross-fitted them by time block, so a
  block's child model saw months on both sides, and training used the control member while
  validation used each member (experiment 36). Look for a gain that changes when the columns are
  fitted strictly out of time.
- **Do not apply the 14-day correction to supply points without checking.** Experiment 36 left it
  off because [experiment 21](../../00021-error-correction-variants-bsp-gsp-demand/result.md) found
  it unproven there, and experiment 24 reports supply points separately. Re-check with more than 4
  supply points.
- **Refit the reference and compare it with the stored predictions.** The refit N1 in experiment 36
  matched experiment 13's stored N1 within 0.003 points at every series, so the contrasts share one
  baseline. Look for the same check in any new pooled comparison.

## What we have not settled

- Whether pooling holds at hundreds of series, and for substations the pool never saw.
- Whether supply points pool at all; 4 series cannot say.
- Whether an N1 trained on ERA5 helps at supply points beyond Boston.
- Whether one model shared across substations, with drift and long history, beats separate models.
  That is the subject of the [differentiable-physics
  background](../differentiable-physics-background.md) and the sub-themes [long history and
  drift](../differentiable-physics-long-history-and-drift/index.md) and [demand calendar and
  weather](../differentiable-physics-demand-calendar-and-weather/index.md).

The [Research plan](plan.md) lists what is planned next, and [Experiments](experiments.md) lists
every experiment. Experiment 103 appears there as also relevant: its data look fixes the rows and
dates the pooled differentiable-physics model uses, and it has no pooling result.
