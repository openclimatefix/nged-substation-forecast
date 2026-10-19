# Weather products and blending

**Of the weather changes tested, only training on estimates of past weather (ERA5) clearly lowered
the error, for 6 solar farms and 3 wind farms up to 6 days ahead, and no extra weather forecast has
yet been shown to help in a way a live service could use.**

A weather product is a source of weather data, such as the ECMWF ensemble (ENS), the Met Office's
UKV (a 2 km forecast of the UK), the US ensemble GEFS, or ICON-EU. The experiments trained XGBoost
models on one product's history, then added a second product as extra input columns or as a blend.
They scored 6 solar farms, 3 wind farms, and 16
[Primary substations](../../glossary.md#primary-substation), each over one validation year. For
Primary demand, even weather as accurate as ERA5 lowered the score by only 0.10 points.

![ERA5 training lowers the solar and wind error up to day 6](../../00001-era5-trained-stage1a-vs-ens-trained/figure_2.svg)

**Training on ERA5 lowered the mean absolute error of the ensemble-mean forecast at 6 solar farms
and 3 wind farms up to day 6, and not beyond.** Source: [experiment
1](../../00001-era5-trained-stage1a-vs-ens-trained/result.md). Lower is better.

## Does it work?

**Training on ERA5 lowered the solar and wind farm error up to day 6.** Over all leads, the
ERA5-trained XGBoost model scored 15.39 against 17.09 for wind (-1.70, 95% interval -2.33 to -1.08)
and 7.67 against 8.25 for solar (-0.57, -0.83 to -0.32), in [points of
capacity](../../glossary.md#capacity-and-points-of-capacity), with no calendar or lag columns
([experiment 1](../../00001-era5-trained-stage1a-vs-ens-trained/result.md)). The contrasts were
revised after two reviews, so they are not pre-registered. A later test used the [quantile
score](../../glossary.md#quantile-score-and-pinball-loss) and power lags at 24 to 54 hours. It found
the ENS-trained model 0.218 points lower than the ERA5-trained [best
guess](../../glossary.md#best-guess) for wind (-0.472 to +0.069) and 0.107 lower for solar (-0.327
to +0.053). Both differences are unresolved ([experiment
54](../../00054-product-trained-xgboost-wind-farms/result.md), [experiment
57](../../00057-product-trained-xgboost-solar-farms/result.md)).

**For demand series it did not carry over.** On 22 series that are not wind or solar farms, 16 of
them Primaries, the ERA5-trained model scored 1.37 points worse (+0.68 to +2.17) with weather
columns only. Starting the ERA5 history in 2023-09 instead of 2019-09 lowered the error by 1.32
(-1.81 to -0.90), a post hoc read that does not separate drift in the series from bad early data
([experiment 3](../../00003-era5-trained-stage1a-vs-ens-trained-demand/result.md)).

**Weather as accurate as ERA5 gave a small gain for Primary demand, and only for a model trained on
ENS.** Swapping ENS for ERA5 weather in the [stack](../../glossary.md#stack) of experiment 19 (a
half-life of 0.5 years) lowered the quantile score by 0.10 (99.38% interval -0.16 to -0.04) at 16
Primaries over 73 runs. Shortwave radiation alone gave -0.084, and inside 2 days there was no gain
([experiment 22](../../00022-era5-weather-ceiling-primary-demand/result.md)). Training on CAMS
sunshine instead of ERA5 shortwave gave -0.035 (97.5% interval -0.070 to +0.001), against a planned
0.2 ([experiment 9](../../00009-cams-trained-stage1a-vs-era5-trained-demand/result.md)). Mapping the
ENS shortwave onto CAMS first ([stage 0](../../glossary.md#stages-0-1a-and-2)) gave a further -0.067
(97.5% interval -0.094 to -0.039). The pair scored 0.102 below the ERA5-trained arm. The verdict
stays inconclusive because the monthly mismatch to CAMS reached 18.6% ([experiment
14](../../00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md)).

**A second global ensemble, GEFS, did not improve the sunshine anomaly.** The anomaly is a
forecast's sunshine minus its own monthly mean, scored against the CAMS anomaly at 5 Primaries on
daylight rows from April to September. Cross-validated squared error was 1.109 times ENS's for GEFS
(worse in all 9 held-out months) and 1.002 times for a fitted blend, against the 0.99 needed
(refuted, [experiment 50](../../00050-second-weather-forecast-for-solar-anomaly/result.md)).

![GEFS and the blend against ENS](../../00050-second-weather-forecast-for-solar-anomaly/figure_1_headline.svg)

**GEFS correlated worse with the CAMS anomaly than ENS did, and the blend matched ENS.** Source:
[experiment 50](../../00050-second-weather-forecast-for-solar-anomaly/result.md).

**A UKV and ENS blend lowered the squared error of the same anomaly by 7.6% over 0 to 54 hours in
cross-validation, and the validation year cannot confirm it.** The validation correlation gain was
+0.011 (-0.006 to +0.027), and 3 of 6 origin-month blocks favoured the blend. The test was post hoc,
and its intervals resample 6 or 7 origin months, so they understate the uncertainty ([experiment 53](../../00053-ukv-vs-ens-solar-anomaly/result.md)).

**UKV wind lowered the wind-farm quantile score by about 0.2 points at 24 to 54 hours, a small gain
that is an upper limit.** An ENS-trained XGBoost model given UKV wind as extra columns, trained on
UKV's own history, scored -0.198 (-0.316 to -0.069) at 3 wind farms. The interval's end nearest zero
does not reach the pre-registered 0.15, and a shuffled-UKV placebo scored +0.013 (-0.012 to +0.040).
One farm gained clearly (-0.413) and one did not (+0.018), post hoc. The UKV data come from the CEDA
archive, which differs statistically from the live feed ([experiment
54](../../00054-product-trained-xgboost-wind-farms/result.md); [UKV-CEDA
study](../../../studies/forecasts/ukv-ceda-blends.md)).

![UKV contrasts at three wind farms](../../00054-product-trained-xgboost-wind-farms/figure_1.svg)

**Giving UKV wind to an ENS-trained model lowered the wind-farm score, and the shuffled-UKV placebo
did not.** Source: [experiment 54](../../00054-product-trained-xgboost-wind-farms/result.md).

**The same test with UKV shortwave, cloud, and temperature at 6 solar farms was inconclusive.** The
score changed by -0.035 (-0.119 to +0.037), so a gain of 0.10 can be neither shown nor ruled out
([experiment 57](../../00057-product-trained-xgboost-solar-farms/result.md)). Averaging the wind
best guess with ICON-EU and the ECMWF high-resolution forecast (IFS) gave -0.055 (-0.200 to +0.072)
at days 1 and 2, also inconclusive, and the products' runs were half a day older than ENS
([experiment 52](../../00052-more-nwp-products-for-wind-farms/result.md)).

## Best design so far

**The best recipe tested here depends on the target, and none beat the reference for demand.**

- **Solar and wind farms, up to day 6:** train the weather-to-power model on ERA5 (and CAMS sunshine
  for solar) and apply it to each of the 51 ENS members ([experiment
  1](../../00001-era5-trained-stage1a-vs-ens-trained/result.md)).
- **Wind farms, 24 to 54 hours:** add UKV wind as columns to an ENS-trained model, trained on UKV's
  history ([experiment 54](../../00054-product-trained-xgboost-wind-farms/result.md)). This is an
  upper limit.
- **Primary substations:** keep the model trained on ENS. If a model is trained on estimates of past
  weather, start the history in 2023-09 ([experiment
  3](../../00003-era5-trained-stage1a-vs-ens-trained-demand/result.md)) and map the ENS shortwave
  onto CAMS ([experiment
  14](../../00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md)).

## Candidates for production

**Two experiments are marked `candidate`, and neither is ready to port as it stands.**

- **ERA5-trained weather-to-power model for solar and wind farms ([experiment
  1](../../00001-era5-trained-stage1a-vs-ens-trained/result.md)).** Production would train on ERA5
  and CAMS history and serve ENS members, which needs the staged design. The gain was measured
  without power lags, and experiments 54 and 57 found the ENS-trained model within an unresolved
  margin of the ERA5-trained best guess at 24 to 54 hours.
- **Stage 0 mapping of ENS shortwave onto CAMS for Primaries ([experiment
  14](../../00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md)).** Production
  would fit the mapping on 14 or more months of ENS and CAMS pairs. The mapping left monthly
  mismatches, widened the member spread by 27%, and needs a CAMS-trained model.

**No second weather product (GEFS, UKV, ICON-EU, IFS) is ready.** Experiments 50, 52, 53, 54, and
57 are `not-for-production`.

## Gotchas and implementation details

- **Train each product on its own history.** The arms are ENS-trained, ENS plus the other product
  as columns, and the other product alone; an ERA5-trained model is only a reference. A model
  trained on ENS and fed ERA5 weather understates what ERA5 training would give ([experiment
  22](../../00022-era5-weather-ceiling-primary-demand/result.md)). The ERA5-trained wind model
  fed uncorrected ICON-EU winds alone scored 2.5 points worse than the best guess, and IFS winds
  1.3 points worse, at days 1 and 2 ([experiment 52](../../00052-more-nwp-products-for-wind-farms/result.md)).
  Re-check with a different model or a bias-corrected product.
- **"Better weather adds little" holds only for a model trained on ENS.** Experiment 22 trained on
  the ENS control member, and ERA5 differs from ENS in bias (mean shortwave 132.6 against 139.1
  W/m2). Experiment 9's CAMS-trained arm gained 0.88 daylight points at observed weather and 0.04
  over all leads with ENS input ([experiment 9](../../00009-cams-trained-stage1a-vs-era5-trained-demand/result.md)).
  Look for a Primary demand model trained on ERA5 and scored on the same stack.
- **Pass a shuffled-product placebo through every step.** In experiment 54 the placebo showed that
  extra columns alone do nothing. In experiment 52 it scored +0.343 because July's equal weights and
  the spread factor, not the shuffled forecasts, drove the score, so fit the spread factor on blends
  made under the same weights ([experiment
  54](../../00054-product-trained-xgboost-wind-farms/result.md), [experiment
  52](../../00052-more-nwp-products-for-wind-farms/result.md)).
- **Use the live feed, and check its variables on a recent run.** The CEDA UKV archive has no 100 m
  wind and lacks four hourly steps, UKV was upgraded on 2026-01-21, and Open-Meteo's UKV history
  stops at lead day 0 ([experiment 54](../../00054-product-trained-xgboost-wind-farms/result.md),
  [experiment 57](../../00057-product-trained-xgboost-solar-farms/result.md)). Feeds gain fields
  over time, so an old date gives a wrong answer. Re-check when a live UKV history exists.
- **Feed the run the service would have.** Experiment 52 fed runs half a day older than ENS, and
  experiment 53 assumed a 3-hour UKV delivery time without measuring it ([experiment
  53](../../00053-ukv-vs-ens-solar-anomaly/result.md)). Look for a measured delivery time before
  reading a blend's gain.
- **Give a guard arm the variant it guards.** The ENS training frame repeats each valid time about
  15 times, so calendar columns let an ENS-trained model memorise the calendar position, and a
  one-row-per-valid-time guard arm with lags failed by construction ([experiment
  1](../../00001-era5-trained-stage1a-vs-ens-trained/result.md), [experiment
  3](../../00003-era5-trained-stage1a-vs-ens-trained-demand/result.md)). Re-check when the columns
  change.
- **Calibrate each source's anomaly, and cross-validate by whole months.** GEFS's raw anomaly was
  about 15% too large, and the interval width follows the number of origin months ([experiment
  50](../../00050-second-weather-forecast-for-solar-anomaly/result.md), [experiment
  53](../../00053-ukv-vs-ens-solar-anomaly/result.md)).
- **Check the hour-of-day peak of every source.** CAMS and ENS are means labelled at the period end
  and UKV is instantaneous, so a mislabelled UKV control peaked at 10 UTC instead of 12 UTC
  ([experiment 53](../../00053-ukv-vs-ens-solar-anomaly/result.md)).
- **Take the planned threshold from a contrast between like arms.** Experiment 9 planned 0.2 and
  resolved 0.035 ([experiment 9](../../00009-cams-trained-stage1a-vs-era5-trained-demand/result.md)).
  Report a pooled share with and without the largest series: a fixed CAMS
  correction removed 46.6% of the daylight squared residual, one substation set the pooled figure,
  and a fixed bias alone removed 39.1% ([experiment
  7](../../00007-cams-static-correction-stage1a-residual/result.md), inconclusive).

## What we have not settled

**Whether a live UKV history gives the wind-farm gain, whether ICON-EU and IFS help when read at the
service's own time, and whether a source of a different kind helps solar farms.** Experiment 50
tested GEFS against the CAMS solar anomaly at 5 Primaries only. The open proposals are in the
[research plan](plan.md).
