---
id: "00022"
slug: era5-weather-ceiling-primary-demand
theme: production-reference-and-best-guess
also_relevant_to: []
status: done
size: simple
hypothesis: >-
  For the 16 Primary substation demand series, giving the winning recipe of experiment 00019 (the
  single-stage ENS-trained quantile model N1 with a recency half-life of 0.5 years, the 14-day error
  correction, and the 1.15 spread factor) ERA5 weather in place of ECMWF ensemble (ENS) weather at
  serving time lowers the quantile score by at least 0.3 points of capacity over all leads.
prediction: >-
  The all-variable ERA5 arm scores 0.4 points below the ENS arm over all leads (plausible range 0.1
  to 0.9), with most of the gain from 3 days on, where ENS error is largest, and almost none up to
  24 hours. Temperature (with dew point) carries most of the gain, shortwave radiation the next
  most, and wind speed, pressure, and precipitation each less than 0.05 points. The seven single
  variable gains add to within 0.15 points of the all-variable gain.
decision_rule: >-
  Eight planned contrasts (the all-variable ERA5 arm and each of the seven weather variables
  replaced alone, each minus the ENS arm), quantile score over all leads, read at 99.38%
  (Bonferroni across eight), whole months and seeds resampled. The all-variable contrast decides the
  verdict. Weather-source work for demand is worth planning when its interval lies wholly below zero
  and its estimate is at most -0.3 points (about 6% of the stack's 5.32), because a second weather
  product adds a data source to the pipeline; it is a small gain when the interval lies wholly
  below zero and the estimate is above -0.3; it is refuted when the interval lies wholly above
  -0.3 and so rules out a gain of that size; otherwise it is inconclusive. A variable is named as
  worth improving when its own interval lies wholly below zero. Scores by lead band are exploratory.
benchmark: >-
  The ENS arm, which is experiment 00019's full stack refitted for this experiment's code path, and
  checked against 00019's stored stack to 3 decimal places.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels), in points of capacity (1
  point is 1% of the series' effective capacity)
verdict: refuted
best_arm: >-
  The all-variable ERA5 arm (5.22 points on the quantile score over all leads, against 5.32 for the
  ENS arm)
headline: >-
  ERA5 weather in place of ENS weather lowered the quantile score over all leads by 0.10 points of
  capacity (99.38% interval -0.16 to -0.04), so the 0.3-point gain the rule asked for is ruled out.
  ERA5 shortwave radiation alone gave -0.08 (-0.13 to -0.04).
lessons: >-
  A weather product as accurate as ERA5 would lower the Primary demand score by 1.9%, nearly all
  from shortwave radiation, and none of it from the all-variable arm up to 2 days ahead. With a model
  trained on ENS weather, weather error is a small part of the Primary demand error.
advice:
  - >-
    Do not plan weather-source work for Primary demand on the grounds of skill, because ERA5 weather
    in place of ENS weather lowered the quantile score by only 0.10 points (C1).
  - >-
    If a second weather product is tried for Primary demand, test its shortwave radiation first,
    because that column supplied -0.08 of the -0.10 (C2).
  - >-
    Look for gains from sources other than weather for Primary demand at leads under 2 days, where
    ERA5 weather did not help (lead-band contrast).
production_status: not-for-production
adopted_in: none
pull_request: https://github.com/openclimatefix/nged-substation-forecast/pull/1227
code: ml-experiments/00022-era5-weather-ceiling-primary-demand/
---

# Experiment 22: how much would Primary demand gain from ERA5 weather in place of ENS weather?

We asked how much better the demand forecast for Primary substations (substations that step power
down to the local distribution network) would be if its weather input were an estimate of past
weather instead of a weather forecast. The answer is an upper limit on what better weather products
could bring, because estimates of past weather arrive about 5 days late and cannot be used live. We
swapped the weather in the best demand forecast and scored 16 Primary substations over 73 forecast
runs. The estimates of past weather lowered the error by only 0.10 points of capacity, so the
0.3-point gain we had predicted is ruled out.

**Refuted: ERA5 weather lowered the quantile score by 0.10 points of capacity (99.38% interval -0.16
to -0.04), so the 0.3-point gain the decision rule asked for is ruled out.** The rows are the 16
Primary substations and 73 forecast runs of experiment 00019, scored with its full stack (N1 with a
half-life of 0.5 years, the 14-day correction, the spread factor of 1.15). The ENS arm scores 5.32
and the all-variable ERA5 arm scores 5.22, in points of capacity (1 point is 1% of a series'
effective capacity). `manual_heuristic` scores 5.89 on the same rows. The ENS arm matches experiment
19's stored stack to within 0.0003 points. ERA5 reaches us about 5 days late, so no serving system
could use it: the arm is a ceiling for a weather product as accurate as ERA5. ENS is the European
Centre for Medium-Range Weather Forecasts' ensemble forecast, which has 51 members. The quantile
score measures how well the forecast quantiles fit what happened, and smaller is better.
`manual_heuristic` is a stored baseline that forecasts from the power at the same time of day in
earlier weeks and years.

![Each ERA5 arm minus the ENS arm](figure_1.svg)

| Weather column from ERA5 | Quantile score minus ENS arm (99.38% interval) |
|---|---|
| All seven columns | -0.100 (-0.162 to -0.044) |
| Shortwave radiation | -0.084 (-0.130 to -0.043) |
| Longwave radiation | -0.016 (-0.028 to -0.008) |
| Temperature at 2 m | -0.013 (-0.039 to +0.011) |
| Dew point at 2 m | -0.003 (-0.010 to +0.002) |
| Surface pressure | +0.008 (-0.007 to +0.025) |
| Precipitation | -0.006 (-0.012 to -0.0002) |
| Wind speed at 10 m | -0.001 (-0.008 to +0.005) |

**Shortwave radiation carries nearly all of the gain.** The seven single-column gains sum to -0.115
against -0.100 for all columns together. Temperature, which the plan expected to lead, gave -0.013
with an interval that includes zero. Precipitation is near zero in both sources, so its interval
excluding zero is not a finding.

![All seven columns, by lead band](figure_2.svg)

**The all-variable ERA5 arm helps only from 3 days ahead.** The gain by lead band is -0.075 at 3 to
6 days (95% interval -0.112 to -0.039), -0.130 at 7 to 10 days (-0.192 to -0.070) and -0.147 at 11
to 15 days (-0.216 to -0.078). It is +0.028 at 0 to 24 hours (+0.005 to +0.047) and +0.002 at 1 to 2
days (-0.036 to +0.038), so ERA5 is not better than ENS inside 2 days. The ENS arm scores 4.50 at 0
to 24 hours and 5.75 at 11 to 15 days. The band intervals are not corrected for five bands.

## Limits

- **The N1 XGBoost model was trained on ENS control-member weather and not on ERA5.** ERA5 differs
  from ENS in bias (for example, its mean shortwave is 132.6 against 139.1 W/m² over the validation
  year), so a model trained on ERA5 might gain more.
- **The 5% to 95% band covers 86% with ERA5 against 85% with ENS, and the 10% to 90% band covers 74%
  against 76%.** The
  spread factor was fitted for ENS and was not refitted.
- **Each arm's correction history comes from that arm's own forecasts.** An ERA5 arm has therefore
  also had its past errors corrected with ERA5 weather.

## Plan, written before any code ran

- **Model.** Each series' N1 model with the half-life of 0.5 years is refitted as in experiment
  19, on the ENS control member up to 2025-06-30, because that experiment stored no fitted models.
  The 14-day correction and the 1.15 spread factor are applied as in experiment 19.
- **Arms.** The ENS arm feeds the 51 ENS members as in experiment 19. The all-variable ERA5 arm
  replaces the seven weather columns (shortwave, longwave, temperature, dew point, surface
  pressure, precipitation, wind speed at 10 m) with ERA5 at the valid time; the same value enters
  every member, so the 51 members collapse to one. Seven single-variable arms replace one column and
  keep the ENS members for the other six. The correction history of each arm comes from that arm's
  own past forecasts.
- **Weather source and availability.** ERA5 reaches us about 5 days late, so it does not exist at
  forecast time. The arms therefore give a ceiling for a weather product with ERA5's accuracy, and
  no deployable system could reach it.
- **Training and serving mismatch.** The model is trained on ENS control-member weather. ERA5
  differs from ENS in bias (for example radiation), so the ERA5 arms may understate the gain from a
  model trained on ERA5; retraining on ERA5 is not tested.
- **Rows.** The 73 runs of `fold_a_every5d_v1` and valid times before 2026-07-01, as in 00019. Leads
  are
  banded 0-24 h, 1-2 d, 3-6 d, 7-10 d, 11-15 d, reported exploratorily.
- **Leakage checks.** The fit stops at 2025-06-30 23:30; lags and the correction use observations
  at least 3.5 hours before the origin; the ERA5 values are those of the valid time only.

### Sizing

- **Stored data:** prediction sets only.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** one; it is a permutation of the serving weather.
- **Callers:** the scripts copy 00019's helpers.

Size: simple, with one Sonnet review of the code and one of the write-up.

### Earlier advice

Followed: test the correction on the winning recipe (00019); read the ENS control in the same way as
00013. No departure.
