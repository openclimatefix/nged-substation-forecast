---
id: "00022"
slug: era5-weather-ceiling-primary-demand
status: planned
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
production_status: not-for-production
adopted_in: none
code: ml-experiments/00022-era5-weather-ceiling-primary-demand/
---

# Experiment 22: how much would Primary demand gain from ERA5 weather in place of ENS weather?

**Planned; no result yet.** The experiment sizes the gain from a perfect weather product, which says
how much weather-source work could give for Primary demand.

## Plan

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
- **Rows.** The 73 runs of `fold_a_every5d_v1` and valid times before 2026-07-01, as in 00019. Leads are
  banded 0-24 h, 1-2 d, 3-6 d, 7-10 d, 11-15 d, reported exploratorily.
- **Leakage checks.** The fit stops at 2025-06-30 23:30; lags and the correction use observations
  at least 3.5 hours before the origin; the ERA5 values are those of the valid time only.

## Sizing

- **Stored data:** prediction sets only.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** one; it is a permutation of the serving weather.
- **Callers:** the scripts copy 00019's helpers.

Size: simple, with one Sonnet review of the code and one of the write-up.

## Earlier advice

Followed: test the correction on the winning recipe (00019); read the ENS control in the same way as
00013. No departure.
