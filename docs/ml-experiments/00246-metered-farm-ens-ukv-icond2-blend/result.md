---
id: '00246'
slug: metered-farm-ens-ukv-icond2-blend
title: >-
  Does adding the DWD ICON-D2 weather forecast to the ensemble plus UKV lower a metered solar farm's
  day-ahead error?
theme: metered-solar-farms
also_relevant_to:
- weather-products-and-blending
status: planned
size: >-
  medium. Stored: private tables and predictions under
  data/ml-experiments/00246-metered-farm-ens-ukv-icond2-blend/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible design (a blend of irradiance
  before the plant, or ICON-D2 columns in a second-stage tree), so both are registered as arms.
  Callers: this folder's scripts, copied from experiment 229, and no new reviewed module. Asset
  graph: not touched.
hypothesis: >-
  The DWD ICON-D2 regional weather forecast (2 km grid, run every 3 hours, 48 hours ahead) carries
  information about cloud over a metered solar farm that neither the ECMWF ensemble nor the Met Office
  UKV carries, so adding it to the ensemble plus UKV lowers the farm's day-ahead error. The evidence
  against is that at day 1 every single product lost to the ensemble mean for solar in the
  matched-lead study, and ICON-D2's errors are likely to be correlated with UKV's. Every gain is an
  upper limit for production, because the UKV archive used is the CEDA research archive.
prediction: >-
  Adding ICON-D2 lowers the median over the six farms of the quantile score at the rows described
  below by under 1.5%, with an interval that includes zero (month-block bootstrap, 99% level). A
  result at least 2% lower with an interval wholly below zero would be a surprise and would call for
  a replicate year.
decision_rule: >-
  Quantile score is twice the mean pinball loss over the 11 levels of `studies.ens_members`, in points
  of capacity, on the daylight half-hours (sun more than 5 degrees above the horizon) of experiment
  229's band 0 whose valid time is at most 29 hours after the ensemble run's 00 UTC start, so that
  the ICON-D2 run made 24 hours before the valid time (published 3 hours after it starts) is
  available at the forecast origin of 08 UTC, with a complete UKV run, and an ICON-D2 value. Folds are
  experiment 229's (replicate 2025-03 to 2025-08, scored 2025-09 to 2025-12, both inside the second
  UKV era), with intervals from resampling the 10 scored months 20,000 times. Five contrasts are
  pre-registered at the 99% level (a Bonferroni correction over five): D1 is the three-source blend
  B3 against experiment 229's blend A3 (re-run on these rows), D2 is the tree with ICON-D2 columns B4
  against experiment 229's tree A5 (re-run on these rows), D3 is the ensemble plus ICON-D2 blend B2
  against the ensemble plant A1, D4 is B3 against its shuffled-ICON-D2 placebo, and D5 is B4 against
  its placebo. Each contrast is the median over the six farms of the per-farm percentage change. The
  hypothesis is supported when D1 or D2 is at least 1.5% lower with an interval excluding zero, and
  the stop rule applies in the other direction (see below). The 1.5% is set by cost: a fourth weather
  source adds a download, a licence, and a publication delay to the serving path for every gain, and
  a gain under 1.5% would not pay for them. Stop rule: if ICON-D2 and UKV-CEDA overlap the farms'
  cleaned power for under 18 months, the page reports that the experiment cannot tell. The optional
  ICON-D2 ensemble arm (three months of history, no replicate) runs only if D1 or D2 is a gain.
benchmark: >-
  Experiment 229's arms A3 (blend of ensemble and UKV plant forecasts) and A5 (tree on ensemble
  columns and UKV shortwave and cloud), re-run on this experiment's rows, plus the ensemble plant A1
  and the manual_heuristic and climatology floors
row_set_id: >-
  Metered solar-farm half-hours with sun more than 5 degrees above the horizon, a null
  `drop_reason` in experiment 179's cleaned series, a complete UKV 00 UTC run, an ICON-D2
  previous-day-1 value, valid 12 to 29 hours after the ensemble run's 00 UTC start, scored
  2025-09-01 to 2025-12-31 with a replicate of 2025-03-01 to 2025-08-31
metric: >-
  Quantile score in points of capacity, MAE of the median, fraction of output fitted on sunny
  half-hours (clear-sky irradiance above 650 W/m²), correlation of each product's plant-forecast error
  with the others', and coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99
code: ml-experiments/00246-metered-farm-ens-ukv-icond2-blend/
production_status: not-for-production
adopted_in: none
---

Planned. The first commit of this branch is the plan, and the write-up replaces this page once the
experiment has run.

## Plan

- **ICON-D2 at forecast time.** The data is Open-Meteo's Previous Runs file for farms A to F
  (`previous_day1` columns: the run made 24 hours before the hour, so the lead is 24 hours for every
  row). Reading the files costs nothing, and no download is needed. The run is assumed published 3 hours
  after it starts. For the forecast origin of 08 UTC that makes a valid time of up to 29 hours after
  the ensemble run's 00 UTC start the latest row whose ICON-D2 run exists, so the rows stop there;
  the stitched analysis-like file is not used.
- **Alignment.** Open-Meteo irradiance is a mean over the hour ending at the stamp, so each
  half-hour row takes the hour containing its midpoint, scaled by the cosine of the zenith at the
  midpoint over the mean cosine of the hour. The hour-of-day profile against the clear-sky curve is
  checked before scoring, as the `data-validation` skill requires.
- **Arms.** R3 and R5 are [experiment 229's](../00229-metered-farm-ens-ukv-blend/result.md) A3 and
  A5 on these rows. B2 is a convex blend of the ensemble and ICON-D2 plant forecasts. B3 is a convex
  blend of ensemble, UKV, and ICON-D2 on the simplex (steps of 0.1), weights by UKV cloud band and
  fitted on the other farms. B4 is the experiment 229 tree with ICON-D2 shortwave, direct radiation,
  air temperature, and wind speed columns. B4x is the same tree with ensemble and ICON-D2 columns
  only. Controls: B3 and B4 with ICON-D2 taken from a donor row of the same month, hour band, and
  cloud band.
- **Night placebo.** The farm tables hold only daylight rows, so a night placebo cannot be built; the
  shuffled controls replace it.
- **Reports.** Weights each product receives per cloud band, the correlation of ICON-D2's plant-forecast
  error with UKV's and the ensemble's, absolute scores, tails, and the sunny fraction fitted against
  experiment 218's references.
- **Prior advice.** The plan follows experiment 229's design and its advice to use sky variables
  only for UKV, and departs from the brief's 12 to 36 hour band, because ICON-D2 is available only to
  29 hours at an 08 UTC origin.
- **Schedule.** One refit per fold, as in experiment 229.
