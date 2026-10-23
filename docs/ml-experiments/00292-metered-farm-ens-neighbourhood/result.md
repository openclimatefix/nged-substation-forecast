---
id: '00292'
slug: metered-farm-ens-neighbourhood
title: >-
  Does the ensemble's irradiance on the cells around a metered solar farm lift the day-ahead
  forecast?
theme: metered-solar-farms
also_relevant_to:
- weather-products-and-blending
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: weather summaries and predictions under
  data/ml-experiments/00292-metered-farm-ens-neighbourhood/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible way to summarise a
  neighbourhood (block statistics, an upwind cell, a convolution), so each is an arm. Callers:
  this folder's scripts, copied from experiments 224 and 243, and no new reviewed module. Asset
  graph: not touched.
hypothesis: >-
  A day-ahead cloud field is often right in shape and wrong in position by tens of kilometres, so
  summaries of the ensemble's irradiance over the hexagonal cells around a metered solar farm carry
  information that the farm's own cell lacks. Giving H5 (the plant plus a neural network that
  outputs all 11 quantile offsets) those summaries lowers its quantile score at 12 to 36 hour
  leads.
prediction: >-
  H5 with the neighbourhood summaries (N1) beats H5 on the farm's own cell (N0) by at least 2% in
  quantile score, with a month-block interval wholly below zero, and beats its placebo by at least
  2%. N1 or the convolution arm (N2) beats XGBoost given the same summaries (X1) by at least 1%.
decision_rule: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), equal weight
  per farm, on experiment 243's scored rows, with 98.75% intervals from resampling the 12 scored
  months 20,000 times (Bonferroni over four contrasts). C1 N1 minus N0. C2 N1 minus the placebo.
  C3 best of N1 and N2 minus X1. C4 N2 minus N0. Supported when C1 and C2 are each at least 2%
  below zero with intervals wholly below zero. The 2% threshold pays for the extra weather
  extraction (19 cells instead of one) and 11 more model inputs in the retraining pipeline. Stop
  rule: if N1 gains under 1% over N0 at 4 or more of the 6 farms, report that the neighbourhood
  adds nothing detectable at 12 to 36 hours and skip N2.
benchmark: >-
  N0 (H5 on the farm's own cell, refitted here and compared with experiment 243's stored H5), X0
  (XGBoost on the own cell), and experiment 224's manual_heuristic and climatology floors
row_set_id: >-
  Experiment 243's scored rows (daylight half-hours at 12 to 36 hour leads, sun more than 5
  degrees above the horizon, farms A to F, 2025-09 to 2026-08)
metric: >-
  Quantile score, MAE of the median and of the mean forecast, pinball loss and coverage at 0.01,
  0.05, 0.95, 0.99, per farm and arm, split by clear-sky index
code: ml-experiments/00292-metered-farm-ens-neighbourhood/
production_status: not-for-production
adopted_in: none
---

**Planned.** This page is the plan; the write-up replaces it once the fits have run.

## Plan

- **Neighbourhood.** The ensemble store holds H3 resolution-5 hexagons (about 15 km between
  centres), not a 0.25 degree lattice, so "3 by 3" and "5 by 5" become the hexagon's ring 1 (7
  cells) and rings 1 and 2 (19 cells). A farm needs 5 of 6 ring-1 cells and 15 of 19 cells in the
  store (sea cells are absent) or it is dropped from every arm.
- **Variables.** The store holds downward shortwave, 2 m temperature, 10 m wind, and others, but no
  cloud cover. Cloud is therefore not used; the share of member-cell pairs with clear-sky index
  above 0.8 stands in for it. Clear-sky irradiance is the farm's, applied to every neighbour.
- **Weather and forecast time.** ECMWF ensemble, 00 UTC run, 51 members, available at forecast
  time. Lagged power is the farm's own power one day earlier (two when the lead exceeds 24 hours).
- **Arms.** N0: H5 on the own cell, as in experiment 243. N1: N0 plus 11 summaries per half-hour
  (mean, standard deviation, minimum, maximum of the member-mean irradiance over the ring-1 and
  rings-1-2 blocks; the share of member-cell pairs with clear-sky index above 0.8 in each block;
  the member-mean irradiance at the ring-1 cell nearest the direction the 10 m wind blows from).
  N2: N0 plus a small convolution over the 19 cells' member mean and member spread (run only if the
  stop rule does not fire). X0 and X1: XGBoost without and with the summaries. Placebo: N1 with the
  summary vector taken from another day of the same farm, month, hour, and own-cell clear-sky band.
- **Schedule.** One fit per arm and seed on the months before 2025-09, five seeds per neural arm,
  the last two tuning months select the epoch (the screening schedule of experiment 243; own
  history only).
- **Advice followed.** Let the network output all 11 offsets (H5), leave day of year out, give every
  arm the lagged power and the spread (experiment 243).
