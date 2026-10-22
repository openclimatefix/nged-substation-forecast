---
id: '00243'
slug: metered-farm-architecture-comparison
title: >-
  Does a physics plant with a small neural head forecast a metered solar farm better than a plant,
  XGBoost, or a neural network alone?
theme: metered-solar-farms
also_relevant_to:
- differentiable-physics-embedded-generation
- calibration-and-uncertainty
status: planned
size: >-
  medium-to-complex. Stored: private tables, trained weights and predictions under
  data/ml-experiments/00243-metered-farm-architecture-comparison/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible design for each head (what the
  network outputs, and what it sees), so each head is fixed in this plan before any fit. Callers:
  this folder's scripts, copied from experiment 224, and no new reviewed module. Asset graph: not
  touched.
hypothesis: >-
  A solar-farm power model (the plant) as the base prediction, with a small neural network that
  adapts its inputs, its output, or both, forecasts a metered solar farm 12 to 36 hours ahead
  better than the plant alone and better than XGBoost, and keeps the plant's advantage on a farm
  it has never seen, while a neural network with no physics does not beat XGBoost on six farms.
prediction: >-
  The best hybrid beats XGBoost's MAE of the median by at least 3% and its pinball loss at the 0.01
  and 0.99 quantiles by at least 5%, while the neural network without physics does not beat
  XGBoost. The hybrids' gain is largest on cloudy half-hours.
decision_rule: >-
  All arms are scored on the rows of experiment 224's scored year at 12 to 36 hour leads that every
  arm holds, with the quantile score (twice the mean pinball loss over 11 levels, in points of
  capacity), equal weight per farm, and 95% intervals from resampling the 12 scored months. Headline
  contrast: the best hybrid minus the plant alone, and minus XGBoost, in quantile score, MAE of the
  median, and pinball loss at the 0.01 and 0.99 quantiles. Supported when the best hybrid meets the
  prediction with the quantile-score interval below zero against both references. Stop rule: if no
  hybrid beats the plant alone by at least 2% in quantile score at 2 of the first 3 farms (A, B, C),
  stop and report that the plant is near the weather ceiling. The 2% threshold reflects the added
  complexity of a trained network beside a fitted plant.
benchmark: >-
  The plant alone (with the offsets of experiment 224), XGBoost on the same rows, and the
  manual_heuristic and climatology floors of experiment 224
row_set_id: >-
  Experiment 224's scored-year rows at the 12 to 36 hour lead band that every arm holds (daylight
  half-hours, sun more than 5 degrees above the horizon, farms A to F, 2025-09 to 2026-08)
metric: >-
  Quantile score in points of capacity, MAE of the median, and pinball loss and coverage at
  0.01, 0.05, 0.95, and 0.99, per farm, arm, and setting
code: ml-experiments/00243-metered-farm-architecture-comparison/
production_status: not-for-production
adopted_in: none
---

Planned. The first commit of this branch is the plan, and the write-up replaces this page.

## Plan

- **Weather and forecast time.** ECMWF ensemble (51 members), 00 UTC run, at the nearest 0.25
  degree cell, 12 to 36 hours ahead, as in [experiment
  224](../00224-metered-farm-dayahead-two-number/result.md): ensemble-mean irradiance, beam, diffuse,
  and temperature, and the members' standard deviation of irradiance. All exist at forecast time.
  Lagged power is the farm's own observed power one day earlier (two days earlier when the lead is
  over 24 hours), which exists at forecast time. This is part (A); part (B) adds UKV variables.
- **Plant.** The fleet-shape plant of experiment 224 (tilt 35 degrees, south, 0.4% loss per kelvin of
  cell temperature), with a scale and a ceiling per farm. Its quantiles are the plant forecast plus
  11 residual quantiles of the tuning months in four bins of the forecast ("the offsets").
- **Arms.** References: the two floors and the plant alone and XGBoost. Pure neural network: a
  multilayer perceptron on the raw weather, sun geometry, lead, and lagged power. H1: the plant times
  a bounded scale factor from a network. H2: the plant plus an additive correction from a network.
  H3: a network adjusts the irradiance and temperature, which then go through the unchanged
  plant (trained end to end). H4: H3 followed by H1. H5: the plant plus a network that outputs all 11
  quantile offsets. H6: H2 without the plant forecast as input (the blind control).
- **Settings.** (i) Own history: each farm's network and plant scale are fitted on its own tuning
  months (before 2025-09). (ii) Zero history, leave-one-farm-out: the networks and XGBoost are fitted
  on the other five farms' tuning months, and the plant scale and ceiling are the median over those
  five farms, so the scored farm contributes no data. The two are reported separately.
- **Inputs left out on purpose.** Day of year is not an input, because a calendar column lets a
  network memorise the date; solar elevation and azimuth carry the season. The hybrids see no farm
  label, so setting (ii) is possible.
- **Schedule.** One fit per arm, seed, and setting on the tuning months; no refit in the scored year
  (the screening schedule; the weekly refits the brief asks for the headline are not run). Five seeds
  per neural arm; the last two tuning months select the training epoch. Pinball loss at the 11
  levels is the training loss; outputs are non-negative and capped at the farm's ceiling by
  construction, and the quantiles are non-crossing by construction.
