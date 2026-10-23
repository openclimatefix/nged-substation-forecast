---
id: '00300'
slug: metered-farm-best-guess-assembly
title: >-
  How good is the assembled best-guess day-ahead forecast for the six metered solar farms, and what
  does a new farm with no history get?
theme: metered-solar-farms
also_relevant_to:
- production-reference-and-best-guess
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private predictions and tables under
  data/ml-experiments/00300-metered-farm-best-guess-assembly/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: one design (the candidate that experiments 270, 287, and
  260 picked), so no design choice is open. Callers: this folder's scripts, copied from experiments
  270, 287, and 260, with no new reviewed module. Asset graph: not touched.
hypothesis: >-
  The assembled candidate (spread-binned plant P2, a generalised additive model (GAM) band on the
  plant's residuals, a mixture-density head (MDN), blended per quantile with XGBoost) scores better
  than XGBoost alone on the quantile score and at the 0.99 quantile, and its advantage is larger
  for a farm with no history because the plant carries what the missing history cannot.
prediction: >-
  Written before any score is read. On own history, the candidate beats XGBoost on the quantile
  score by 1% to 2% (98.75% interval wholly below zero), beats it by at least 20% at the 0.99
  pinball loss, and is within 1% of XGBoost on the absolute error of the median. With zero history
  (leave-one-farm-out) the candidate beats XGBoost by at least 3%. The 3% and the 20% are the
  predictions that could fail.
decision_rule: >-
  Quantile score in points of capacity, equal weight per farm, month-block resampling of the 12
  scored months at 98.75% (Bonferroni across four contrasts: candidate minus XGBoost on the quantile
  score in own history and in zero history, candidate minus XGBoost at the 0.99 pinball loss in own
  history, and the candidate minus XGBoost on the absolute error of the median in own history).
  Supported when the own-history score interval is wholly below zero, the 0.99 gain is at least
  20%, and the zero-history gain is at least 3%. Partly supported when one or two of the three
  hold. The 1% score threshold is set by the cost of the extra stages (a plant, a band model, a
  blend), which a gain under 1% would not repay. Stop rule: if the candidate's score interval
  against XGBoost includes zero in both settings, report that the plant-plus-band and XGBoost are
  equivalent for scoring and that the choice is explainability; add no arms.
benchmark: >-
  XGBoost production-style (experiment 275's x2 in own history, experiment 270's l6 in zero
  history, both read from stored predictions when the refit schedule below matches, otherwise refit
  here), the manual_heuristic and climatology floors of experiment 224, and capacity_forecast
row_set_id: >-
  Experiment 287's rows: experiment 224's scored-year daylight half-hours at 12 to 36 hour leads
  that every arm holds (farms A to F, 2025-09 to 2026-08)
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), mean absolute
  error (MAE) of the median, pinball loss and coverage at 0.01, 0.05, 0.95, and 0.99, per farm, arm,
  lead, clear-sky band, hour, and season
code: ml-experiments/00300-metered-farm-best-guess-assembly/
advice: []
production_status: not-for-production
adopted_in: none
---

This page is the plan. The write-up replaces it when the fits finish.

## Plan

- **Question.** How does the candidate that experiments
  [270](../00270-metered-farm-xgboost-vs-plant-diagnostic/result.md),
  [287](../00287-metered-farm-spread-binned-plant-heads/result.md), and
  [260](../00260-metered-farm-quantile-combination/result.md) picked score in absolute terms against
  the references a reader will ask for, and what does a farm with no history get?
- **Arms.** Floors (`manual_heuristic`, `climatology`; `capacity_forecast` scored only where its
  inputs exist). P2 alone. XGBoost. The GAM candidate. The GAM plus the MDN (5 seeds). The
  per-quantile blend of GAM, MDN, and XGBoost. Ablations, dropping one at a time from the GAM
  candidate: the spread, lagged power, the availability ratio, the band model (equals P2), and the
  blend (equals the best single band). Zero-history versions of every arm.
- **Schedule.** Monthly refits on all earlier months, with the first scored month fitted on the
  months before 2025-09. Weekly refits are used only for P2, the GAM, and XGBoost if the monthly
  run shows time to spare. The write-up says which schedule produced which number. The MDN has 5
  seeds, and the ablations use the GAM only.
- **Row set and weather.** Experiment 287's rows. Forecast weather is the ECMWF ensemble 00 UTC run
  (available at forecast time); lagged power is the farm's own output one or two days earlier. The
  stored ensemble tables cover leads of 12 to 36 hours only, so the lead axis of the headline chart
  covers that range and not 0.5 to 54 hours.
- **Reference rows.** Experiment 218's plant with observed weather (0.973 sunny fraction) and the
  plant with the ensemble mean (0.91) are quoted from the stored results, not refitted.
- **Curtailment.** The node-plus-generators metric needs a substation node with its generators, and
  these six farms are single series, so the farm cards state what the metric cannot score.
- **Sizing triggers.** Stored: private predictions only. Serving path: not touched. Degradation
  rule: not touched. Defensible designs: one. Callers: this folder's scripts.
