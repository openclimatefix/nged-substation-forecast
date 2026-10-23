---
id: '00310'
slug: metered-farm-long-lead-scoring
title: >-
  Does the metered solar-farm forecast beat climatology at leads of 1 to 10 days, and which single model family is best?
theme: metered-solar-farms
also_relevant_to:
- production-reference-and-best-guess
status: planned
size: >-
  medium. Stored: private tables and predictions under
  data/ml-experiments/00310-metered-farm-long-lead-scoring/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: one design (experiment 300's candidate, refitted per lead
  band), so no design choice is open. Callers: this folder's scripts, copied from experiments 224,
  270, and 300, with no new reviewed module. Asset graph: not touched.
hypothesis: >-
  Experiment 300's candidate (spread-binned plant, additive band, mixture head, blend with XGBoost)
  still beats XGBoost at every day band from 1 to 10 days ahead, and its margin over climatology
  shrinks as the lead grows.
prediction: >-
  Written before any score is read. The candidate beats XGBoost on the quantile score in every day
  band (98.75% month-block intervals wholly below zero at days 1 to 5). Its margin over climatology
  falls under 5% by days 8 to 10. Lagged power adds nothing beyond day 2, so the candidate's gain over
  XGBoost comes from the plant and the spread, and widens with lead. The single-family question has
  no prediction: the table at the top reports the best single family per day band.
decision_rule: >-
  Quantile score in points of capacity, equal weight per farm, month-block resampling of the 12
  scored months at 98.75% (Bonferroni across four contrasts per day band: candidate minus XGBoost,
  candidate minus `climatology`, candidate minus the candidate without lagged power, and the
  candidate minus the best single family). The threshold for porting a blend over one model family
  is a 1% fall in the quantile score, because a blend adds stages. Stop rule: if the candidate does
  not beat `climatology` by at least 3% in a day band from day 5 onward, report the lead at which
  the forecast stops adding skill and do not tune the heads at longer leads.
benchmark: >-
  `climatology` and `manual_heuristic` (experiment 224's definitions; the heuristic keeps only
  analogue weeks whose observation precedes the run), XGBoost, and the plant alone
row_set_id: >-
  Scored-year (2025-09 to 2026-08) daylight half-hours at leads of 12 to 252 hours from the 00 UTC
  ensemble run that every arm holds, per day band
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), MAE of the
  median, pinball loss and coverage at 0.01, 0.05, 0.95, and 0.99, per day band, farm (A to F), and
  history setting
code: ml-experiments/00310-metered-farm-long-lead-scoring/
---

**Plan.** Day band `d` takes, for each half-hour, the 00 UTC ECMWF ensemble run whose lead is
`12 + 24(d-1)` to `36 + 24(d-1)` hours. Bands are day 1, day 2, day 3, days 4 to 5, days 6 to 7, and
days 8 to 10. The ensemble steps are 3 hours up to lead 144 h and 6 hours after that, so from day 6
the shortwave is a 6-hour mean spread over the twelve half-hours by the cosine of the zenith, and the
temperature is interpolated between 6-hourly stamps. Each band is refitted separately with monthly
refits on all earlier months (screening schedule, not weekly). The farm's own power enters at a lag of
`ceil(lead / 24)` days, so it precedes the run. Arms: `climatology`, `manual_heuristic`, the plant
alone (P2), the additive model (GAM), the mixture-density network (MDN, three seeds), XGBoost, the
blend of the three, and the GAM without lagged power.
