---
id: '00359'
slug: d8-gam-bg2-quantile-blend
title: >-
  Does averaging the GAM stack's quantiles with BG2's beat BG2 alone at D8?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
size: >-
  simple. Stored: private files under data/ml-experiments/00359-d8-gam-bg2-quantile-blend/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: one defensible design (a
  quantile-by-quantile weighted average of two stored forecasts). Callers: the scripts read experiment
  355's and experiment 354's stored score tables and call experiment 354's scoring helpers. Asset
  graph: not touched.
hypothesis: >-
  At D8 (Marsh Lane), the GAM stack carries information that BG2 lacks, so averaging the two
  forecasts' quantiles scores better than BG2 alone.
prediction: >-
  Written before any new score is read. The 0.5 blend scores below BG2's 7.593 on the whole scored
  year and on its second half (2026-01 to 2026-06), by at least 0.2 points of quantile score.
  Experiment 355 found that neither BG2's median nor BG2's spread alone closes most of the stack's
  0.745-point gap, which suggests the two forecasts err differently. The prediction fails if the
  0.5 blend scores at or above 7.593 on the second half.
decision_rule: >-
  Weights 0.25, 0.5, and 0.75 on the GAM stack are chosen on the first half of the scored year
  (valid months 2025-07 to 2025-12) by the lowest mean score, and reported on the second half
  (2026-01 to 2026-06) and on the whole year. Four planned paired contrasts, month-resampled (20,000
  resamples), 95% intervals as the brief states (the 99% Bonferroni intervals are printed beside
  them): C1 the 0.5 blend minus BG2 on the second half; C2 the same on the whole year; C3 the
  first-half-chosen blend minus BG2 on the second half; C4 the 0.5 blend minus the GAM stack on the
  second half. Stop rule: if C1 is not wholly below zero, report and stop. Threshold for `supported`:
  C1 wholly below zero and C2 below -0.2 points, because a blend runs two stacks in production and
  the gain has to pay for the second one. `partly supported` when C1 is wholly below zero but C2 is
  not below -0.2; `not supported` otherwise. The placebos at D2, D3, and D4, and the lead-band,
  monthly, and sunny April-to-June slices, are descriptive. Gate: the script must reproduce 8.338
  (GAM stack, weekly refit, own-residual band) and 7.593 (BG2) before any blend is read.
benchmark: >-
  BG2, climatology, and manual_heuristic as stored by experiment 87; experiment 355's weekly-refit
  D8 stack (8.338) as the GAM stack
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of the series'
  effective capacity (1 point is 0.1 MW at D8); median error in MW; q05 to q95 coverage
code: ml-experiments/00359-d8-gam-bg2-quantile-blend/
date: 2026-10-28
---

## Plan

- **Question.** Does the GAM stack (D8's research default 1 with a weekly refit and an
  own-residual band, experiment 355's arm `W`) add information to BG2, so that the mean of their
  quantiles beats BG2? "BG2's XGBoost stack" is a different BG2 member and is not used here.
- **Arms.** BG2 alone; GAM stack alone; blends `0.25`, `0.5`, `0.75`, where the number is the
  weight on the GAM stack and each of the 11 quantiles is averaged separately. No fits and no
  production code. The shared 96,986 rows are experiment 354's.
- **Weight choice.** By the lowest mean score on the first half of the scored year.
- **Slices.** Lead bands 0 to 24, 24 to 72, and 72 to 168 hours; month; sunny April-to-June
  half-hours; tail calibration (coverage inside q05 to q95, share below q01 and above q99).
- **Placebos.** The same blend at D2, D3, and D4 (Primaries with no engine), with the stored stack
  forecast of experiment 354's arm 1 as the GAM stack, because the own-residual band raised the
  score there.
- **Band construction (read-only check).** The result states in two sentences how the GAM stack
  builds its band, after reading experiments 252 and 354.
- **Caveat on every headline number.** Interim: one substation, one scored year, weights chosen
  on a half-year.
