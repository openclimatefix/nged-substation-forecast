---
id: '00289'
slug: summer-gap-weather-input-discriminator
title: >-
  Does the ensemble sunshine forecast cause the April to June gap between the best-guess stack and
  BG2?
theme: production-reference-and-best-guess
also_relevant_to:
  - differentiable-physics-embedded-generation
  - solar-disaggregation
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00289-summer-gap-weather-input-discriminator/
  only. Serving path: not touched. Degradation rule: not touched. Designs: two are defensible (a
  zero-fit re-forecast with realised sunshine, or a refit with realised sunshine as the driver); this
  plan pre-registers the zero-fit one and names the refit as the fallback if the result is
  undecided. Callers: this folder's scripts, which call reviewed `studies.*` modules and copy
  experiment 111's and 136's forecasting code without changing it. No refits, no GPU, no downloads.
hypothesis: >-
  The best-guess stack's weaker April to June score than BG2 at the decision Primaries comes from
  the ensemble's sunshine forecast (or its ratio calibration), and not from demand absorbing the
  solar term or from the error-correction layers.
prediction: >-
  Written before any forecast is re-made. (P0, gate) Re-forecasting the base engine with the
  stored ensemble weather reproduces its stored mean to within 1e-6 points at every row. (P1) On
  daylight half-hours in April to June at leads of 0 to 36 hours at D5, D9, D10, and D16, arm D's
  quantile-score gap to BG2 shrinks by at least half when the ensemble's sunshine is replaced by realised (CAMS)
  sunshine. (P2) The mean absolute error of the mean forecast shrinks by at least a third of its gap to BG2
  on the same rows. (P0b) The irradiance that the engine builds from CAMS agrees with the raw CAMS
  hourly values to within 1 W/m² on average, with no half-hour shift in the hour-of-day profile. If P1 fails, the gap stays at least half as large and the cause is demand absorbing
  the solar term or the error-correction layers.
decision_rule: >-
  The gap is arm D minus BG2 in quantile score on April to June daylight half-hours at leads of 0
  to 36 hours, at D5, D9, D10, and D16, averaged over the four Primaries so that no one Primary
  decides. C2 (primary) is arm D with ensemble sunshine minus arm D with realised sunshine in mean
  absolute error of the mean forecast, on the same rows; C1 (secondary) is the same contrast in
  quantile score. Intervals resample whole ISO weeks of forecast initialisation time (about 13
  weeks) at the 97.5% level (Bonferroni over two contrasts). The ensemble sunshine is implicated if
  C2 is at least a third of arm D's mean absolute error gap to BG2 and its interval excludes zero,
  and C1 is at least half of the quantile-score gap. It is cleared if C2's upper bound is below a
  third of that gap. Anything else is undecided and the page names the refit as the next step. If
  C1 and C2 disagree the page reports "undecided" and says which one moved. C1 holds the stored
  quantile widths fixed, so it cannot show that realised sunshine would narrow the spread; a
  cleared verdict says only that the level of the mean is not the cause. Realised sunshine is a
  ceiling, so every gain is an upper bound for a better forecast. The threshold is a fraction of
  the gap, because a better sunshine input would add a data source, so zero is not the right
  threshold. A negative control repeats the swap in July to September, where the stack beats BG2:
  the swap should not help there by more than it helps in April to June.
benchmark: >-
  BG2 (experiment 87 p_s6_bg2) on the 561,207 rows of experiments 228, 251, and 266 (six substations,
  2025-07-02 to 2026-06-30, 09:00 UTC runs, leads 0.5 to 168 hours, outside hand-filtered load
  shifts).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity),
  and mean absolute error of the mean forecast, in points.
code: ml-experiments/00289-summer-gap-weather-input-discriminator/
---

**Question.** [Experiment 266](../00266-summer-gap-diagnostics-solar-term/result.md) found that the
best-guess demand stack trails BG2 in April to June at three Primaries, and cleared the ensemble
sunshine forecast only as far as an agreement test goes. This experiment asks directly: if the stack
is given the sunshine that actually happened, does its gap to BG2 close?

## Plan

- **Weather swap.** Arm D's base engine is experiment 136's A2 (monthly fits, ensemble-driven, with
  the Kalman filter). Re-forecast it with the ensemble's shortwave replaced by realised CAMS
  irradiance at the forecast's target half-hours only (the analysis proxy and the filter's past
  residuals stay as stored), with the irradiance ratio set to 1 and the solar shrink to 1. The
  monthly fits were trained on calibrated ensemble irradiance, so the realised input is slightly out
  of distribution; a second arm with ensemble irradiance and the same identity calibration
  separates the calibration effect from the input effect. Temperature stays the ensemble's, so only sunshine changes. Nothing is refitted.
- **Arm D with realised sunshine.** Add the per-row change in the base engine's mean to every
  quantile of stored arm D, the method experiment 251 used for its clip arm. The offset tables,
  winter night offset, and recipe R of arm D are not re-simulated; this is a caveat.
- **Rows.** The six stored Primaries, April to June daylight half-hours (clear-sky irradiance above
  20 W/m²); the decision Primaries D5, D9, D10, and D16 are the pooled headline, and D11 and D14
  are reported beside them.
- **Gate (P0).** The re-forecast with ensemble weather must reproduce experiment 136's stored A2
  mean.
- **Stratification.** Results are reported by lead band (0 to 24, 24 to 72, 72 to 168 hours) and
  per substation; the decision uses 0 to 36 hours. A lead of 72 hours or more is an upper bound
  only, since no forecast has that sunshine.
- **Reported:** absolute quantile score and mean absolute error for BG2, arm D with ensemble
  sunshine, arm D with realised sunshine, and the base engine alone; tail coverage and pinball loss
  at 0.01, 0.05, 0.95, and 0.99; per-substation results.
- **Caveat.** CAMS sunshine exists only after the fact, so arm D with realised sunshine is a
  ceiling for an improved sunshine input, not a deployable forecast.
- **Checks.** Median minus mean forecast per site, the mean-forecast error beside the quantile
  score, and the audit triggers, recorded in the write-up.
- **Earlier advice followed.** Judge any change on April to June at D5, D9, and D10 as well as the
  whole year (experiment 266).
