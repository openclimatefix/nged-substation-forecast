---
id: '00294'
slug: regime-aware-model-at-d8-engine-state-term
title: >-
  Does a regime-aware model beat a single model at a Primary with a persistent generator regime?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private prediction files under
  data/ml-experiments/00294-regime-aware-model-at-d8-engine-state-term/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one defensible (which engine state
  the forecast sees, and which history the single model trains on), so the plan fixes both below.
  Callers: this folder's scripts, which copy experiment 191's GAM-K engine and experiment 291's
  engine detector and call reviewed `studies.*` modules unchanged. Asset graph: not touched.
hypothesis: >-
  D8's metered power excludes a gas engine's output (experiment 194) while the engine, which runs
  in spells (experiment 291), takes net load below what the weather and calendar predict. A
  single model trained across those spells has no way to know which half-hours the engine ran, so
  its forecast and its bands carry the engine as noise. A model that keeps the calendar, weather,
  and solar shapes shared across the whole history but adds one regime-specific term, an offset
  per half-hour the engine runs (non-negative, taking power off net load), forecasts D8 better
  than the single model. The offset is estimated from the data, and its size is not assumed.
prediction: >-
  Written before any forecast is scored. (P1) The regime-aware model (arm 3) lowers D8's quantile
  score, our stand-in for the continuous ranked probability score (CRPS), by at least 5% against
  the single model on the whole history (arm 1), and lowers D8's lower-tail
  threshold-weighted score (twCRPS) by at least 5%. (P2) Arm 3 scores within 3% of the band-widening
  comparator (arm 4, experiment 264's post hoc baseline) on D8's quantile score. (P3) The same
  engine term does not raise the quantile score at D5, D10, or D11, which have no engine, by more
  than 0.5%. (P4) The fitted offset per running engine lies between 1 and 3 MW in at least 80% of
  the monthly refits that have a running spell in their window (the engine's recorded unit was
  1.97 MW). Each can fail. P1 fails if the band-widening comparator alone does as well and the
  engine term adds nothing beyond it.
decision_rule: >-
  Fixed before any score is read. Rows: D8, D5, D10, and D11 in the scored year (valid times from
  2025-07-02 to 2026-06-30), leads 0.5 to 168 hours, outside the hand-filtered events, on the
  stored runs of experiment 191 (seed 0). The quantile score is twice the mean pinball loss over
  the 11 quantiles, in points of effective capacity. The twCRPS is the same score applied to
  min(forecast quantile, t) and min(reading, t) with t the 10th percentile of D8's readings in the
  scored year (a lower-tail weight, computed from the readings once and used for every arm). The
  deciding arm 3 uses the engine state of the last settled day (see Plan), because the state
  at the target half-hour is built from the very reading being forecast. Intervals resample the
  12 calendar months (20,000 draws) at 98.75%, a Bonferroni correction over the four planned
  contrasts. C1: arm 3 minus arm 1, quantile score, relative to arm 1. C2: arm 3 minus arm 4,
  relative to arm 4. C3: arm 3 minus arm 1 at each control. C4: C1 on the twCRPS. `supported`: C1
  and C4 are each at most -5% with the interval wholly below zero, C2's upper end is below +3%,
  and C3's upper end is at most +0.5% at all three controls. `partly supported`: C1 holds and one
  of C2 and C3 fails, or C1 holds on the quantile score and not the twCRPS. `not supported`:
  otherwise, which includes arm 4 scoring better than arm 3 by more than 3% with the interval above
  zero. `inconclusive` if the reproduction gate fails (below). The 5% threshold is set by cost: the
  term adds an engine detector and a stored state column to the training loop and to the
  forecast's inputs. The 3% and 0.5% limits are not copied from an earlier experiment's gain:
  3% is under the noise between two sensible band-widening rules in experiment 264 (arm 3 and arm
  2b differed by 2.5%), and 0.5% is experiment 264's no-harm rule. The reproduction gate: arm 2
  (experiment 191's reference, rebuilt here) must reproduce experiment 191's stored D8 quantile score
  of 10.95 to within 0.05.
benchmark: >-
  Arm 1 (the single model), arm 2 (experiment 191's masked reference), and arm 4 (experiment 264's
  stored post hoc baseline "arm_emp" and its arm 2b), on the same keys at D8.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (CRPS stand-in), threshold-weighted quantile score, q05-to-q95 coverage, mean
  absolute error of the mean forecast, tail calibration at q01, q05, q95, and q99, and a Brier
  stand-in for the curtailment proxy, each for all rows, ordinary rows, and engine-running rows, with
  winter separately.
code: ml-experiments/00294-regime-aware-model-at-d8-engine-state-term/
---

**This page is the plan; the write-up replaces it when the run finishes.**

**The question.** D8's metered power excludes the output of a gas engine of up to three units, so
the engine's running spells appear as multi-day shortfalls (experiments
[194](../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md)
and [291](../00291-d8-drops-engine-spells-load-shift/result.md)). We ask whether a model that knows
when the engine ran forecasts D8 better than a single model that does not.

**Where the engine states come from.** The engine-state term uses the running-state probabilities
of the detector of experiments
[196](../00196-ps2-quantised-engine-detector-on-d8-with-known-answer-gate/result.md) and 291. That
detector reads the same residual in which the multi-day drops of experiment 291 were found. The term
therefore does not confirm that the drops are an engine, because the states and the drops share
one source. They test only whether the states are useful to a forecast, on readings the states were
not fitted to.

## Departures from the brief

- **The brief's single model on the whole history is run twice, because D8's usable history starts
  at two different dates.** Experiments 152 and 191 fit D8 only from 2025-04-01, after 8.91 MW of
  11 kV solar connected on 2025-03-24 and the series changed meaning (see
  [`data-processing.md`](../data-processing.md)). Every row from 2025-04-01 is already after the
  break of 2023-08-15, so a post-break indicator on those rows is constant. Arms 1 and 3 therefore
  train on the whole history (all of D8's rows, with a post-break indicator that does vary), and
  arm 2 is experiment 191's reference on rows from 2025-04-01. Whether the earlier rows help or
  harm is part of the answer, and the page reports arm 1 against arm 2 for that reason.
- **The brief's "arm D / the experiment 220 stack at D8" does not exist.** Experiment 220's stack
  covers five other Primaries and D8 has no such stack. The engine is experiment 152's GAM-K
  (generalised additive model with a Kalman filter on recent errors), run at D8 by experiment 191,
  and every arm here is built from it.

## Plan

- **Rows.** The 292 daily runs of experiment 191, forecast origin 09 UTC, scored runs from
  2025-07-01 (not the every-fifth-day stride runs), 11 quantiles, as experiment 264 scored them.
- **Arm 1, single model, whole history.** Experiment 191's GAM-K engine with its price columns
  removed, trained on all of D8's rows from the first usable instant to each monthly cut (no
  24-month window, no mask).
- **Arm 2, experiment 191's reference.** The same engine on rows from 2025-04-01 with a 24-month
  window. It is rebuilt here and must reproduce experiment 191's stored scores.
- **Arm 3, regime-aware model.** Arm 1 plus two columns: an indicator that is 1 from 2023-08-15
  (free sign), and an engine-state column, `-P(one or more units run)` at the row, with a
  non-negative coefficient so the term can only take power off net load. The Kalman filter's
  residual stream subtracts both terms at the training state. At forecast time the engine state is
  the **filtered** probability (using readings up to that half-hour only) at the same half-hour of
  the last day settled at the forecast origin, which is how experiment 191's arm `p` read prices.
- **Arm 3-hindsight (not deciding).** The same fit, with the engine state at the target half-hour
  itself. The state is built from the reading being forecast, so this arm is a leaky ceiling that
  shows how much of arm 3's gap to arm 4 is the unknown future state.
- **Arm 3-indicator (not deciding).** Arm 3 without the engine column, to separate the break
  indicator from the engine term.
- **Arm 3-window (not deciding).** Arm 3's two columns on arm 2's rows. The break indicator drops
  out there (constant), so this arm shows the engine term alone.
- **Arm 4, bands widened from past residuals.** Experiment 264's stored arms `arm_emp` (the post
  hoc baseline) and `arm2b`, on experiment 191's reference median. These cover D8 only.
- **Controls.** Arms 1 and 3 at D5, D10, and D11, with the engine states that experiment 291's
  detector gives there (it flags almost nothing on those dates). The break indicator and the engine
  column enter in the same way. The term should fit near zero.
- **Engine states.** Experiment 196's frozen detector, copied with experiment 291's code, run on
  the stored residual panel at all four Primaries. Lab shortcuts, named here because they are not
  available live: the detector's parameters were fitted on 2023-09 to 2026-01, and the expected
  power it subtracts comes from a model fitted with hindsight. The filtered probability removes
  only the smoothing's use of later readings.
- **The term's value out of sample.** Three numbers. (a) The coefficient of each monthly refit,
  in MW per running engine, against the engine's recorded unit of 1.97 MW. (b) The slope of the
  forecast error of arm 1 on the persisted state across the scored year, which is the term's value
  on readings the fit never saw, with a month-resampled interval. (c) Arm 1's and arm 3's mean
  absolute error of the mean forecast on engine-running half-hours (filtered probability above
  0.5) and on the others.
- **Metrics.** For every arm, separately for all rows, ordinary rows (filtered probability of
  0.5 or less at the target), and engine-running rows, and with December to February reported on its
  own: the quantile score, the twCRPS, mean absolute error, q05-to-q95 coverage, tail calibration
  at q01, q05, q95, and q99, and the Brier score of "the reading falls below D8's 2nd percentile",
  the stand-in experiment 264 used for the curtailment proxy of experiment
  [268](../00268-lsc-curtailment-metric-v2-member-wise-ens/result.md). Experiment 268's metric
  scores nodes with generators and does not cover D8, so the stand-in is not a cost in pounds.
- **Schedule.** Monthly refits, as in experiment 191, so a contrast of 0.05 points or more is
  safe from the cadence; the page says whether any contrast could depend on it.
- **Compute.** CPU only, 4 cores, each run under `systemd-run --user --scope -p MemoryMax=16G -p
  OOMPolicy=continue`.
- **Advice followed and departed from.** Follows 264's advice to check a widening method at sites
  with no engine, and to give the as-of fit an earlier quarter of residuals. Departs from
  experiment 191's choice of masking D8 before 2025-04-01, in arms 1 and 3 only, because the brief
  asks for the whole history.

## What the next experiment should answer

Written after the results are read.
