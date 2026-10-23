---
id: '00306'
slug: summer-gap-solar-interactions-with-temperature-and-day-type
title: >-
  Does letting the solar term interact with temperature and day type close the April-to-June gap to
  BG2?
theme: production-reference-and-best-guess
also_relevant_to:
  - differentiable-physics-embedded-generation
  - differentiable-physics-forecast-engine
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00306-summer-gap-solar-interactions-with-temperature-and-day-type/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one is defensible
  (the interaction columns as a new `studies.dp_gam` option or inside the experiment's copy of
  experiment 114's design code; the heating driver as one averaged hinge or one column per
  smoothing window; the columns sign-bounded or free); this plan pre-registers the experiment-local
  copy, one averaged hinge, and free signs, and names the alternatives as next steps. Callers: this
  folder's scripts, which copy the machinery of experiments 114, 136, and 289 and call reviewed
  `studies.*` modules unchanged. CPU only, 6 cores, thread caps 1 to 2.
hypothesis: >-
  The April-to-June daylight excess of arm D (the convex demand model with a Kalman filter on recent
  errors) over BG2 (the production-style XGBoost forecast) at the substations D5 and D9 is an
  additive-structure problem. On cold clear spring days heating demand and rooftop solar output are
  both large, and an additive model must give the solar term one weight on every day. Experiment 299
  found the D5 excess largest on clear midday half-hours (+1.67 points). Letting the solar shape
  interact with temperature (shape times the heating hinge) and with day type (shape times a
  weekend indicator) lets the model give those days a different solar weight.
prediction: >-
  Written before any refit is scored. Each line can fail. Excess means arm D's quantile score minus
  BG2's on April-to-June daylight half-hours at leads up to 36 hours (the rows of experiment 289's
  headline). (P1) The interaction arm (both columns) closes at least 30% of the excess at D5 and at
  D9, each taken separately: closure is (control excess minus arm excess) divided by control
  excess, where the control is arm D rebuilt by this experiment's own pipeline. (P2) At the
  placebo substations D10 and D16, the mean quantile score of the interaction arm moves by under
  0.02 points in absolute value (April to June, same rows). (P3) July to September worsens by
  under 0.02 points, averaged over D5 and D9. (P4) A placebo column (the solar shape times the
  parity of the day of the year) closes under 10% of the excess at D5 and D9. If the closure at D5
  and D9 is under 10%, the excess is not an additive-structure problem.
decision_rule: >-
  Intervals are 95% intervals that resample whole ISO weeks of forecast initialisation time, the
  same weeks at every substation (14 weeks in April to June), as in experiment 289. Supported when
  P1 to P3 all hold and the lower bound of the interval of the D5-and-D9 average reduction in score
  is above zero. Refuted when the point closure averaged over D5 and D9 is under 10%. Partly
  supported otherwise (closure between 10% and 30%, or P1 holding at one of D5 and D9). Small
  gain when the interval of the reduction excludes zero but the closure stays under 30% and P2 and
  P3 hold. The threshold for porting is any reduction whose interval excludes zero with P2 and P3
  holding, because two extra coefficients in a convex fit add no model, no data source, and no
  pipeline step; the 30% bar is for how much of the excess the interaction must explain to be the
  cause. P4 failing (the placebo closing 10% or more) voids a supported verdict, because any extra
  column would then do as well.
benchmark: >-
  BG2 (experiment 87 p_s6_bg2) on the 561,207 rows of experiments 228, 251, 266, and 289 (six
  substations, 2025-07-02 to 2026-06-30, 09:00 UTC runs, leads 0.5 to 168 hours, outside
  hand-filtered load shifts).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity),
  mean absolute error of the median in points, and the share of the April-to-June excess closed.
code: ml-experiments/00306-summer-gap-solar-interactions-with-temperature-and-day-type/
---

**This page is a plan; the write-up replaces it when the fits finish.**

We ask whether adding two columns to arm D's convex fit, the rooftop-solar shape times a heating
hinge and the solar shape times a weekend indicator, closes the April-to-June gap to BG2. Arm D is
the engine of experiment 136's arm A2 (a monthly refit of experiment 114's T-all model, driven by
the calibrated ensemble-mean weather, with a Kalman filter on recent errors). Nothing in arm D is
changed except the two columns; a control rebuilt through the same pipeline checks that the
pipeline reproduces the stored arm D.

## Arms

- **Control (`a2`).** Experiment 136's stored monthly T-all fits, run through this folder's
  forecast pipeline. Gate: its per-row scores match the stored arm D of experiment 251 to 1e-4.
- **`s_both` (the experiment).** T-all plus `shape * mean heating hinge` and `shape * weekend`.
- **`s_heat`, `s_wknd`.** Each column alone, to attribute any closure.
- **`s_placebo`.** T-all plus `shape * parity of the day of the year`.
- The shape is the derated unit solar shape (`studies.dp_gam.derated_unit_shape`). The heating
  driver is the hinge `relu(15.5 - T) / 10` averaged over the four smoothed temperatures. The
  weekend indicator is 1 on Saturdays, Sundays, and bank holidays. The new columns are free in sign
  and carry the heating columns' ridge (0.1). The existing pinned solar subtraction is unchanged.

## Plan notes

- **Advice followed.** Stratify by site and season, and test the solar term at D5 and D9 on clear
  half-hours, as experiments 289 and 299 advise; keep the correction layers unchanged, as 299
  advises.
- **Scope.** Cadence is monthly refits, as arm D. No head, no export clip, no coasting.
- **Tails.** Coverage and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles are reported for
  the control and `s_both`. The node curtailment metric is not computed, because this is a
  diagnostic of one term of the model.
