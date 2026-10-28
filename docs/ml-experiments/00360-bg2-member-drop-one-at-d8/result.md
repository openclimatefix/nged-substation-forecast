---
id: '00360'
slug: bg2-member-drop-one-at-d8
title: >-
  Which of BG2's five members explains its lower score than the GAM stack at substation D8?
theme: production-reference-and-best-guess
also_relevant_to:
  - retraining-and-training-history
status: done
kind: other
size: >-
  simple. Stored: private files under data/ml-experiments/00360-bg2-member-drop-one-at-d8/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: one (re-blend stored member
  forecasts by BG2's own rule). Callers: the script imports experiment 87's assembly code and
  experiment 355's stored D8 rows. Asset graph: not touched.
hypothesis: >-
  BG2's lead over the GAM stack at D8 is spread across its five-member blend, so no single member
  explains it.
prediction: >-
  Written before any score is read. Dropping any one member moves D8's quantile score by less than
  0.2 points (the five-member blend scores 7.593; the GAM stack scores 8.338 weekly). The best single
  member alone scores within 0.3 points of the blend.
decision_rule: >-
  Descriptive, no threshold: each drop-one arm minus the five-member blend, paired, whole valid months
  resampled (20,000 resamples, seed 0), 95% intervals. `supported` when every drop-one difference
  is under 0.2 points; `refuted` when one is 0.2 or more with an interval above zero; `inconclusive`
  otherwise. Gate: the re-blended five-member arm reproduces 7.593 on the 96,986 rows of experiment
  354 before any other arm is read.
benchmark: >-
  The five-member blend (BG2), the GAM stack (experiment 355 arm W), manual_heuristic, climatology
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of the series' effective
  capacity; median error in MW; q05 to q95 coverage
verdict: supported
best_arm: >-
  The analogue member alone, 7.472 (95% interval 6.882 to 8.052) against the five-member blend's 7.593
  (6.934 to 8.264), 96,986 rows at D8.
headline: >-
  No single drop moves D8's quantile score by more than 0.09 points in either direction (largest
  rise +0.082 without the weekly XGBoost member, largest fall -0.090 without the XGBoost stack
  member). Four of the five members score 7.47 to 7.69 alone, all lower than the GAM stack's 8.338,
  and the XGBoost stack member alone scores 8.562.
lessons:
  - >-
    BG2's lower score at D8 is not a blending gain: the blend scores 7.593 and the best single
    member 7.472, so the lower score comes from the members' own forecasts.
  - >-
    The XGBoost stack member alone scores 8.562 at D8, and dropping it improves the blend's score by
    0.090 points (-0.224 to +0.008), so it does not explain BG2's lower score.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  BG2's lower score than the GAM stack at substation D8 does not come from blending five forecasts.
  Four of its members (weekly XGBoost, monthly XGBoost, analogue, pooled by substation type) each
  score 7.47 to 7.69 alone, against 8.338 for the GAM stack, and dropping any one member changes the
  blend's score by less than 0.1 points. The XGBoost stack member alone scores 8.562, so the lower
  score more likely comes from what the other four members learn. This is one substation and one
  scored year.
advice:
  - >-
    Look for the difference between BG2's score and the GAM stack's at D8 in the four non-stack
    members' own inputs, not in the blend, because each of those members alone scores 7.47 to 7.69
    against the GAM stack's 8.338 (drop-one table).
  - >-
    Do not add blend members to close the D8 gap, because dropping any single member moves the score
    by at most 0.09 points (drop-one table).
code: ml-experiments/00360-bg2-member-drop-one-at-d8/
---

**Take-home message: BG2's lower score at substation D8 is not a gain from blending.** BG2 is the
second "best guess" forecast for Primary substations (the substations that step 33 kV or 66 kV down
to 11 kV): a blend of five forecasts. D8 is the Primary substation at Marsh Lane. The GAM stack is
the generalised additive model forecast of Primary demand research default 1, refitted weekly
(experiment 355). Scores are the quantile score, twice the mean pinball loss over 11 quantile
levels, in points (percent of effective capacity; lower is better). Four of BG2's five members score
7.47 to 7.69 alone, against 8.338 for the GAM stack. Dropping any one member moves the blend's 7.593
by at most 0.09 points. The XGBoost stack member alone scores 8.562, so the lower score comes from
the other four members' own forecasts. The prediction written before any score was read held: no
drop moved D8's score by 0.2 points or more.

**What we did.** We rebuilt BG2's blend from experiment 87's stored member forecasts, with no new
fits. The five members are weekly-refitted XGBoost, monthly-refitted XGBoost, an analogue model, a
model pooled by substation type, and the XGBoost stack (the N1 model with a 14-day error
correction). We re-blended the members with one dropped, re-choosing the blend's monthly weights
from earlier months each time, and scored each member alone. Experiment 87 reports drop-one
ablations for the Primary substations as a group and none for D8 on its own. The rows are the
96,986 half-hourly forecast rows of experiments 353 to 355 at D8. D8's effective capacity is 10.4 MW,
so one point is 0.104 MW. The re-blended five-member arm reproduces BG2's stored 7.593.

**Caveat.** The result is interim: one substation and one scored year. The experiment cannot show why
one member scores best.

![Quantile score at D8 for each blend and each member alone](figure_1_members_at_d8.svg)

## Score at D8 for each blend and each member alone

| Forecast | Score | 95% interval | Minus five-member blend | Mean absolute error of the median (MW) | q05 to q95 coverage |
|---|---|---|---|---|---|
| All five members (BG2) | 7.593 | 6.934 to 8.264 | 0 | 1.440 | 0.904 |
| Without weekly XGBoost | 7.675 | 7.025 to 8.333 | +0.082 (+0.006 to +0.180) | 1.456 | 0.895 |
| Without monthly XGBoost | 7.588 | 6.930 to 8.257 | -0.005 (-0.016 to 0.000) | 1.439 | 0.904 |
| Without analogue | 7.660 | 6.997 to 8.334 | +0.067 (+0.048 to +0.085) | 1.458 | 0.906 |
| Without pooled by type | 7.594 | 6.934 to 8.265 | +0.001 (-0.001 to +0.003) | 1.440 | 0.904 |
| Without XGBoost stack | 7.503 | 6.890 to 8.108 | -0.090 (-0.224 to +0.008) | 1.419 | 0.929 |
| Weekly XGBoost alone | 7.560 | 6.913 to 8.198 | -0.033 (-0.167 to +0.080) | 1.432 | 0.927 |
| Monthly XGBoost alone | 7.694 | 7.003 to 8.413 | +0.101 (-0.088 to +0.312) | 1.455 | 0.928 |
| Analogue alone | 7.472 | 6.882 to 8.052 | -0.121 (-0.255 to -0.013) | 1.408 | 0.925 |
| Pooled by type alone | 7.633 | 7.130 to 8.123 | +0.040 (-0.220 to +0.244) | 1.460 | 0.957 |
| XGBoost stack alone | 8.562 | 7.746 to 9.408 | +0.969 (+0.679 to +1.287) | 1.624 | 0.825 |

Intervals are 95%, from 20,000 resamples of whole valid months (seed 0). Differences are in points.
The nominal q05 to q95 coverage is 0.90.

## Control check at three other Primary substations (D2, D3, D4)

Each substation has 121,492 half-hourly rows: all stored rows with a lead of up to 168 hours and an
observed power value.

- **D2:** five-member blend 2.181; drops move it by -0.003 to +0.053 points (largest: without the XGBoost
  stack, interval +0.026 to +0.081); the best single member (weekly XGBoost) scores 2.221.
- **D3:** blend 1.661; drops move it by -0.002 to +0.020; the best single member scores 1.673.
- **D4:** blend 2.403; drops move it by -0.020 to +0.021, all intervals include zero; the best single
  member (weekly XGBoost) scores 2.374, 0.029 better than the blend.
- The blend is within 0.03 points of the best single member at D3 and D4, and 0.04 points better
  at D2. The pooled-by-type member alone is the worst of the five at each of the three.

## Checks

- **Mean forecast.** The stored member forecasts hold quantiles only, so there is no mean forecast
  to compare with the median. The mean absolute error of the median in the table ranks the arms in
  the same order as the quantile score for the arms with the lowest scores (analogue alone, 1.408
  MW; the blend without the XGBoost stack, 1.419 MW).
- **A member beats the blend that contains it.** The analogue member alone scores 0.121 points
  better than the five-member blend (interval -0.255 to -0.013). A possible reason is that the
  blend's weights are chosen on earlier months across all 16 Primary substations, so they fit no
  single substation. We did not test this reason, and no independent audit of the method has run.
  The control check at D2, D3, and D4 shows no member beating the blend by more than 0.03 points.

## Caveats and what to try next

- **Blend weights.** Each arm's weights are re-chosen over all 16 Primary substations, so the D8
  numbers also reflect the other 15 substations.
- **Actual values.** D8 uses experiment 355's observed power, which models the meter's fold at the
  MVA level. The control check uses the raw observed series.
- **Scorer.** The 11 quantile levels are averaged without weights, as in experiments 252 to 355.
  Experiment 87's mass-weighted scorer gives 9.6 for the same BG2 rows, so the two scores are not
  comparable.
- **One way to look further.** Compare the inputs of the four non-stack members with the GAM
  stack's at D8.
- **Code.** `ml-experiments/00360-bg2-member-drop-one-at-d8/` imports experiment 87's folder
  (`e00087_assemble.py`, `e00087_common.py`).
