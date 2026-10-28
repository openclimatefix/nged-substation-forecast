---
id: '00360'
slug: bg2-member-drop-one-at-d8
title: >-
  Which of BG2's five Primary members carries its lead over the GAM stack at D8?
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
  No single drop moves D8's quantile score by more than +0.082 points (largest: dropping the weekly
  XGBoost member, 95% interval +0.006 to +0.180). Four of the five members score 7.47 to 7.69 alone,
  all below the GAM stack's 8.338; only the XGBoost stack member alone is worse, at 8.562.
lessons:
  - >-
    BG2's lead at D8 is not a blending gain: the blend scores 7.593 and the best single member 7.472,
    so the lead comes from the members' own models.
  - >-
    The XGBoost stack member alone scores 8.562 at D8, and dropping it lowers the blend by 0.090
    points (-0.224 to +0.008), so it carries none of the lead.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  BG2's better score at D8 than the GAM stack does not come from blending five models. Four of its
  members (weekly XGBoost, monthly XGBoost, analogue, pooled by substation type) each score 7.5 to 7.7
  alone, against 8.34 for the GAM stack, and dropping any one member from the blend changes the score
  by less than 0.1 points. The stack member of BG2 alone scores 8.56, so the difference lies in what
  the other four models learn, and the next experiment should compare their inputs with the GAM
  stack's. This is one substation and one scored year.
advice:
  - >-
    Look for D8's gap to BG2 in the XGBoost and analogue members' own inputs, not in the blend, because
    each of those members alone scores 7.5 to 7.7 against the GAM stack's 8.338 (arms table).
  - >-
    Do not add blend members to close the D8 gap, because dropping any single member moves the score
    by at most 0.082 points (drop-one table).
code: ml-experiments/00360-bg2-member-drop-one-at-d8/
---

**Take-home message: BG2's lead at D8 is not a gain from blending.** Four of BG2's five members score
7.47 to 7.69 alone on the quantile score, against 8.338 for the GAM stack (Primary demand research
default 1, weekly refit, experiment 355). Dropping any one member moves the blend's 7.593 by at most
+0.082 points. The XGBoost stack member (the N1 model with the 14-day correction) scores 8.562 alone,
which is worse than the GAM stack, so the lead belongs to the other four members' models. The
prediction written before any score was read held: no drop moved D8 by 0.2 points or more.

We asked which of BG2's five members explains its lead at D8 (Marsh Lane). We rebuilt BG2's blend from
experiment 87's stored member forecasts, with no new fits. We then re-blended the members with one
dropped (re-choosing the monthly blend weights each time) and scored each member alone. Experiment 87
reports drop-one ablations for the Primaries as a group and none for D8 on its own. The scores are the
quantile score in points of capacity (one point is 0.1 MW at D8; lower is better), on the 96,986 rows
of experiments 353 to 355. The re-blended five-member arm reproduces BG2's stored 7.593. This is an
interim result: one substation, one scored year, and blending gains are not a causal account of why a
single model leads.

![Quantile score at D8 for each blend and each member alone](figure_1_members_at_d8.svg)

## Arms, D8 (96,986 rows, 95% intervals resampling whole valid months, 20,000 resamples, seed 0)

| Arm | Score | 95% interval | Minus five-member blend | Median error (MW) | q05 to q95 coverage |
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

## Placebo Primaries (all stored rows up to 168 h with a finite actual, 121,492 rows each)

- **D2:** five-member blend 2.181; drops move it by -0.003 to +0.053 (largest: without the XGBoost
  stack, interval +0.026 to +0.081); the best single member (weekly XGBoost) scores 2.221.
- **D3:** blend 1.661; drops move it by -0.002 to +0.020; the best single member scores 1.673.
- **D4:** blend 2.403; drops move it by -0.020 to +0.021, all intervals include zero; the best single
  member (weekly XGBoost) scores 2.374, 0.029 better than the blend.
- At the placebos the blend is within 0.03 of the best single member, and the pooled-by-type member
  alone is the worst of the five at each.

## Checks

- **Median minus mean forecast:** the stored member forecasts hold quantiles only, so no mean forecast
  exists to compare; the median error in MW is in the table above.
- **Mean-forecast error beside the quantile score:** the median error moves with the score in every
  arm (lowest for analogue alone, 1.408 MW, and for the blend without the stack, 1.419 MW).
- **Triggers:** trigger 5 fires in a weak form. Analogue alone beats the blend that contains it by
  0.121 points (interval -0.255 to -0.013). A plausible reason is that the blend weights are chosen
  on earlier months across all 16 Primaries and so fit no single substation. This reason was not
  tested here, and no audit has run. The other triggers do not apply (no control, one site, no
  contradicted prediction, no later design depends on the result).

## Caveats and what to try next

- **Caveat.** Interim, one substation, one scored year; the blend weights are re-chosen per arm over
  all 16 Primaries, so the D8 numbers also reflect the other 15 substations.
- **Rows.** D8 uses experiment 355's actual (the MVA-fold series). The placebos use the raw series.
- **Scorer.** The 11 levels are averaged without weights, as in experiments 252 to 355; experiment
  87's mass-weighted scorer gives 9.6 for the same BG2 rows and is not comparable to the 7.593.
- **Next.** Compare the inputs of the weekly XGBoost and analogue members with the GAM stack's at D8,
  since the lead sits in those models.
- **Code.** `ml-experiments/00360-bg2-member-drop-one-at-d8/` imports experiment 87's folder
  (`e00087_assemble.py`, `e00087_common.py`).
