---
id: '00314'
slug: primary-demand-gated-or-pinball-band-head
title: >-
  Does a gated or pinball-trained band head pass the no-harm test that experiment 311's band head
  failed?
kind: change
theme: calibration-and-uncertainty
also_relevant_to:
- production-reference-and-best-guess
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00314-primary-demand-gated-or-pinball-band-head/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible (gate or retrain;
  the plan pre-registers both and their combination). Callers: this folder's scripts, which import
  experiment 311's rows and fits and experiment 278's feature builder. Trains a small neural head on
  the CPU; no download. Sonnet only, by the maintainer's instruction on 2026-10-23.
hypothesis: >-
  Experiment 311's mixture-density band head lowered the quantile score of Primary demand research
  default 1 but harmed sunny daylight half-hours, and its gain rested on D10. Keeping the default's
  bands on sunny daylight half-hours (gate), training the head directly on the 11-level pinball loss
  (quantile network), or both, removes the harm and leaves a gain that does not depend on one
  Primary.
prediction: >-
  Written before any score is read; each can fail. (P1) Arm G (gated mixture head) keeps at least
  80% of experiment 311's pooled gain (-0.037 points), meaning a gain of at least 0.030 points.
  (P2) Arm G passes the no-harm test on sunny daylight half-hours (one-sided 95% upper bound of the
  harm at most 0.05 points). (P3) The gain of arm G without D10 is at least 0.02 points, with a 99%
  interval wholly below zero. (P4) Arm P (pinball quantile network) matches experiment 311's
  mixture head within 0.01 points of pooled quantile score. Stop rule: if no arm passes no-harm on
  sunny daylight and P3 fails, report that the band head is a D10-specific fix and stop; no arm S.
decision_rule: >-
  Primary demand research default 1 is the default. Arms: R (the default), G (the experiment 311
  mixture head, with the default's bands kept on sunny daylight half-hours), P (the pinball
  quantile network), GP (P gated as G). Experiment 311's arm M is reported beside them from its
  stored forecasts. The gate flag is experiment 311's protected group, `sunny & daylight` from the
  stored rows of experiment 251, fixed before scoring. k = 3 contrasts (G, P, GP minus R), so each
  gain interval is read at the level 1 - 0.01/3 (99.67%), with month-block resampling. At most one
  arm is adopted. The change is a new head, so the minimum gain is 0.03 points by the research-default
  rule page. Condition 1 (gain): the interval wholly below zero and the point estimate at most
  -0.03. Condition 2 (no harm, one-sided 95% upper bound on the pooled six Primaries): at most 0.02
  points on the headline; at most 0.05 on each of the 0.01 and 0.99 pinball tails, on cold winter
  nights, and on sunny daylight half-hours; at most 2% relative on twCRPS at the 95th percentile.
  Condition 3 (replication, replicate year 2024-25, the stored bands of experiment 169 as
  reference): the same sign and an estimate of at least half of the smaller of the main estimate and
  0.03. Condition 4 (controls): leaving out any single Primary keeps a point estimate of at most
  -0.02 points and a one-sided 95% upper bound below zero; the reproduction gate and the median
  check pass; no audit trigger is open. Condition 5 (serving): every input is available at forecast
  time. Verdict `supported` if an arm meets conditions 1 to 5, `small gain` if the interval is below
  zero but the estimate is under the threshold, `not supported` if the interval includes zero,
  `refuted` if it lies wholly above zero.
benchmark: >-
  Primary demand research default 1 (stored forecasts of experiments 220 and 251, arm 1), with
  `manual_heuristic`, `climatology`, and BG2 beside every table.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score over 11 levels (twice the mean pinball loss), in points of effective capacity;
  twCRPS at the 95th percentile; pinball loss and coverage at q01, q05, q95, q99.
code: ml-experiments/00314-primary-demand-gated-or-pinball-band-head/
date: 2026-10-23
---

**Planned.** Question: can experiment 311's band-head gain survive a repair of its two failures,
the harm on sunny daylight half-hours and the dependence on D10? Rows, metric, refit cadence
(monthly, 24 rolling months, three seeds), and inputs are those of
[experiment 311](../00311-primary-demand-mixture-density-band-head/result.md); the pinball head is a
monotone quantile network on the same inputs. Caveat: leads 0.5 to 168 hours are scored, so leads of
7 to 10 days are not and NGED's horizon is 1 to 10 days.
