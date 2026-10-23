---
id: '00311'
slug: primary-demand-mixture-density-band-head
title: >-
  Does a mixture-density neural network for the forecast bands lower the quantile score of Primary
  demand research default 1, with the gain in the protected tail groups?
kind: change
theme: calibration-and-uncertainty
also_relevant_to:
- production-reference-and-best-guess
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00311-primary-demand-mixture-density-band-head/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (the inputs, the loss, and what the mixture is centred on); the plan pre-registers one
  and a second arm that keeps the cold-day widening. Callers: this folder's scripts, which import
  the feature builder of experiment 278 and the scoring pattern of experiment 251. Trains a new
  neural head on the CPU; no download. Sonnet only, by the maintainer's instruction on 2026-10-23.
hypothesis: >-
  Primary demand research default 1 (the GAM stack A+H+S of experiments 220 and 251) gets its bands
  from a smooth spread table and a cold-day widening of the upper offsets, around a median shifted
  by the neural head. A mixture-density neural network, which reads the weather state and may be
  skewed, gives bands with a lower quantile score, and the gain sits in the protected tail groups.
prediction: >-
  Written before any score is read; each can fail. (P1) The default plus the mixture-density band
  head lowers the six-Primary quantile score by at least 0.02 points. (P2) At least 60% of that
  gain sits in the protected groups (the 0.01 and 0.99 tails and cold winter nights), counted as the
  share of the total gain that comes from rows in those groups. (P3) The 0.95 coverage of the
  mixture-density bands on ordinary rows (rows in no protected group) is within 2 points of 95%.
  (P4) The gain at D8-like misfit Primaries is not applicable, because D8 is not in the row set, so
  the page says nothing about D8. Stop rule: if the arm worsens the headline by more than 0.02
  points on the first six months of the scored year (2025-07 to 2025-12), stop the fits and report.
decision_rule: >-
  Primary demand research default 1 is the default; the two pre-registered arms are the mixture-density band
  head (arm M) and the same with the cold-day widening kept on top (arm MW), so k = 2 and the gain
  interval is read at the level 1 - 0.01/2 (99.5%). At most one arm is adopted. The change is a new
  head, so the minimum gain is 0.03 points by the research-default rule page (the brief for this
  experiment quoted 0.02 for a new fitted step; the rule page owns the value, and P1 keeps the
  brief's 0.02 so the prediction can be scored on either). Condition 1 (gain): the month-block
  interval of (arm minus default) on the six-Primary quantile score lies wholly below zero and the
  point estimate is at most -0.03. Condition 2 (no harm, one-sided 95% upper bound of the harm on
  the pooled six Primaries): at most 0.02 points on the headline; at most 0.05 points on each of the
  0.01 and 0.99 pinball tails (pinball loss of the single level, not doubled), on cold winter nights
  and on sunny daylight half-hours; at most 2% relative on twCRPS at the 95th percentile. Each
  Primary alone is reported with no pass or fail. Condition 3 (replication, replicate year
  2024-25): the same sign and a replicate estimate of at least half of the smaller of the main
  estimate and 0.03. Condition 4 (controls): the controls below pass and no audit trigger is open.
  Condition 5 (serving): every input is available at forecast time. Verdict `supported` if an arm
  meets conditions 1 to 5, `small gain` if the interval is below zero but the estimate is under the
  threshold, `not supported` if the interval includes zero, `refuted` if it lies wholly above zero.
benchmark: >-
  Primary demand research default 1 (stored forecasts of experiments 220 and 251, arm 1), with
  `manual_heuristic`, `climatology`, and BG2 (the XGBoost research benchmark `p_s6_bg2` of
  experiment 87, which is not the production forecast) beside every table.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score over 11 levels (twice the mean pinball loss), in points of effective capacity;
  twCRPS at the 95th percentile; pinball loss and coverage at q01, q05, q95, q99.
code: ml-experiments/00311-primary-demand-mixture-density-band-head/
production_status: not-for-production
---

**Plan.** The question is whether a neural network that outputs a mixture of Gaussian curves can
give Primary substation demand bands that score better than the bands of Primary demand research
default 1 (the GAM stack A+H+S), which are a smooth spread table plus a widening on forecast-cold
days. Mixture-density heads gained 1.3% on metered solar farms in [experiment
287](../00287-metered-farm-spread-binned-plant-heads/result.md) and were the best single family in
[experiment 300](../00300-metered-farm-best-guess-assembly/result.md). The head here follows the
design of experiment 300 and the Primary features of [experiment
278](../00278-stack-retraining-cadence/result.md).

- **Arms.** Default: arm 1 of experiments 220 and 251 (stored; median, bands, and solar summary
  unchanged). Arm M: the median of arm 1, with every other quantile set by the mixture-density head,
  which is the average over 3 seeds. Arm MW: arm M with the cold-day widening of the default applied
  to the head's upper offsets. At most one arm is adopted.
- **The head.** A three-component Gaussian mixture on the residual of arm D's mean forecast from the
  actual power, in units of the stored band half-width, with inputs of experiment 278's arm 2 (arm
  D's mean, the Kalman filter's level and anomaly, the stored 5% to 95% width, lead, ensemble-driver
  weather, calendar flags, and the Economy 7 flag). The head is trained on the negative
  log-likelihood. Its quantiles are shifted so that its median equals the default's stored median.
  This keeps the median unchanged and avoids training on the in-sample residuals of the neural
  median head H.
- **Cadence.** The head is refitted at each month start on the rows of the preceding 24 months whose
  target lies before the refit date, with every fourth month held out for early stopping, as
  experiment 278 does. Experiments 220 and 251 hold no replicate year, because their neural median
  head and solar summary were built for the scored year only. The replicate year therefore compares
  the head with the stored bands of arm A (experiment 169) around arm D's median, and is labelled so.
- **Cold-day widening.** The default's widening multiplies the upper offsets at q75, q90, q95, and
  q99 by 1.653, 1.363, 1.261, and 1.162 on rows whose ensemble-mean 24-hour temperature is below 2 C
  (read from the stored forecasts of experiment 169, identical for all six Primaries).
- **Rows.** `fold_a_daily_gap_v1`; the six Primaries D5, D9, D10, D11, D14, D16 (the set of
  experiments 220 and 251, 561,207 rows); scored year 2025-07-02 to 2026-06-30; replicate year
  2024-25 (the even months, which experiment 169 did not tune on); outside hand-filtered events;
  leads 0.5 to 168 hours. **Caveat:** NGED's horizon is 1 to 10 days, so leads of 7 to 10 days are
  not scored.
- **Controls.** (1) The default's six-Primary quantile score reproduces experiment 251's stored
  figure within 0.005 points. (2) An assertion that every training target of every refit lies
  before its refit date. (3) The arm MW equals the arm M on rows whose ensemble-mean temperature is
  at or above 2 C. (4) The head's median is shifted onto the default's, so the arm M median equals
  the default median exactly (assertion).
- **Reported groups.** Each Primary alone; cold winter nights; sunny daylight half-hours; Christmas
  and bank holidays; the tails q01, q05, q95, q99 with coverage; the Economy 7 step as an example
  week, and the manual heuristic, climatology, and BG2 beside every table.
- **Resources.** CPU only, 6 cores, one torch thread per worker, systemd-run with MemoryMax=16G.
  Weather at lag times is the ensemble weather available at forecast time. The scored months stay
  inside 2025-07-02 to 2026-06-30, and no valid time on or after 2026-07-01 is scored.
