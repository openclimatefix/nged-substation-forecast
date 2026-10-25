---
id: "00348"
slug: wind-plant-mdn-upper-tail
title: >-
  Does fixing the upper tail of the frozen physics plant plus mixture head let it beat Wind farms
  research default 1 on the three metered wind farms?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: planned
date: 2026-10-25
kind: change
size: >-
  medium. Stored: private head outputs and predictions under
  data/ml-experiments/00348-wind-plant-mdn-upper-tail/ only. Serving path: not touched (the page
  says what serving each arm needs). Degradation rule: not touched. Designs: several are defensible
  for the tail (loss weights, a point mass, a monotone increment network, a residual band), so the
  plan fixes five before any score is read. Callers: this folder's scripts, which import experiments
  230, 303, 310, 326, 336, and 347 and call no reviewed module beyond `studies.*`. Sonnet only, one
  code review and one prose review.
hypothesis: >-
  Experiment 347's best arm (y2: Stage 3 frozen physics plant, 51 ensemble members through it, a
  three-component Gaussian mixture head on the summary of the resulting power values) beats Wind
  farms research default 1 by 0.234 points over true days 1 to 10 but is over-confident in the
  upper tail (2.3% of observations above q99, 0.99 pinball loss up to 0.084 worse). One of five
  tail fixes (z1 tail-weighted loss, z2 capped quantiles with a point mass at the plant's scale, z3
  monotone increments above q90, z4 z1 plus z2, z5 the solar recipe of a plant residual band blended
  with the default's XGBoost quantiles) brings the share above q99 under 1.5% and the 0.99 pinball
  loss to within the tolerance, while keeping the headline gain.
prediction: >-
  Written before any score is read. z4 brings the share above q99 from 2.3% to under 1.5% and the
  0.99 pinball loss to within 0.03 of the default's. z2 does most of that at W1 and W2 and less at
  W3. z1 alone lowers the 0.99 pinball loss by under 0.03 and raises the median-band loss by under
  0.03. z5 lowers the 0.99 pinball loss by at least 10% against the default, puts the share above
  q99 under 1.5%, and is within 0.1 of z4 on the headline. The headline gain of every arm over the
  default stays within 0.05 of y2's (-0.234 over days 1 to 10), and no arm lowers the headline by
  more than 0.05 beyond y2. A head fed only the plant's median reproduces at least 70% of the gain.
decision_rule: >-
  The five adoption conditions of the research-default page, read in the order z4, z5, z3, z2, z1;
  the first arm that passes all five is adopted (at most one). Condition 1: the 99.8% month-block
  interval (level 1 - 0.01/5) of (arm minus default) on the kept-row quantile score over true days
  1 to 10 lies wholly below zero and the point estimate is at least the minimum gain (0.03 for z1,
  z3, z4, new heads; 0.02 for z2 and z5, new fitted steps). Condition 2: the one-sided 95% upper
  bound of the harm to the pooled 0.01 and 0.99 pinball losses is at most 0.05 points each (0.02
  for the headline's tolerance). Condition 3: the replicate (hindcast 2024-04 to 2024-12, scored on
  the daily runs of 2025-01 to 2025-06) has the same sign and a t test on monthly differences does
  not contradict it. Condition 4: reproduction gate (the y2 refit matches experiment 347's within
  0.01 on the kept rows), shuffled-summary placebo, five seeds, leakage check, and no audit trigger
  unresolved. Condition 5: the page says what serving needs. A winner on condition 1 that fails
  another stays a candidate.
benchmark: >-
  The default: Wind farms research default 1 (the research-default log names no wind version yet;
  experiments 259, 339, and 347 use the ERA5-trained XGBoost on the 51 members plus experiment
  259's offset table, 11.352 points on the kept rows over every lead), and no later version is
  logged. Beside it, descriptive: y2 as in experiment 347, XGBoost alone, XGBoost with the adapter
  column (experiment 326 arm c), experiment 324's W4.
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline, all half-hours beside). Hindcast for the head: 2024-04-01 to 2025-06-16.
metric: >-
  Quantile score over 11 levels in points of capacity, headline scope true days 1 to 10 (lead 24 to
  240 hours, kept rows), with days 1 to 2, days 3 to 10, and every lead beside; tolerance 0.02 for
  the headline and 0.05 for the 0.01 and 0.99 pinball losses.
code: ml-experiments/00348-wind-plant-mdn-upper-tail/
---

**Take-home: planned; no score has been read.**

## Plan

- **Base.** Experiment 347's frozen Stage 3 plant and stored power-ensemble summaries (median,
  mean, standard deviation, 10th and 90th percentile, share above saturation speed), the same
  pooled head, hindcast rows, early stopping, and seeds. The head sees ensemble weather only.
- **z1.** Quantiles from the mixture's CDF, trained on the pinball loss over the 11 levels with
  weight 3 at q01, q90, q95, q99 and weight 1 elsewhere. Gradients pass through the quantile by
  implicit differentiation of the CDF. Ceiling 1.0 as in y2.
- **z2.** The mixture gets a point mass at the farm's fitted scale (`scale` of the Stage 3 plant)
  with weight `pi` output by the head (init near 0.05, reads the saturation share); quantile =
  `min(G^-1(tau / (1 - pi)), scale)`. Trained with the unweighted pinball loss, so the extra
  reference `y2p` (z1's loss, all weights 1, no point mass, ceiling 1.0) separates the loss from
  the tail components.
- **z3.** The head's mixture supplies q01 to q90; a small network (16 hidden units; inputs the
  plant's p90, saturation share, lead) gives q95 and q99 as non-negative increments above q90
  (softplus), capped at the scale. Trained with z1's weighted pinball loss.
- **z4.** z1's weighted loss with z2's point mass and cap.
- **z5 (added 2026-10-25 by the maintainer's Discuss sessions, before any score was read).** On
  stored quantiles: (a) per-farm offsets, `Q_tau(actual - plant quantile_tau)` by bin of the plant's
  mean (6 quantile bins) and the ensemble spread (3 terciles), fitted on the hindcast; (b) an
  additive band, `plant median + Q_tau(actual - plant median)` per mean bin and lead band (under
  24 h, 24 to 72 h, 72 to 240 h, over 240 h); (c) the plant side is (b), blended per quantile and
  lead band with the default's XGBoost quantiles, `w` in steps of 0.1 chosen by pinball loss on the
  scored months before the one being predicted (weight 0.5 for the first month). Each family is
  reported alone beside the blend. No neural network. The default's XGBoost forecasts exist only
  for the scored year, so the blend weights use earlier scored months, and the replicate cannot
  score z5.
- **Controls.** A base-only head per arm (plant median, lead, farm), the shuffled-summary placebo,
  five seeds (replicate three), the reproduction gate (y2 refit against experiment 347), the
  leakage check (plant frozen, head sees ensemble weather only).
- **Compute.** Heads on one GPU and CPU, under 1 GPU hour, up to 14 cores as separate processes
  (never a forked pool), `OMP_NUM_THREADS=1`, `MemoryMax=16G` per job.
- **Serving.** z1 to z4 serve as y2 does (the plant on each member, then the head); z5 needs the
  plant, a stored table, and the default's XGBoost quantiles.
