---
id: "00341"
slug: wind-capacity-steps-from-strong-wind
title: >-
  How small a change in a wind farm's usable capacity can its output in strong wind show, and how
  fast?
theme: generator-forecasting
also_relevant_to: []
status: planned
date: 2026-10-24
kind: other
size: >-
  medium. Stored: private files under data/ml-experiments/00341-wind-capacity-steps-from-strong-wind/
  only (the capacity-state flag table). Serving path: not touched, because no forecast is produced.
  Degradation rule: not touched. Designs: several are defensible (the statistic, the block length,
  the penalty), and this plan pre-registers one grid. Callers: this folder's scripts, which read
  experiment 230's exported cores and expected power and the observed power of W1 to W3. No code is
  imported from another experiment's folder. One Sonnet review of code and write-up, then one
  Sonnet prose review.
hypothesis: >-
  For the three metered wind farms (W1, W2, W3), a change-point search on the upper quantiles
  (99.9th percentile and maximum) of observed power in the half-hours when ERA5 wind is well above
  the speed at which the fitted power curve saturates finds a loss of 25% of capacity within 60
  days at W1 and W2, with fewer false alarms than experiment 233's ratio monitor under a 9% seasonal
  swing. Steps come in whole turbines. Weighting lower-wind half-hours by the inverse of their
  speed-error variance speeds detection up. A neighbouring farm in the same ERA5 cell cancels the
  shared speed error.
prediction: >-
  Written before any run. (P1) The 99.9th percentile and the maximum in the strong set find a 25%
  step lasting at least 90 days within 60 days at more than 80% of start dates at W1 and W2, and
  at fewer than 50% of start dates at W3; a one-turbine step is found at W3 (50%) and not at W1
  (7.7%). (P2) The change-point search on the strong-set 99.9th percentile raises fewer false
  alarms under the 9% swing than experiment 233's post hoc rows. (P3) A step of 10% or less is
  found at no farm within 60 days. (P4) The state-space fit tells a clipped farm from a farm with
  turbines out in at least 80% of 14-day episodes of 25% size at W1 and W2, and cannot at W3. (P5)
  Episodes shorter than 7 days are found in under half of the injected cases at W3 and under 70%
  at W1 and W2. (P6) On the real series W1 shows at least one change point of about 10% to 25% of
  scale that explains part of its gap to its registered capacity. (P7) The weighted-evidence
  detectors B and C detect a 25% step lasting 90 days with a median delay at least 20% shorter than
  A at all three farms, and raise no more false alarms than A under the 9% swing; C beats B under a
  5% speed bias over 30 days. (P8) A neighbouring farm under 30 km detects a one-turbine step at
  W1 and W2 within 60 days at more than 50% of start dates; the benefit is gone beyond 60 km. P8 is
  untestable if the public-farm count shows no farm within 60 km, and the write-up then says so.
decision_rule: >-
  Kind `other`, so there is no adoption rule. Each of P1 to P8 is reported as held, failed, or
  untestable, and the verdict is `supported` if at least six held, `partly supported` if three to
  five held, and `not supported` otherwise. Detection share and delay are per farm over 30 random
  start dates (fixed seed, 2024-07-01 to 2025-12-31, the same 30 for every cell), with 95%
  intervals from a bootstrap over whole calendar months of start dates. A start already flagged on
  the unmodified series within one block of its start is excluded and counted. The smallest
  step found at 80% power within 60 days is the smallest injected size whose share found within 60
  days is at least 0.8, per farm and per arm. Penalty rule: the change-point penalty is
  c x (robust noise variance) x ln(n) with c = 2 pre-registered. If the unmodified calibration
  window (2021-09-01 to 2024-06-30) raises more than one change point per farm-year at c = 2, the
  experiment reports the trade-off curve for c in 1, 2, 4, 8, 16, picks the smallest c with at most
  one per farm-year at every farm, writes it to `work/penalty.json` before the real series is
  analysed, and uses it for every arm. Cap: 5 hours.
benchmark: >-
  Experiment 233's rows (arm D: every half-hour at 20% of scale or more, unweighted ratio),
  re-implemented here with the same change-point search. No earlier experiment tested the
  strong-wind quantile statistic.
row_set_id: >-
  none shared. All half-hours of W1 to W3 from 2019-09-01 to 2026-06-30 with an observed power value
  and an expected power from experiment 230's export, dropping the first weeks after a new meter as
  experiment 230 does. No valid time on or after 2026-07-01 is used.
metric: >-
  Share of injected changes found within 60 days and the median delay in days, per farm, arm, size,
  and duration; excess change points per farm-year on the unmodified series and under the controls;
  the confusion table of the state-space fit; the real series' change points and episodes
---

Planned. See the front matter for the hypothesis, predictions, and decision rule. The design follows
the worker brief of 2026-10-24, summarised here:

- **Strong set.** ERA5 100 m speed at least 2 m/s above the speed where experiment 230's core
  reaches 95% of its fitted scale, and at least 2 m/s below its cut-out speed (1 and 3 m/s as
  sensitivity).
- **Statistics.** Non-overlapping blocks of 3, 7, 14, 28, and 56 days; the 90th, 99th, and 99.9th
  percentiles and the maximum of observed power over scale; missing when a block holds fewer than 20
  strong half-hours.
- **Detectors.** An exact penalised least-squares change-point search (optimal partitioning) and a
  two-sided CUSUM; a lattice-constrained search and a state-space (Viterbi) fit with states
  nameplate, k turbines out, and an export clip.
- **Arms.** A: strong set only. B: inverse-variance-weighted log ratio of observed over expected
  power from 25% to 100% of scale. C: B with a per-block speed-scale nuisance term. D: experiment
  233's rows. E: W2 against W3 (shared ERA5 cell). F and G: public neighbours (counted first).
- **Known-answer test.** Steps and episodes of 1 turbine, 2 turbines, 10%, 25%, 50%, and half a
  turbine, lasting 3 to 365 days; export clips at 90%, 75%, and 50% of the farm's 99.9th percentile;
  seasonal swings of 9%, 15%, and 25%; a drift of 1% per year; a 5% speed bias over 30 days.
