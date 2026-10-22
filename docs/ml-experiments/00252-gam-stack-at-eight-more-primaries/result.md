---
id: '00252'
slug: gam-stack-at-eight-more-primaries
title: >-
  Does the best-guess demand stack score within 0.05 points of BG2 at the eight Primaries that have
  no stack forecast yet?
theme: production-reference-and-best-guess
also_relevant_to:
  - differentiable-physics-embedded-generation
  - calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00252-gam-stack-at-eight-more-primaries/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (which pieces count as "the stack", and how the MVA reading enters the fit), so the
  plan fixes one: experiment 247's chain (T-all monthly fits, shared filter noise, winter night
  offset, recipe R, pinned solar capacity, 60% clip), fitted on the apparent-power reading as
  experiment 152's arm N does. Callers: this folder's scripts, which copy experiments 247, 152, and
  228 and call reviewed `studies.*` modules without changing them. Asset graph: not touched.
hypothesis: >-
  The best-guess GAM stack (GAM-K with the heating split by day type, shared filter noise, pinned
  solar capacity, ensemble-mean weather, winter night offset, and the 60% PV clip) scores within
  0.05 points of BG2 over leads 0.5 to 168 hours at D2, D3, D4, D8, D12, D19, D21, and D22, so that
  every substation card with a BG2 line can also show a best-guess line and example weeks.
prediction: >-
  Written before any score is read. (1) A0 minus BG2 is -0.12 to +0.05 points pooled over the 8
  sites, and at least 5 of 8 sites are at or below +0.05. (2) D8 (multi-day drops, probable
  embedded gas peaker) and D12 (reverse flow, MVA) are the two worst sites against BG2, and A0 is
  worse than BG2 at D8 by 0.1 or more. (3) A1 minus A0 is within +/-0.02 at sites with no PV
  estimate. (4) A2 minus A0 is -0.03 to 0.00 at the sites with an inferred PV capacity of 2 MW or
  more, and within +/-0.01 elsewhere.
decision_rule: >-
  Per-site and pooled paired month-resampled bootstrap (20,000 resamples, 95%) of A0 minus BG2 on
  the quantile score in points of capacity. Supported when the pooled upper bound is at most +0.05
  and at least 6 of 8 sites have a point estimate at or below +0.05. A site that fails gets no
  best-guess line on its card, and the card says so. A2 is adopted only if it closes at least 50%
  of the sunny-daylight gap to BG2, the bar of experiment 250, because a neural component adds a
  fitted model per site. Cards need no gain, only a fair comparison, so the card decision does not
  depend on beating BG2. Control gate: the stored stack scores at D5, D9, D10, D11, D14, and D16
  must be reproduced within 0.001 before any contrast is read.
benchmark: >-
  BG2, climatology, and manual_heuristic as stored by experiment 87 and experiment 152
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in MW and in points of capacity,
  MAE of q50, the card's condition splits (sunny daylight, winter nights, cold nights, Christmas,
  bank holidays), and q95 coverage on cold nights
code: ml-experiments/00252-gam-stack-at-eight-more-primaries/
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

**Planned; no score has been read.** Fourteen substation cards have a BG2 line, but only D5, D9,
D10, D11, D14, and D16 have a best-guess line and example weeks. This experiment fits the same
stack at D2, D3, D4, D8, D12, D19, D21, and D22 and asks whether it stays within 0.05 points of
BG2.

## Arms

**Plan amendment, committed before any score was read.** The brief assumed a pinned solar capacity
"from the card's own inference where one exists". No real-time capacity vintage exists for the eight
sites (experiment 45's vintages cover the MW Primaries only), so the pinned solar term is zero at
all eight. That changes three arms.

- **A0 (verdict arm).** Experiment 247's chain with a zero pinned solar term: T-all monthly fits
  (heating split by day type), shared filter noise (POOL), ensemble-mean weather, winter night
  offset, and recipe R (experiment 155's arm D), fitted on the apparent-power reading as experiment
  152's arm N does. The 60% clip acts on the pinned solar output, so it is inert at zero capacity.
  Experiment 169's band fixes are not applied (their code is tied to the six sites).
- **A1 (dropped).** A0 without the solar term is A0 itself, so the arm is empty.
- **A2 (not run).** The monotone network of experiment 250 acts on a non-zero clipped solar term,
  and none exists here, so prediction 4 is untested.
- **Control gate.** The same code, reading experiment 111's stores, reproduces experiment 155's
  stored arm D at D10 and D11 (maximum difference 0.0 in the median and the mean).
- **References.** BG2, climatology, and the manual heuristic of experiment 87.

## Sites, rows, and caveats

- **Sites.** D2, D3, D4, D8, D12, D19, D21, D22. Control: D5, D9, D10, D11, D14, D16.
- **Rows.** `fold_a_daily_gap_v1`, scored year 2025-07 to 2026-06, outside hand-filtered load-shift
  events, no scoring after 2026-07-01.
- **Schedule.** Monthly refits on the 24-month window (experiment 247's schedule); the contrast does
  not depend on the cadence.
- **Compute.** CPU, 4 cores, about 4 hours.
- **Risks.** D8 and D12 may lack the 24 months of clean history the stack needs (D8 is scored from
  2025-04-01); a shortfall is reported rather than hidden. D12's unit is MVA, so its score is not
  comparable in MW. The pinned capacity comes from a method that recovers about 0.78 of an injected
  capacity (experiments 231 and 241), and the Kalman filter may absorb a wrong pin.
