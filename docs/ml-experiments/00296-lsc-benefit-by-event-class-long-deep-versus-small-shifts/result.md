---
id: "00296"
slug: lsc-benefit-by-event-class-long-deep-versus-small-shifts
title: >-
  How much of the value of knowing about load shifts comes from long and deep shifts, and how much from smaller ones?
theme: lsc-detectors
also_relevant_to:
  - curtailment-cost
status: planned
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00296-lsc-benefit-by-event-class-long-deep-versus-small-shifts/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible for the
  frequency of each class and for how a class-restricted oracle is scored, so both are fixed below
  before any score is read. Callers: this folder's scripts only; they are copies of experiment 283's
  modules. Asset graph: not touched.
hypothesis: >-
  Load shifts of at least 2 days and at least 10% of a Primary substation's load carry most of the
  benefit of knowing every load shift exactly, and shifts that are shorter than 12 hours or under
  10% of load carry little. This is the question behind the "long-and-deep" detector of the LSC
  decision program.
prediction: >-
  Written before any score is read. In the central frequency scenario, an oracle that knows only the
  shifts of at least 2 days and at least 10% of load (PERFECT_LD) captures at least 70% of the
  benefit of the oracle that knows every shift, on the threshold-weighted CRPS (twCRPS, 95th
  percentile of earlier power) against the true normal-arrangement series and on the curtailment
  cost against observed power, in the rows within 7 days of a shift. An oracle that knows only the
  shifts shorter than 12 hours or under 10% of load captures under 15%. The prediction fails if the
  short or small shifts carry more than 30% of the benefit, and then a detector for smaller shifts
  is worth building and the cells that carry the benefit set its target.
decision_rule: >-
  Fixed before any score is read. Arms (experiment 283's machinery, unchanged): K0 (Kalman filter on
  recent power, no knowledge), PERFECT (every shift known at its true onset, readings inside it set
  aside, end known at once), PERFECT_LAG (the same after a live service's delays), PERFECT_LD (only
  shifts of at least 2 days and at least 10% of load known), PERFECT_SMALL (only the others known),
  and the same four with the known megawatts subtracted from the readings and added back to the
  forecast (the _ADD arms, the design experiment 283 found necessary for curtailment cost against
  observed power). Injected worlds: new worlds, one Primary substation each, covering a grid of 6
  durations (2 to 6 h, 6 to 24 h, 1 to 3 d, 3 to 7 d, 7 to 30 d, over 30 d) by 4 depths (5%, 10%,
  25%, 50% of the site's load), constant and demand-shaped, half receivers and half donors, 24
  shifts per cell across the six scored Primaries, at least 7 days apart and away from every logged
  closure. Solar-shaped shifts are not run. The benefit of an arm in a cell is the sum over the cell's
  shifts of the K0 score minus the arm's score over the rows within 7 days of the shift. Because
  shifts sit at least 7 days apart, the benefit of a class-restricted oracle in a cell equals the
  benefit of PERFECT where the cell is known and zero where it is not; this is checked against the
  explicitly run PERFECT_LD and PERFECT_SMALL arms. The ladder arms (minimum duration 12 h, 6 h,
  2 h; minimum depth 25%, 10%, 5%) are computed this way. Annual benefit is the sum over cells of the
  per-shift benefit times the shifts per Primary-year of the cell. Three frequency scenarios (low,
  central, high) come from the sources listed in the write-up, with the undercounted classes named.
  The headline contrast is the share of the PERFECT benefit captured by PERFECT_LD, per metric and
  row group, with 95% intervals from resampling whole shifts within each cell. Metrics: twCRPS at the
  95th and 99th percentile of power before the scored year, CRPS, the error of the median and of
  the mean, and the node-only curtailment cost of `studies.node_curtailment` at the 0.5 scaled
  limit (GBP 2,000 per MWh; a relative comparison between arms at a Primary, no money saved), 12
  to 36 hour leads, with 0.75 and 0.9 swept. Verdict: `supported` when both shares at least 70%
  (point estimate) and the short or small share is under 15% on twCRPS and on cost; `partly
  supported` when it holds on one of the two metrics; `not supported` otherwise. The threshold
  is zero because the quantity is a share of an existing benefit and not a port decision. Realistic
  shares (weighting each cell by a detector's recall) are a follow-up and are not scored here:
  Part 1 of the long-deep detector brief has not run.
benchmark: >-
  Experiment 283's gains of PERFECT over K0 (twCRPS +23.1% in the 48 hours after a shift of 25% of
  load or more that lasts over a day).
row_set_id: >-
  New injected worlds of this experiment on the same GAM-K cached member forecasts as experiment
  208 and 283; not comparable with the leaderboard
metric: >-
  Share of the PERFECT benefit captured by a restricted oracle, on twCRPS against the true normal
  arrangement and on node-only curtailment cost against observed power
code: ml-experiments/00296-lsc-benefit-by-event-class-long-deep-versus-small-shifts/
production_status: not-for-production
adopted_in: none
date: 2026-10-23
---

**Planned.** The write-up replaces this page when the scores are in. The front matter holds the
hypothesis, the prediction, and the decision rule, written before any score was read.
