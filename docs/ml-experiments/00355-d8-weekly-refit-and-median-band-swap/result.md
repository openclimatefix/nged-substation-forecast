---
id: '00355'
slug: d8-weekly-refit-and-median-band-swap
title: >-
  Does refitting D8's stack weekly close its gap to BG2, and how much of the gap is the median
  rather than the band?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
  - calibration-and-uncertainty
status: planned
kind: other
size: >-
  medium. Stored: private files under data/ml-experiments/00355-d8-weekly-refit-and-median-band-swap/
  only. Serving path: not touched. Degradation rule: not touched. Designs: the cadence test has one
  defensible design (same fit, cuts every 7 days); the swaps are arithmetic on stored scores.
  Callers: the scripts import experiments 252, 264, and 354 and call their fit, forecast, and band
  code. Asset graph: not touched.
hypothesis: >-
  (H1) Refitting D8's stack weekly instead of monthly is not the main reason BG2 beats the stack at
  D8. (H2) Most of the stack's remaining gap to BG2 sits in the band, not the median.
prediction: >-
  Written before any new score is read. H1: weekly minus monthly is under 0.2 points of quantile
  score in magnitude at D8 (monthly refits as stored, fit b with arm 3 bands scored 8.382 against
  BG2's 7.593), and quarterly minus monthly is under 0.3 points. H2: swapping BG2's spread onto the
  stack's median closes more than half of the stack's remaining gap to BG2, and swapping the stack's
  own-residual band onto BG2's median closes under a third of the gap in the other direction.
decision_rule: >-
  Paired month-resampled contrasts (20,000 resamples), 99% intervals (Bonferroni for four): C1
  weekly minus monthly at D8, all rows; C2 quarterly minus monthly; C3 weekly minus monthly in the
  first three scored months; C4 the stack's median with BG2's spread minus the stack as stored.
  Threshold for C1: the interval wholly below -0.2 points, because weekly refits multiply the
  fitting work by about 4.3 for every Primary. `supported` when C1 is not wholly below -0.2 and the
  monthly-to-BG2 gap is mostly band (the H2 split holds); `refuted` when C1 is wholly below -0.2;
  otherwise `inconclusive`. Gate: arm M must reproduce experiment 354's stored score for fit b with
  arm 3 bands (8.382) and BG2 7.593 before any weekly score is read.
benchmark: >-
  BG2, climatology, and manual_heuristic as stored by experiment 87; experiment 354's fit b with
  arm 3 bands as the monthly arm
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of the series'
  effective capacity (1 point is 0.1 MW at D8); median error in MW; q05 to q95 coverage; error on
  sunny daylight half-hours
code: ml-experiments/00355-d8-weekly-refit-and-median-band-swap/
verdict: null
production_status: not-for-production
adopted_in: none
date: 2026-10-26
---

**Plan (before any score is read).** The write-up replaces this section.

- **Question.** D8 (Marsh Lane) scores 8.38 with the best stack of experiment 354 against BG2's
  7.59. BG2 refits weekly and the stack refits monthly. The maintainer asked, on 2026-10-26, whether
  weekly refits close the gap, and how the gap splits between the median and the band.
- **Cadence arms (fits at D8).** The stack of experiment 354's fit b (from 2023-09 with a level
  step per hour band at 2025-04-01, stage-2 arm D, own-residual band by hour block and season):
  (M) monthly cuts as stored, (W) a cut every 7 days, (Q) every third monthly cut. Only data
  available at each cut is used.
- **Swap arms (no fit).** On arm M's rows: (S1) arm M's median with BG2's spread (BG2 quantile
  minus BG2 median, added to arm M's median); (S2) BG2's median with arm M's own-residual band; (S3)
  BG2's median with the stack's donated spread; M and BG2 as stored. Report the score, median error
  in MW, and q05 to q95 coverage.
- **Placebos.** The cadence contrast at D2, D3, and D4 (stored default window, no level step).
- **Slices.** Sunny daylight half-hours and the first three scored months, separately.
- **Scoring.** Scored year 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours, the 96,986 rows of
  experiment 354, month-resampled 95% and 99% intervals.
- **Weather at forecast time.** Same ensemble forecast weather and past-weather proxy as experiment
  252, so every input exists at forecast time.
- **Earlier advice followed.** Experiment 354: fit b with the own-residual band. Experiment 278: weekly
  equalled monthly on six other Primaries, so a null is the prior.
- **Compute.** CPU only, 6 spawned workers, one thread each, `MemoryMax=16G`.
