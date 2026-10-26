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
status: done
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
  `partly supported` when C1 is not wholly below -0.2 but the gap does not split as predicted;
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
verdict: partly supported
best_arm: >-
  Weekly refit (W) with the own-residual band scores 8.338 against the monthly refit's 8.382 and BG2's
  7.593 on 96,986 rows. W minus M is -0.045 points (99% interval -0.118 to +0.001).
headline: >-
  Weekly refits lower D8's score by 0.045 points (99% interval -0.118 to +0.001) and quarterly refits
  raise it by 0.034 (+0.003 to +0.082), so refit cadence is not why BG2 beats the stack. Swapping in
  BG2's band closes 14% of the stack's 0.79-point gap and swapping in BG2's median closes 27%.
lessons:
  - >-
    Refit cadence does not explain D8's gap: weekly minus monthly is -0.045 points, and the
    gain sits in July to September 2025 (-0.148 points, -0.309 to 0.000).
  - >-
    Neither BG2's median nor BG2's spread closes most of the gap alone; the two parts may work only as a pair.
  - >-
    BG2's median is 0.07 MW (4.6%) more accurate than the stack's, and its spread covers 90.4% of
    readings inside q05 to q95 against the stack's 84.6%.
take_home_message: >-
  Refitting D8's forecast every week instead of every month lowers its score by only 0.045 points
  of capacity, so BG2's weekly refit is not why it beats the stack at D8. The remaining 0.79-point
  gap is not mostly the band or mostly the median: BG2's band on the stack's median closes 14% of
  it and BG2's median under the stack's band closes 27%. The next step is to look at how BG2's
  median and spread are fitted together rather than at either part alone.
advice:
  - >-
    Keep monthly refits for D8's stack in R&D; weekly changed the score by -0.045 points (99%
    interval -0.118 to +0.001) (Figure 2).
  - >-
    Do not expect a donated BG2-style spread on the stack's median to close D8's gap; it closed 14%
    (Figure 2).
  - >-
    Report any weekly-refit gain separately for the first three scored months, where it is largest
    (Figure 3).
production_status: not-for-production
adopted_in: none
date: 2026-10-26
---

**Refitting D8's forecast weekly instead of monthly barely changes its score, so BG2's weekly refit
is not why it beats our model at D8, and the remaining gap does not come from the median or the band
alone.** D8 is the Marsh Lane electricity substation. BG2 is the XGBoost model that is our current
best reference forecast, and "the stack" is our additive-model forecast, which we compare against
it. We refitted the stack (the version of [experiment
354](../00354-d8-own-residual-bands-and-longer-fit-history/result.md): fitted from 2023-09 with a
step in its level at 2025-04-01, and a forecast band taken from D8's own past errors) monthly,
weekly, and quarterly. We also put BG2's median forecast, and separately BG2's band (the spread of
its quantile forecasts around the median), into the stack's forecast on the same 96,986 half-hours.
The score is the mean pinball loss over 11 quantiles, in points of D8's capacity (1 point is 0.1
MW), and a lower score is better. Weekly minus monthly is -0.045 points (99% interval -0.118 to
+0.001). BG2 stays 0.79 points better than the monthly stack, and neither swap closes most of that
gap.

![Figure 1](figure_1_gap_to_bg2.svg)

![Figure 2](figure_2_cadence_and_swaps.svg)

![Figure 3](figure_3_monthly.svg)

- **Arms.** (M) monthly refit, 12 cuts, the stored fit b of experiment 354 with its own-residual
  band by hour block and season (arm 3 there). (W)
  weekly, 52 cuts. (Q) quarterly, 4 cuts. Swaps on arm M's rows: (S1) M's median with BG2's spread,
  (S2) BG2's median with M's own-residual spread, (S3) BG2's median with the donated spread of
  experiment 252. Reuses the code of experiments 252, 264, and 354.
- **Gate.** M scores 8.382 and BG2 7.593, as stored in experiment 354.
- **Absolute scores (all rows, quantile score).** M 8.382, W 8.338, Q 8.416, S1 8.271, S2 8.171, S3
  9.995, BG2 7.593, `manual_heuristic` 8.469, climatology 9.757.
- **Cadence contrasts (99% intervals, 20,000 month resamples).** W minus M -0.045 (-0.118 to
  +0.001); Q minus M +0.034 (+0.003 to +0.082). W minus M in July to September 2025 is -0.148
  (-0.309
  to 0.000), in 2026 -0.013 (-0.033 to +0.008), on sunny half-hours -0.077 (-0.285 to +0.080).
- **Swap contrasts against M.** S1 -0.111 (-0.999 to +0.325), which closes 14% of the 0.79-point
  gap to BG2. S2 -0.212 (-0.618 to +0.232), which closes 27%. S3 +1.613 (+0.602 to +2.624), so the
  donated spread is the worst part (the finding of experiment 354). In July to September 2025 S1
  scores 8.454 against M's 9.585 (-1.131, -3.421 to +0.066), so the band difference is concentrated
  in the first three months.
- **Median error and coverage.** Median absolute error is 1.509 MW for M, 1.503 for W, 1.514 for Q,
  and 1.440 for BG2. The share of readings inside q05 to q95 is 84.6% for M, 84.7% for W, 84.6% for
  Q, and 90.4% for BG2. Share below q01: M 3.9%, W 3.8%, BG2 0.2%. Share above q99: M 2.9%, BG2
  0.2%.
- **Pre-registered predictions.** H1 held: W minus M is under 0.2 points and Q minus M under 0.3.
  H2 failed: S1 closes 14%, not more than half, though S2 closes 27%, under a third as predicted.
  Verdict `partly supported`: the cadence is not the cause, but the gap does not split into a
  band part and a median part.
- **Reading of H2 (post hoc).** The two swaps together close 41% of the gap. BG2's spread is fitted
  to
  BG2's own median errors, so swapping either part alone breaks the pairing. The pairing explanation
  is a hypothesis, not a
  tested result.
- **Not run.** The placebo cadence contrast at D2, D3, and D4 failed in the forecast step (a lead
  band had no tuning residual to fit the spread on, `ValueError` in `studies.dp_ens`). The brief
  made the placebos optional ("if cheap"), so they were dropped, and the cadence result rests on D8
  only. The cadence result therefore may not carry to other Primaries; experiment 278 found weekly
  equal
  to monthly at six other Primaries.
- **Lab shortcuts.** The own-residual band uses only errors known when the forecast is made, and
  assumes a reading arrives 6 hours after its valid time. Horizon
  is 0.5 to 168 hours, not NGED's 1 to 10 days. Weather at forecast time is the ensemble forecast,
  as
  in experiment 252.
- **Checks.** Median minus mean forecast (points of capacity): M -1.44, W -1.37, Q -1.71, which is
  the own-residual band's shift of the median from the filter's mean, as in experiment 354. Error of
  the mean forecast beside the median's: M 1.505 against 1.509 MW, W 1.500 against 1.503 MW, so the
  cadence contrast has the same sign on the mean forecast. Audit triggers (the `ml-experiment`
  skill's seven conditions): none fired. The reproduction gate
  held, no result contradicts its prediction, and only one site was scored.
- **What to try next.** Fit BG2's median and spread jointly on the stack's residuals at D8 (for
  example a model of the stack's error on BG2's error), rather than swapping parts. Test weekly
  against monthly again at a Primary with long history before adopting weekly refits anywhere.

## Plan as registered before any score was read

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
- **Earlier advice followed.** Experiment 354: fit b with the own-residual band. Experiment 278:
  weekly
  equalled monthly on six other Primaries, so a null is the prior.
- **Compute.** CPU only, 6 spawned workers, one thread each, `MemoryMax=16G`.
