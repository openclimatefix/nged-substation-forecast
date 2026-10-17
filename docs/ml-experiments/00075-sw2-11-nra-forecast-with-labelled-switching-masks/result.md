---
id: "00075"
slug: sw2-11-nra-forecast-with-labelled-switching-masks
status: done
size: complex
hypothesis: >-
  On the 292 daily ENS forecast runs of experiment 48, a weekly-refitted Primary substation demand
  model whose training targets leave out the labelled switching half-hours forecasts the normal
  running arrangement (NRA) no worse than the plain weekly model, and giving the model the signed
  MW that switching moved at each lagged power input, or replacing each lag by the lag minus that
  MW, lowers the quantile score on half-hours outside switching.
prediction: >-
  Masking the training targets changes the NRA-row score by -0.01 to +0.01 points of capacity, since
  experiment 44 bounded a perfect detector's gain at 0.34% of 5.44 (0.02 points). Adding the signed
  MW as three columns changes the score by -0.03 to +0.01 points, and replacing the lags by the
  lags minus the MW changes it by -0.03 to +0.01. Neither magnitude interval is expected to exclude
  zero, because only 22 labelled events fall in the scored year.
decision_rule: >-
  Three planned contrasts on the NRA rows (the scored half-hours outside NGED's logged closures
  and outside the labelled events), quantile score over all leads, read at 98.33% (Bonferroni
  across three), whole validation months resampled. C1, the masked arm minus the plain arm: the
  mask is adopted (non-inferior) when the interval's upper end is at most +0.02 points, 0.34% of
  the plain arm's score, which is the most that experiment 44 says a perfect detector could gain,
  and so the size of harm worth worrying about; only the upper end of the two-sided interval is
  read, which is conservative; C1 is a gain when the interval lies wholly below zero. C2, the arm
  with signed MW columns minus the masked arm, and C3, the arm with lags minus MW minus the masked
  arm: each is supported when its interval lies wholly below zero, because either arm needs a
  detector on the serving path. The v1 NRA candidate is the supported magnitude arm; when both are
  supported, the one with the lower score on the NRA rows, and the page reports `plain` minus that
  arm descriptively. With no magnitude arm supported, the masked arm is the candidate when C1 is
  non-inferior and the plain arm otherwise. Every other comparison is descriptive.
benchmark: >-
  The plain weekly arm on the same rows, experiment 48's weekly row (5.44 on all rows), the 14-day
  corrected stack (5.64), `manual_heuristic`, and `climatology`, each from experiment 48's stored
  arms.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels), in points of capacity,
  on the NRA rows; the metered view (all half-hours) and the lag-in-event subset are reported and
  decide nothing
verdict: inconclusive
best_arm: >-
  None separable from the others: on the 2,928,842 NRA cases the plain arm scored 5.257, the masked
  arm 5.261, the arm with signed MW columns 5.274, and the arm with corrected lags 5.264 points of
  capacity.
headline: >-
  C1, masked minus plain: +0.004 points (98.33% interval -0.026 to +0.025). C2, signed MW columns
  minus masked: +0.013 (-0.013 to +0.052). C3, corrected lags minus masked: +0.004 (-0.019 to
  +0.019).
lessons: >-
  Outside switching, a forecast trained without the labelled switching half-hours scores the same
  as one trained on everything, to within 0.03 points, and neither magnitude input helped. Only
  1.3% of scored half-hours have a lag inside an event, so the data cannot show a gain even if one
  exists.
advice:
  - >-
    Do not spend effort on a switching detector to improve the Primary forecast's score outside
    switching: an oracle mask and oracle MW columns left it within 0.06 points (C1 to C3, figure 1).
  - >-
    Test lag-corrected inputs together with the MW columns before ruling the corrected-lag idea out,
    because on the 37,394 half-hours with a lag inside an event the corrected lags cut the score by
    0.42 (-1.51 to +0.02) while the MW columns alone raised it by 1.07 (D8, D9, figure 2).
  - >-
    Score a masking experiment on a longer history than one validation year, because only 22
    labelled events fall in the 292 runs and 8 months hold any lag inside an event (figure 2).
  - >-
    Judge a refit arm against a same-code repeat of the plain arm, because per-case scores of the
    same refit differ between graphics-card runs: only 9% of cases matched experiment 48 to 0.001
    (reproduction check).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00075-sw2-11-nra-forecast-with-labelled-switching-masks/
---

# Experiment 75: can a Primary demand forecast learn the normal running arrangement from labelled switching?

**Take-home message: outside switching, the Primary substation demand forecast was already close to
the normal running arrangement, so neither removing the labelled switching from training nor
telling the model how much load switching moved changed the quantile score by more than 0.06
points of capacity.** The experiment cannot show that removing the switching is harmless within its
0.02-point margin. It also cannot show a gain from either extra input. The plain weekly-refitted
XGBoost demand model stays the v1 candidate. Subtracting the moved MW from each lag did help on the
few half-hours whose lag sits inside a labelled event, so SW2-12 (the same test with a detector's
labels) should try it first. The verdict is **inconclusive**.

An electricity network operator sometimes closes a normally open switch, so that one Primary
substation (a substation that steps down to the local distribution network) carries load that
belongs to a neighbour. The metered power then differs from the power in the normal running
arrangement (NRA). We asked whether an XGBoost demand model forecasts the NRA as well when its
training targets leave out the half-hours of labelled switching events, and whether the signed MW
that switching moved at each lagged power input, or each lag minus that MW, lowers the error.

We refitted the model every week for 16 Primary substations on four arms. We scored the arms on the
292 daily forecast runs of [experiment
48](../00048-weekly-retraining-confirmation-and-extension-demand-generators/result.md), on
half-hours outside labelled switching and logged closures. The score is the quantile score, in
points of capacity, where smaller is better. The 82 labelled events come from [experiment
41](../00041-sw2-0a-labeller-gate/result.md). Their onsets, ends, and moved MW are hindsight values,
so the result is an upper bound on what a detector could achieve.

![Planned contrasts](figure_1.svg)

![Contrasts where a lag sits inside an event](figure_2.svg)

![Absolute score of every arm](figure_3.svg)

- **Arms (all weekly, seed 0, N1 recipe of experiment 48).** `plain`: trained on all rows. `nra`:
  training rows inside a labelled event are dropped. `nra_mw`: `nra` plus the signed MW at each of
  the three lags. `nra_lag`: `nra` with each lag replaced by the lag minus that MW.
- **Rows.** 3,199,243 cases, of which 2,928,842 are NRA rows. Every arm holds the same cases.
- **Score on the NRA rows (points of capacity).** `plain` 5.257, `nra` 5.261, `nra_mw` 5.274,
  `nra_lag` 5.264. Benchmarks from experiment 48: weekly with the 14-day correction 5.334, the
  stack (frozen model, 14-day correction) 5.448, `manual_heuristic` 6.121, `climatology` 7.479.
  The two baselines hold slightly fewer NRA cases (2,927,210 and 2,783,317).
- **C1, `nra` minus `plain`: +0.004 (98.33% interval -0.026 to +0.025).** The upper end exceeds the
  +0.02 margin, which is the most that [experiment
  44](../00044-sw2-2-switching-share-of-best-guess-score/result.md) says a perfect detector could
  gain. The training-target mask is therefore not shown to be harmless. It is not shown to be
  harmful either, and the plain arm is the v1 NRA candidate.
- **C2, `nra_mw` minus `nra`: +0.013 (-0.013 to +0.052). C3, `nra_lag` minus `nra`: +0.004 (-0.019
  to +0.019).** Neither interval lies below zero, so neither arm is supported.
- **Half-hours with a lag inside an event: 37,394 (1.3%), in 8 months.** There the mask lowered the
  score by 0.50 (95% interval -2.08 to +0.35), the corrected lags lowered it by a further 0.42
  (-1.51 to +0.02), and the MW columns raised it by 1.07 (-0.19 to +3.31). These intervals are not
  corrected for several comparisons.
- **Reproduction check.** The `plain` arm's mean score on all cases, 5.4403, matches experiment
  48's stored weekly score, 5.4402. Only 9% of per-case scores match to 0.001. The largest gap is
  4.39 points, because repeated graphics-card fits are not bit-for-bit identical. The 0.004 in C1
  is therefore at the level of fit-to-fit noise.
- **Caveats.** Only 22 of the 82 labelled events fall in the scored year. Events before January
  2022 are not masked in training. The result rests on one validation year and one seed.
- **Next.** SW2-12 repeats the arms with the detector's labels. It adds first the corrected lags
  together with the MW columns, which this experiment did not run.
