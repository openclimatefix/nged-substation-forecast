---
id: '00327'
slug: lsc-label-free-detect-mask-train-pipeline
title: >-
  Does masking load shifts found by a label-free detector improve a forecast of the normal running
  arrangement?
theme: lsc-detectors
also_relevant_to:
  - curtailment-cost
  - calibration-and-uncertainty
status: finished
date: 2026-10-24
kind: other
size: >-
  medium. (1) Stored: private forecasts, alarm tables, and score tables under
  data/ml-experiments/00327-lsc-label-free-detect-mask-train-pipeline/ only. (2) Serving path: not
  touched. (3) Degradation rule: not touched. (4) Designs: more than one is defensible for the
  unusualness signal, the label-free threshold, and the mask padding, so each is fixed below
  before any score is read. (5) Callers: this folder's scripts only; they import the folders of
  experiments 130, 212, 258, 297, 309, and 315 and call the reviewed `studies.*` modules
  unchanged. The maintainer's brief limits the work to CPU, 4 cores, one Sonnet plan review, and
  one Sonnet code review.
hypothesis: >-
  A forecast of the normal running arrangement (NRA) can be trained and corrected on power with
  the load shifts set to missing, where a label-free detector finds the load shifts. The detector
  reads only how unusual the observed power is inside the predictive bands of a demand model that
  uses no lagged power, and no NGED switching label enters any threshold, mask, or training row.
prediction: >-
  Written before any score is read. P1: arm A (the label-free pipeline) beats arm C (no mask) on
  the 95% quantile pinball loss against the true NRA series, in the DURING and AFTER row groups of
  the injected worlds, by at least 0.05 points of effective capacity with a 95% interval above
  zero, and changes the ordinary rows (G1) of the real record by less than 0.02 points. P1 fails
  if arm C wins or ties in DURING and AFTER. P2: arm A finds at least as many of the 19
  long-and-deep real shifts as arm B at the same alarm rate per Primary-year. P3: the detector's
  demand model, refitted on power that has its pass-1 alarms masked, finds at least 20% more of
  the 35 real shifts than the same model fitted on unmasked power (pass 2 against pass 1), which
  would show that fitting on shifts hides them.
decision_rule: >-
  Fixed before any score is read. Rows and weather: past weather (ERA5 and CAMS) for the detector
  and training stages throughout. Real record: experiment 258's harness (15 Primaries, 6-hourly
  origins, 336 half-hour leads), scored on even calendar months only, on target rows outside every
  logged or labelled load shift span (G1 ordinary; G4 AFTER, a target within 48 hours after a
  span's end; G5, a target within 7 days after it), and keeping origins inside spans. Injected
  worlds: experiment 297's 44 worlds at six Primaries, scored against the true NRA series on all
  rows, with G0 all rows, G1 ordinary, G2 BEFORE_ALARM, G3 DURING, G4 AFTER, G5 within 7 days;
  alarm thresholds fixed from the real record's odd months and seeds, then scored on even. Stages.
  Detector GAM (arm C's model): experiment 130's convex demand model with no lagged power and no
  drift, 24-month window, ridge 0.1, refitted on the 1st of each month, trained on every half-hour
  with a reading (no hand mask, no logged span removed), its forecasts out of sample. Every demand
  model of every arm, arm D included, is refitted in this experiment; no stored fit of experiment
  130, 212, 258, or 297 is read for a forecast. Unusualness: each half-hour's normal score of the
  residual inside that series' and hour band's residual quantile function (experiment 309's
  band-position signal), where the quantile function is fitted on all odd-month half-hours with no
  label filter. Experiment 309's band code filters on logged closures, so it is rewritten here,
  not reused, and a test asserts that no label loader is imported on the path of arms A, B, and E.
  The quantile function therefore includes the shifts (the circularity trap), which the write-up
  reports as such. Signal: the 12 hour rolling mean (24 half-hours) of the signed normal score,
  evaluated at each 6-hourly origin. Alarm: an episode opens when the absolute 12 hour mean
  exceeds the Primary's threshold at a flagged origin and merges flagged origins within 48 hours,
  with experiment 298's causal end rule. Threshold: per Primary, the smallest value that makes the
  odd-month alarm rate 4.7 episodes per Primary-year (5, a round number near the median RO-NC rate
  in experiment 217, chosen here with no label look; sensitivity at 2.5 and 10). Neighbour filter:
  experiment 315's rule unchanged (theta 0.25, drop at two or more same-sign neighbours). Mask:
  the alarm interval from the onset stamp to the known end, with no padding; the end is the cut
  when not yet known, as in experiment 212. Train stage: the same convex model refitted at each
  monthly cut on the rows outside the alarms known at that cut, and the Kalman filter of
  `studies.dp_assimilation` re-run with the alarmed readings treated as missing, its four
  hyperparameters refitted per arm on odd months. Quantile offsets per arm: fitted on that arm's
  odd- month residuals, all half-hours. Arms: A (this pipeline, detector pass 1); B (the 24 hour
  mean residual with the neighbour filter, experiments 308 and 315, at the same 4.7 rate before
  the filter, through the same mask and train stages); C (the unmasked model and the unmasked
  filter); D (reference only: the logged and labelled spans masked, which is experiment 130's
  stored fits); E (arm A with two passes: pass 2 re-detects with the model refitted on the pass-1
  masked rows). A post hoc arm, A with the mask in the filter only, shows which stage carries any
  gain. Injected worlds: the demand model is the frozen one of experiment 297 for every arm, so
  arms differ in the filter's mask only, and the training stage is measured on the real record
  only (experiment 212 found simulated transfers move the training score by 0.009 points). Rows
  scored on the real record are the ordinary rows that no arm (A, B, E, D) masks, so a masked
  half-hour cannot flatter an arm; the number of extra rows excluded is printed. Metrics: the mean
  quantile score over 11 quantiles (a CRPS surrogate), the 95% quantile pinball loss (a
  threshold-weighted CRPS surrogate), the mean absolute error of the median, q01 and q99 coverage
  and pinball loss. Curtailment cost is not scored (experiment 277 for the real record; the
  node-plus-generators metric needs a node and generators this harness does not carry); q95 and
  q99 pinball stand in. Contrasts, with the 95% intervals resampling whole shifts and whole
  calendar months, 20,000 draws, read at the Bonferroni level for four contrasts (98.75%): C1, A
  minus C on DURING and AFTER pinball at the 95% quantile, injected worlds; C2, A minus C on G1 of
  the real record; C3, A minus B on matched long-and-deep shifts at equal pre-filter alarm rate;
  C4, E minus A on the share of the 35 real shifts matched. Only C1 and C2 decide the verdict; C3
  and C4 are descriptive and carry P2 and P3 (P3's 20% is a stated expectation, not a bar).
  Threshold for porting: detecting and masking adds a pipeline stage, so a gain must be at least
  0.05 points of effective capacity (the brief's bar, set by the cost of that stage) on the DURING
  and AFTER rows pooled, and C2 must have a point estimate within +/-0.02 and a 98.75% interval
  inside +/-0.05. The write-up prints experiment 212's pinball scale beside the bar. Verdict:
  supported when C1 meets the bar and C2 holds; partly supported when exactly one of the two
  holds; not supported when C1's interval is not above zero. Also reported beside, none a bar:
  each stage alone (alarms, matched shifts, alarms per match, share of the 35 real shifts alarmed
  in pass 1 and pass 2 by size and duration, the 19 long-and-deep shifts, D8 and D9 separately),
  winter, the scores on rows no arm masks, and an ENS-weather scoring of the final forecast at the
  six Primaries whose ENS frames exist (D5, D9, D10, D11, D14, D16), with the number of real
  shifts it covers. The ENS scoring runs after the past-weather results are reviewed and cannot
  change a verdict. Every input and its source is listed in the write-up. Labels enter only the
  scoring groups, the match rule, and arm D.
benchmark: >-
  Arm C, the research default as it stands (unmasked model and unmasked filter), and arm D, the
  same with the logged and labelled spans masked.
row_set_id: >-
  experiment 258's real-record rows on even months outside every labelled span, and experiment
  297's injected-world rows; not comparable with the leaderboard
metric: >-
  95% quantile pinball loss and mean quantile score, in points of effective capacity; matched real
  shifts and alarms per matched shift for the detector alone.
code: ml-experiments/00327-lsc-label-free-detect-mask-train-pipeline/
verdict: partly supported
best_arm: >-
  A, the label-free pipeline: during and after an injected shift its 95% pinball loss is 0.75
  points of capacity lower than with no mask (98.75% interval -1.53 to -0.26)
headline: >-
  Masking the load shifts that a label-free detector flags lowers the error during and after
  injected shifts by 0.75 points of capacity (1.79 to 1.04) and does not harm ordinary days
  (-0.016, 98.75% interval -0.060 to +0.010). Without site D11 the gain is 0.30 points.
lessons: >-
  The gain comes from masking the readings in the Kalman filter, not from retraining the demand
  model on masked power. The detector found 25 of 35 real shifts, about as many as the plain 24
  hour residual detector (26), and a second pass found none more.
take_home_message: >-
  A forecast of the normal running arrangement recovers faster after a load shift when the
  readings that a label-free detector flags are treated as missing: during and after injected
  shifts the 95% pinball loss falls from 1.79 to 1.04 points of capacity. On ordinary days the
  masked forecast is no worse, and slightly better. Without one site (D11) the gain shrinks to 0.30 points,
  and retraining the demand model on masked power adds nothing beyond masking the filter.
advice:
  - >-
    Mask the flagged readings in the Kalman filter first, and treat retraining the demand model on
    masked power as optional, because the filter-only arm matched the full pipeline (AF, C2).
  - >-
    Give every alarm a closing rule that holds when no end is known, such as a 14-day cap that
    ends the mask, because an alarm with no close would mask the series to its end (data-processing page).
  - >-
    Do not expect a second detection pass on a demand model refitted without the alarmed readings
    to find more shifts; it found 24 of 35 against 25 for one pass (C4).
  - >-
    Test the unusualness signal against the plain 24 hour residual at more than one site before
    preferring it, because the two detectors found 25 and 26 of the 35 real shifts (C3).
  - >-
    Report every injected-world gain with and without the site that holds most of it, because
    dropping D11 shrank the gain from 0.75 to 0.30 points (leave-one-out table).
production_status: candidate
adopted_in: none
---

**Masking the readings that a label-free detector flags as load shifts makes a forecast of the
normal running arrangement (NRA) recover faster after a shift, and loses no accuracy on ordinary
days.** The masked forecasts were trained and corrected with no label from the network operator
(NGED). On the injected worlds, the rows during and after a shift score 1.04 points of capacity on
the 95% pinball loss with the label-free pipeline (arm A), against 1.79 with no mask and 0.81 with
the true spans masked. On the real record's ordinary rows, arm A scores 0.016 points better than no
mask, and its 98.75% interval runs to -0.060 on the better side. The verdict is *partly supported*,
because the pre-registered bar for the ordinary rows asked for that interval to lie inside plus or
minus 0.05.

![Difference from no mask](figure_1_difference_from_no_mask.svg)

## What the experiment did

**The detector reads only how unusual the observed power is.** A lagless demand model (the convex
model of [experiment 130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md), ridge 0.1,
24-month window, refitted monthly, ERA5 reanalysis and CAMS past weather) gives quantiles for each
half-hour. The observed power is converted to a normal score in those quantiles, per Primary and
hour band (the band position of experiment 309). The 12 hour mean of the score is read at each
6-hourly origin. An alarm opens when the absolute mean passes a per-Primary threshold, set so that
odd months alarm 5 times per Primary-year with no label look. An alarm closes when the signed mean
falls below half the threshold at two consecutive origins, or after 14 days. The neighbour filter of
experiment 315 then drops alarms that two or more neighbours echo.

**The alarmed half-hours are removed from both the demand model's training rows and the Kalman
filter's inputs.** The demand model is refitted without the alarmed half-hours at each monthly cut,
and the two-state Kalman filter of `studies.dp_assimilation` treats the alarmed half-hours as
missing and re-runs from the onset. Arms:

- **A:** the pipeline above.
- **B:** the 24 hour mean residual with the same neighbour filter, through the same mask and train
  stages.
- **C:** no mask.
- **D:** the logged and labelled spans masked (a reference that uses labels).
- **E:** arm A with a second detection pass on the demand model refitted without the pass-1 alarms.
- **AF:** arm C's demand model with arm A's mask in the filter only (post hoc, to see which stage
  carries the gain).

On the injected worlds (experiment 297's 44 worlds at six Primaries) every arm uses the frozen
demand model, so the arms differ in the filter's mask only. The real record is experiment 258's
harness: 15 Primaries, 6-hourly origins, 336 half-hour leads, scored on even months.

## Results

**The headline contrasts (C1 and C2) decide the verdict.** Points of capacity; negative is better
than no mask.

| Contrast | Arm A minus no mask | 95% interval | 98.75% interval | Bar |
|---|---|---|---|---|
| C1: injected worlds, during and after a shift, 95% pinball loss | -0.752 | -1.340 to -0.322 | -1.533 to -0.257 | at least 0.05 better, interval below zero: **met** |
| C2: real record, ordinary rows, 95% pinball loss | -0.016 | -0.048 to +0.006 | -0.060 to +0.010 | point within 0.02, interval inside 0.05: **not met** (lower end) |
| Real record, ordinary rows, mean quantile score | -0.036 | -0.073 to -0.007 | not a bar | descriptive |
| Real record, 48 hours after a shift, 95% pinball loss | -0.953 | -2.393 to -0.179 | -2.953 to -0.118 | descriptive |

**Arm A beats arm B on the injected worlds.** During and after a shift, arm A minus arm B is -0.276
points (95% interval -0.486 to -0.104). On the real record's 48 hours after a shift the two arms are
close (A -0.953, B -1.095, no-mask score 1.670).

**With no mask, the 95% pinball loss is 1.79 during and after injected shifts and 0.71 on real
ordinary rows.**

| Rows | No mask (K0 or C) | Arm A | Arm B | Logged or true spans masked |
|---|---|---|---|---|
| Injected, during and after: 95% pinball loss | 1.789 | 1.037 | 1.313 | 0.811 |
| Injected, during and after: mean quantile score | 4.462 | 3.179 | 3.644 | 2.557 |
| Real, ordinary rows: 95% pinball loss | 0.712 | 0.696 | 0.697 | 0.707 |
| Real, ordinary rows: mean quantile score | 2.761 | 2.725 | 2.738 | 2.744 |
| Real, 48 hours after: 95% pinball loss | 1.670 | 0.717 | 0.575 | 0.556 |

![Ordinary-row score per Primary](figure_2_ordinary_score_per_primary.svg)

**The tails stay calibrated on ordinary days, and the upper tail improves after a shift.** On the
real record's ordinary rows the share of readings below the 0.01 quantile is 0.0081 for arm A and
0.0084 for no mask. The share above the 0.99 quantile is 0.0068 for arm A and 0.0077 for no mask. On
the injected worlds, during and after a shift, the share above the 0.99 quantile is 0.211 with no
mask, 0.101 with arm A, 0.154 with arm B, and 0.026 with the true spans masked. Arm A still
under-covers the upper tail there, by a factor of ten.

**The gain comes from the filter mask.** Arm AF (the filter mask alone) scores -0.046 on ordinary
rows and -0.954 at the 48 hours after a shift, against -0.036 and -0.953 for arm A. Retraining the
demand model on masked power adds nothing measurable. At the short-history Primary 7, whose first 24
months are the fitting window, the monthly error of arm A and the no-mask arm is about the same
after month 30.

![Monthly error at the short-history Primary](figure_3_short_history_primary.svg)

## The detector alone

**Arm A finds about as many of the 35 real shifts as arm B, and a second pass finds none more.** Of
the 35 labelled shifts, arm A matches 25 (95% interval 20 to 30) with 221 alarms and 85 false
alarms, which is 8.8 alarms per matched shift. Arm B matches 26 (21 to 31) with 231 alarms and 85
false alarms. Arm E (pass 2) matches 24 (18 to 29). The median lag from onset to alarm is 17 hours
for arm A and 24 hours for arm B. Of the 19 long-and-deep shifts, A and B each match 15 and E
matches 14. The neighbour filter cuts arm A's alarms from 313 to 221 and its false alarms from 148
to 85, at the loss of one matched shift.

![The detectors alone](figure_4_detectors_alone.svg)

**P1 holds as written, P2 is met, and P3 is refuted.**

- **P1 (the headline gain) holds as written.** Arm A beats no mask by 0.75 on the injected rows, and
  the ordinary rows change by -0.016, under the 0.02 limit. The verdict's stricter interval bar on
  the ordinary rows (inside plus or minus 0.05) is not met.
- **P2 (arm A matches at least as many long-and-deep shifts as arm B) is met.** Arms A and B each
  match 15 of the 19. Over all 35 shifts arm A matches 25 against arm B's 26 (A minus B -1, interval
  -5 to +3).
- **P3 (pass 2 finds 20% more shifts than pass 1) is refuted.** Pass 2 finds 24 against 25 (E minus
  A -1, interval -3 to 0).

## Checks

- **Median minus mean forecast.** During and after an injected shift the mean absolute error of the
  median is 5.43 for arm A and 5.42 for the mean forecast, and 7.29 against 7.25 with no mask. The
  mean forecast gives the same contrast as the median (A minus no mask -1.84 against -1.85).
- **Per site.** The mean-forecast error falls under arm A at 5 of the 6 sites (D10 -6.2, D14 -4.4,
  D9 -2.1, D11 -2.1, D16 -0.7 points) and rises at D5 (+0.9).
- **Leave one site out.** Without D11, arm A's gain on the injected rows falls from 0.75 to 0.30
  points, and the true-spans reference shows no gain (+0.05). The gain on the real record's ordinary
  rows survives dropping any one of the 15 Primaries (-0.030 to -0.041 in the quantile score).
- **Audit triggers.** Two audit triggers fired: a nested arm (AF) beat the full pipeline on ordinary
  rows (-0.046 against -0.036), and leaving out D11 shrank the injected gain. Both are reported
  above and neither changes the verdict. A Sonnet audit ran in place of the Opus audit, because the
  brief allows no Opus.
- **Fits.** No masked fit was refused for lack of training rows (0 for arms A, B, and E).

## Inputs and what uses labels

**Labels enter only the scoring groups, the match rule, and arm D.** Four constants carry label
knowledge from earlier work into every arm, including arm C:

- the filter's `step_penalty` and the prior for its four hyperparameters (experiment 106's
  `tuned.json`);
- the frozen demand model of experiment 297 in the injected worlds;
- the neighbour list, a literal dictionary copied from experiment 227's module, whose source is not
  documented;
- the neighbour filter's theta of 0.25 (set by experiment 315 against labels).

The quantile function, the alarm thresholds, the offset tables, and the filter hyperparameters are
fitted on all odd-month half-hours, which include the shifts and lie partly after some even-month
cuts. Fitting on all odd-month half-hours is non-causal, and every arm shares it. The detector's
demand model also sees the shifts as targets, so a shift it has learned as normal will not alarm.

## Deviations and what was not run

- The 95% pinball loss replaces the threshold-weighted continuous ranked probability score, and the
  11-quantile score replaces the continuous ranked probability score.
- The mixture-density arm (a neural network that outputs a density in place of quantiles), the
  scoring with ECMWF ensemble (ENS) weather at six Primaries, and the curtailment cost were not run.
  The ENS stage could not change a verdict, and the harness carries no node and generators for the
  curtailment metric.
- The end rule is adapted from experiment 147's rule R2: an alarm closes when the signed mean falls
  below half the threshold at two consecutive origins, or after 14 days.
- The training stage is not rerun on the injected worlds.

## What the next experiment should answer

**A test of the filter mask alone at more sites, on real shifts rather than injected ones, would
settle the case.** The injected gain rests mostly on D11. A test on the 35 real shifts at the six
Primaries with ENS frames would show whether the 48-hour gain (-0.95) holds with the weather that
production serves.
