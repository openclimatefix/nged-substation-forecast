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
status: planned
date: 2026-10-24
kind: other
size: >-
  medium. (1) Stored: private forecasts, alarm tables, and score tables under
  data/ml-experiments/00327-lsc-label-free-detect-mask-train-pipeline/ only. (2) Serving path: not
  touched. (3) Degradation rule: not touched. (4) Designs: more than one is defensible for the
  unusualness signal, the label-free threshold, and the mask padding, so each is fixed below
  before any score is read. (5) Callers: this folder's scripts only; they import the folders of
  experiments 130, 212, 258, 297, 309, and 315 and call the reviewed `studies.*` modules unchanged.
  The maintainer's brief limits the work to CPU, 4 cores, one Sonnet plan review, and one Sonnet
  code review.
hypothesis: >-
  A forecast of the normal running arrangement (NRA) can be trained and corrected on power with
  the load shifts set to missing, where a label-free detector finds the load shifts. The detector
  reads only how unusual the observed power is inside the predictive bands of a demand model that
  uses no lagged power, and no NGED switching label enters any threshold, mask, or training row.
prediction: >-
  Written before any score is read. P1: arm A (the label-free pipeline) beats arm C (no mask) on
  the 95% quantile pinball loss against the true NRA series, in the DURING and AFTER row groups of
  the injected worlds, by at least 0.05 points of effective capacity with a 95% interval above
  zero, and changes the ordinary rows (G1) of the real record by less than 0.02 points. P1 fails if
  arm C wins or ties in DURING and AFTER. P2: arm A finds at least as many of the 19
  long-and-deep real shifts as arm B at the same alarm rate per Primary-year. P3: the detector's
  demand model, refitted on power that has its pass-1 alarms masked, finds at least 20% more of
  the 35 real shifts than the same model fitted on unmasked power (pass 2 against pass 1), which
  would show that fitting on shifts hides them.
decision_rule: >-
  Fixed before any score is read. Rows and weather: past weather (ERA5 and CAMS) for the
  detector and training stages throughout. Real record: experiment 258's harness (15 Primaries,
  6-hourly origins, 336 half-hour leads), scored on even calendar months only, on target rows
  outside every logged or labelled load shift span (G1 ordinary; G4 AFTER, a target within 48 hours
  after a span's end; G5, a target within 7 days after it), and keeping origins inside spans.
  Injected worlds: experiment 297's 44 worlds at six Primaries, scored against the true NRA series
  on all rows, with G0 all rows, G1 ordinary, G2 BEFORE_ALARM, G3 DURING, G4 AFTER, G5 within 7
  days; alarm thresholds fixed from the real record's odd months and seeds, then scored on even.
  Stages. Detector GAM (arm C's model): experiment 130's convex demand model with no lagged power
  and no drift, 24-month window, ridge 0.1, refitted on the 1st of each month, trained on
  every half-hour with a reading (no hand mask, no logged span removed), its forecasts out of
  sample. Unusualness: each half-hour's normal score of the residual inside that series' and hour
  band's residual quantile function (experiment 309's band-position signal), where the quantile
  function is fitted on all odd-month half-hours with no label filter. Signal: the 12 hour rolling
  mean (24 half-hours) of the signed normal score, evaluated at each 6-hourly origin. Alarm: an
  episode opens when the absolute 12 hour mean exceeds the Primary's threshold at a flagged origin
  and merges flagged origins within 48 hours, with experiment 298's causal end rule. Threshold: per
  Primary, the smallest value that makes the odd-month alarm rate 4.7 episodes per Primary-year
  (the median RO-NC rate in experiment 217), sensitivity at 2.5 and 10. Neighbour filter:
  experiment 315's rule unchanged (theta 0.25, drop at two or more same-sign neighbours). Mask: the
  alarm interval from the onset stamp to the known end, with no padding; the end is the cut when
  not yet known, as in experiment 212. Train stage: the same convex model refitted at each monthly
  cut on the rows outside the alarms known at that cut, and the Kalman filter of
  `studies.dp_assimilation` re-run with the alarmed readings treated as missing, its four
  hyperparameters refitted per arm on odd months. Quantile offsets per arm: fitted on that arm's odd-
  month residuals, all half-hours. Arms: A (this pipeline, detector pass 1); B (the 24 hour mean
  residual with the neighbour filter, experiments 308 and 315, at the same 4.7 rate before the
  filter, through the same mask and train stages); C (the unmasked model and the unmasked filter);
  D (reference only: the logged and labelled spans masked, which is experiment 130's stored fits);
  E (arm A with two passes: pass 2 re-detects with the model refitted on the pass-1 masked rows).
  A post hoc arm, A with the mask in the filter only, shows which stage carries any gain. Injected
  worlds: the demand model is the frozen one of experiment 297 for every arm, so arms differ in
  the filter's mask only, and the training stage is measured on the real record only (experiment
  212 found simulated transfers move the training score by 0.009 points). Metrics: the mean quantile
  score over 11 quantiles (a CRPS surrogate), the 95% quantile pinball loss (a threshold-weighted
  CRPS surrogate), the mean absolute error of the median, q01 and q99 coverage and pinball loss.
  Curtailment cost is not scored on the real record (experiment 277) and is scored on the injected
  worlds only if experiment 297's cost code runs unchanged. Contrasts, with the 95% intervals
  resampling whole shifts and whole calendar months, 20,000 draws, read at the Bonferroni level for
  four contrasts (98.75%): C1, A minus C on DURING and AFTER pinball at the 95% quantile, injected
  worlds; C2, A minus C on G1 of the real record; C3, A minus B on matched long-and-deep shifts at
  equal pre-filter alarm rate; C4, E minus A on the share of the 35 real shifts matched. Threshold
  for porting: detecting and masking adds a pipeline stage, so a gain must be at least 0.05 points
  of effective capacity (the brief's bar, set by the cost of that stage), and C2 must lie within
  +/-0.02. Verdict: supported when C1 holds the threshold and C2 stays within bounds;
  partly supported when only one holds; not supported when C1's interval is not above zero. Also
  reported beside, none a bar: each stage alone (alarms, matched shifts, alarms per match, share of
  the 35 real shifts alarmed in pass 1 and pass 2 by size and duration, the 19 long-and-deep
  shifts, D8 and D9 separately), winter, the scores on rows no arm masks, and an ENS-weather
  scoring of the final forecast at the six Primaries whose ENS frames exist (D5, D9, D10, D11, D14,
  D16), with the number of real shifts it covers. The ENS scoring runs after the past-weather
  results are reviewed and cannot change a verdict. Every input and its source is listed in the
  write-up. Labels enter only the scoring groups, the match rule, and arm D.
benchmark: >-
  Arm C, the research default as it stands (unmasked model and unmasked filter), and arm D, the same
  with the logged and labelled spans masked.
row_set_id: >-
  experiment 258's real-record rows on even months outside every labelled span, and experiment
  297's injected-world rows; not comparable with the leaderboard
metric: >-
  95% quantile pinball loss and mean quantile score, in points of effective capacity; matched real
  shifts and alarms per matched shift for the detector alone.
code: ml-experiments/00327-lsc-label-free-detect-mask-train-pipeline/
---

**This page is the plan; the write-up replaces it when the runs finish.** The experiment asks
whether the forecast gains from masking power readings that a detector, using no NGED switching
label, flags as load shifts. The detector reads how unusual the observed power is inside the
predictive bands of a demand model with no lagged power.
