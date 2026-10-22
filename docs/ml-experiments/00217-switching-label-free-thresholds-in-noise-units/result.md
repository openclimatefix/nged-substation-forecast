---
id: "00217"
slug: switching-label-free-thresholds-in-noise-units
title: >-
  Does one threshold in noise units give predictable alarm rates across Primary substations, with no
  switching labels?
theme: switching-detectors
also_relevant_to: []
status: planned
size: >-
  simple. Stored: private tables under data/ml-experiments/00217-switching-label-free-thresholds-in-noise-units/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one (descriptive, plus one
  pre-registered prediction; the candidate noise estimates are listed below). Callers: this folder's
  scripts only, which import the detector modules of experiments 130 and 156 and read stored alarm
  tables of experiment 195.
hypothesis: >-
  At a fixed threshold in noise units, the switching detectors raise alarms at a rate that is similar
  across Primary substations, so a new Primary needs no tuning. Substations D8 and D9 are the
  exceptions, because their residuals have heavy tails.
prediction: >-
  At least 80% of the Primaries other than D8 and D9 (11 of 13) have an alarm rate within a factor
  of 2 of the cross-Primary median, for each of RO-NC and the pair alarm. A robust (median absolute
  deviation, MAD) noise estimate brings D8 and D9 inside a factor of 3 of the median. It fails if
  heavy tails need a per-series treatment; the page then says that version 2 needs an automatic
  per-series noise rule, and names the residual statistic that predicted the alarm rate best.
decision_rule: >-
  Descriptive, except the prediction. Alarm rate is alarms per Primary-year over the whole residual
  history, with 95% intervals that resample whole calendar months. The prediction holds when both
  its parts hold for RO-NC (the pair alarm is reported alongside). Verdict `supported` when both
  parts hold, `partly supported` when exactly one holds, `refuted` when neither holds. No threshold
  of effect size applies because nothing is ported. Statistic ranking: leave-one-Primary-out
  prediction error of log alarm rate from each single residual statistic.
benchmark: >-
  The cross-Primary median alarm rate of each detector at its stored threshold.
row_set_id: none (alarm counts, not forecasts)
metric: Alarms per Primary-year, in winter (December to February) and in the rest of the year.
---

# Does one threshold in noise units give predictable alarm rates across Primaries, with no labels?

**Plan only; no result yet.** Version 2 of the service will have no NGED switching labels, so a
detector's threshold has to come from the residual alone.

## Design

- **Detectors.** RO-NC on the arm B residual of [experiment
  130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md) (variant X, 24-hour window, 4.35
  noise units); the pair alarm of [experiment
  156](../00156-switching-faster-onset-alarm-from-opposite-steps/result.md) (4-hour window, 3.75
  noise units); the learned score of [experiment
  195](../00195-switching-neural-plus-detector-features/result.md) (stored alarms of the re-ranker,
  at its operating point). The learned score is trained on labels, so its rates are descriptive only.
- **Alarms.** All causal peak alarms over the whole residual history of every Primary with at least
  12 months of residual. No label, logged closure, or labelled transfer is read.
- **Residual statistics per Primary.** The current noise estimate (MAD of the step statistic in odd
  months), the MAD of the step statistic over all months at 2, 4, 6, and 24 hours, the kurtosis of
  the step statistic, the share of days where the model is off by more than 3 noise units for 24
  hours or more, and the same share on holidays and Easter.
- **Candidate noise estimates (fixed before any rate is seen).** R1: MAD over all months at the
  detector's window. R2: the 95th percentile of the absolute step statistic divided by 1.96, so
  heavy tails raise the scale. Thresholds are recomputed with each and the spread of alarm rates is
  compared.
- **Predictor.** Log alarm rate regressed on one statistic at a time, leave-one-Primary-out.

## Out of scope

The 35 logged transfers are not read.
