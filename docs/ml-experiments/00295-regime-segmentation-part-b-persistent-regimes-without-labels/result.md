---
id: '00295'
slug: regime-segmentation-part-b-persistent-regimes-without-labels
title: >-
  Can persistent regimes be segmented in a Primary's history without labels?
theme: lsc-detectors
also_relevant_to:
  - retraining-and-training-history
status: planned
size: >-
  medium. Stored: private tables under
  data/ml-experiments/00295-regime-segmentation-part-b-persistent-regimes-without-labels/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one is defensible
  for the residual, the penalty, and the classifier, so each is fixed below before any score is
  read. Callers: this folder's scripts, which call the reviewed `studies.dp_gam` fit and predict
  functions unchanged. Asset graph: not touched.
hypothesis: >-
  A segmentation of a Primary's whole history with hindsight, using only the residual of a
  lagless convex additive model (GAM), dates a persistent change of level to within 14 days and
  tells a persistent step from a temporary one, without labels. A step is persistent when the
  level does not return and no neighbouring Primary shows the opposite step, and temporary
  otherwise.
prediction: >-
  Written before any fit. On the known answer, D8's break on 2023-08-15, the segmentation places
  the break within 14 days. At the 15 Primaries with no injection, it finds under 1 false break
  per Primary-year. On injected steps in held-out Primaries and even-numbered months, it labels at
  least 80% of persistent steps as persistent and at least 80% of temporary shifts as temporary.
  It is expected to fail on two counts. The PV capacity steps of experiment 207 may produce false
  breaks in summer, and a permanent transfer that moves load between two Primaries is expected to
  be called temporary, because the neighbour rule reads any mirrored step as a load shift.
decision_rule: >-
  Verdict `supported` when all three parts of the prediction hold: D8 dated within 14 days, under 1
  false break per Primary-year (point estimate, with the upper end of the 98.75% bootstrap
  interval over Primaries under 1.5), and both the persistent recall and the temporary recall at
  least 80% on the held-out injections, with each 98.75% interval lower bound above 70%. A
  classifier that labels everything persistent therefore fails on temporary recall. `partly
  supported` when two parts hold. `not supported` otherwise. Steps within 150 days of the end of
  the data are not injected, because they cannot show a return. Every interval is 98.75% (the
  Bonferroni level for four contrasts). The forecast gain is reported and does not set the verdict.
  The offset is a candidate for production at any gain whose paired interval excludes zero, since
  it costs one extra number per series (threshold zero), and only if it also beats the trailing
  60-day mean residual that needs no segmentation. Tuning uses odd calendar months and tuning
  Primaries; reporting uses even calendar months and held-out Primaries (below).
benchmark: >-
  No forecast benchmark. For the forecast gain: the same monthly-refitted GAM_ONLY without an
  offset, and with a trailing 60-day mean residual as the offset (no segmentation).
row_set_id: >-
  Own rows (not comparable with the leaderboard). Segmentation: the daily mean residual of every
  Primary in the 16-Primary cache of experiment 106, 2019-09-17 to 2026-06-30. Forecast gain: the
  six monthly forecast periods that follow each regime's confirmation date.
metric: >-
  Break dating error in days; false breaks per Primary-year at the Primaries without injection;
  classification accuracy of injected persistent and temporary steps; mean absolute error change
  from the latest-regime offset in points of effective capacity.
code: ml-experiments/00295-regime-segmentation-part-b-persistent-regimes-without-labels/
---

**Plan only: the write-up replaces this page when the experiment has run.**

**Question.** Each Primary substation sits in one of several regimes. A load shift is a temporary
regime. A new large generator or load, or a change in what the meter includes, is a persistent
regime. This experiment asks whether a persistent regime can be found in a Primary's history from
the residual of a lagless GAM alone.

**Prior work taken forward.** Part A of the brief (another session) tests a regime-aware forecast at
D8 given a known break date. [Experiment
194](../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md)
dated D8's break in the week of 2023-08-14, and is the known answer. [Experiment
207](../00207-dp-meter-regime-steps-e2-step-capacity-known-answer/result.md) found that a
step-capable solar path invents steps unless penalised, so false breaks from PV are the known risk.
[Experiment 274](../00274-lsc-long-arrangement-audit-gam-refit/result.md) found that a refitted
lagless GAM learns a permanent step to half its size in about 12 months, which is why the residual
is taken from a fit that has seen the whole history and why the classifier sees the step as a
change of level rather than as a persistent forecast error. The method is penalised optimal
partitioning of a multivariate series, as in [Killick et al.
(2012)](https://doi.org/10.1080/01621459.2012.737745).

**Design.**

- **Residual.** One GAM_ONLY fit (the settings of experiment 274) per Primary on all rows to
  2026-06-30; the residual is the reading minus the fit. Two daily series per Primary: the mean
  residual over the day, and the mean residual over 00:00 to 06:00 minus the mean over 10:00 to
  16:00 (a change of daily shape). Both are divided by the Primary's robust noise (1.4826 times the
  median absolute deviation of day-to-day changes, over the square root of 2).
- **Segmentation.** Optimal partitioning by dynamic programming with an L2 cost on the two series,
  a penalty per break `beta` times the log of the number of days, and a minimum segment length of
  `m` days. The grid is `beta` in {2, 4, 8, 16, 32, 64} and `m` in {7, 14, 30}.
- **Tuning and reporting split.** The 15 Primaries other than D8 are split by sorted label into
  tuning Primaries (odd positions) and held-out Primaries (even positions). The pair (`beta`, `m`)
  is chosen on injections with onsets in odd calendar months at the tuning Primaries: it
  maximises the mean of persistent and temporary recall, subject to under 1 false break per
  Primary-year at the tuning Primaries' unmodified series. Every reported figure uses injections
  with onsets in even months at the held-out Primaries, and false breaks at the held-out
  Primaries' unmodified series (the unmodified series may hold real regimes, which count as false
  breaks, so the false-break rate is an upper bound).
- **Labelling a segment boundary.** *Temporary* when the level after the next boundary returns
  within 0.5 noise units of the level before it, within 120 days; or when a neighbouring Primary
  (the five Primaries with the highest residual correlation) shows a boundary within 3 days with
  the opposite sign and at least half the size. *Persistent* when neither holds and the boundary
  is at least 150 days before the end of the data. *Measurement* only for a boundary within 3
  days of a date on a supplied list; the list is empty, since no meter-change dates are on file,
  so that class is not scored.
- **Injections.** Onsets are the first of a month from 2021 to 2025, at least 150 days before the
  end of the data and at least 90 days from 2023-08-15. Persistent steps of 10% and 25% of the
  Primary's load, flat (the same MW at every half-hour) and with a profile (a share of the
  Primary's own power, so the step follows demand), at the receiver only. Temporary shifts of 25%
  for 14, 30, and 60 days, with the mirror (-x) at the Primary of highest residual correlation. A
  persistent transfer arm (a permanent +25% at the receiver and a permanent -25% at its mirror)
  has no correct label under the rule and is reported as the share called temporary. The GAM is
  fitted on the injected series, so the month, holiday, and weather terms absorb part of the step.
  A no-absorption arm fits on the clean series and adds the injection to the residual afterwards.
  The share of each step that survives in the residual (the shift in segment level over the
  injected step) is reported.
- **Scoring an injection.** The boundary within 14 days of the onset with the largest level shift
  is the injection's boundary. A missing boundary counts as a wrong label. Recall is the share of
  injections given the correct label; the confusion matrix is printed.
- **PV check.** The experiment 207 step path is not refitted. A false break at a control is called
  PV-like when its shift in the 10:00 to 16:00 residual is at least twice its shift in the
  00:00 to 06:00 residual and the night shift is under 0.5 noise units; counts of PV-like and
  other false breaks are reported, and the verdict counts both.
- **Forecast gain.** At each monthly cut (the first of a month, 24-month window) from the month
  after a step's onset to 6 months after, for D8 after 2023-08-15 and for all injected persistent
  steps at the held-out Primaries, the GAM_ONLY fit gives in-sample residuals, the segmentation
  runs on the days before the cut only, and the offset is the mean residual since the latest
  boundary (none if there is no boundary). The two comparison arms are no offset and the trailing
  60-day mean residual. The score is the mean absolute error over the month after the cut, in
  points of effective capacity, over all injected steps whether or not they were found.
- **Compute.** CPU only, 4 cores, 16 GB memory cap, `OMP_NUM_THREADS=1` per worker. One fit takes
  about 0.3 seconds, so the whole experiment is a few hours at most.

**Planned contrasts (Bonferroni across the four, level 98.75%).**

- C1: D8 dating error against 14 days.
- C2: false breaks per Primary-year at the held-out controls against 1.
- C3: persistent and temporary recall of the injections, each against 0.8.
- C4: forecast error with the latest-regime offset minus without it, and minus the trailing
  60-day offset, on rows after the step.

**Checks to report.** The median forecast minus the mean forecast per site (the GAM_ONLY fit has no
spread, so this is zero by construction and stated as such); the mean-forecast error beside the C4
score; which of the seven audit triggers fired.

**Advice followed or departed from.** The plan follows the advice of experiment 207 to use a total
variation weight near 3 (the penalty `beta` grid includes it) and to count false steps as net
change from one summer to the next as well as single changes. It follows experiment 274's advice to
score refits at the end date of a regime as well as at the first of a month. It departs from none.
