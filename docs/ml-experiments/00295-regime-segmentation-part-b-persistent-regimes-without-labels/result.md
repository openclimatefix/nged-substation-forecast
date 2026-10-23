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
  the break within 14 days. At the 14 Primaries with no injection, it finds under 1 false break
  per Primary-year. It labels at least 80% of injected persistent steps as persistent. It is
  expected to fail on two counts. The PV capacity steps of experiment 207 (the PV term of the GAM
  explains these in part) may produce false breaks in summer, and a persistent step shorter than
  about 60 days before the end of the data is expected to be called temporary.
decision_rule: >-
  Verdict `supported` when all three parts of the prediction hold: D8 dated within 14 days, under 1
  false break per Primary-year (point estimate, and the upper end of a 95% bootstrap interval over
  Primaries under 1.5), and a persistent-step classification accuracy of at least 80% on the
  held-out (even-month) onsets with a 95% interval lower bound above 70%. `partly supported` when
  two hold. `not supported` otherwise. The forecast gain from the latest-regime offset is reported
  and does not set the verdict. The offset is a candidate for production at any gain whose
  paired interval excludes zero, since it costs one extra number per series (threshold zero). The
  penalty and the minimum segment length are chosen on the odd calendar months and every reported
  figure uses the even calendar months, as the brief requires.
benchmark: >-
  No forecast benchmark. For the forecast gain: the same monthly-refitted GAM_ONLY without the
  offset, on the same rows. Reference runs: a segmentation of the residual with the penalty set to
  infinity (no breaks) and a random-date control with the same number of breaks.
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
D8 given a known break date. [Experiment 194](../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md)
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
  median absolute deviation of day-to-day changes) and by the Primary's capacity unit for reporting.
- **Segmentation.** Optimal partitioning by dynamic programming with an L2 cost on the two series,
  a penalty per break `beta` times the log of the number of days, and a minimum segment length of
  `m` days. The grid is `beta` in {2, 4, 8, 16, 32} and `m` in {14, 30, 60}.
- **Choosing the penalty.** The pair (`beta`, `m`) is chosen on the odd calendar months: it
  maximises the persistent classification accuracy on injections with onsets in odd months,
  subject to under 1 false break per Primary-year counting only the breaks dated in odd months
  (doubled to annualise) at the 14 Primaries that have no injection. All reported figures use
  onsets and breaks in even months.
- **Labelling a segment boundary.** *Temporary* when the level after the next boundary returns
  within 0.5 noise units of the level before it, within 120 days; or when a neighbouring Primary
  (the five Primaries with the highest residual correlation) shows a boundary within 3 days with
  the opposite sign and at least half the size. *Persistent* otherwise. *Measurement* only for a
  boundary within 3 days of a date on a supplied list; the list is empty, since no meter-change
  dates are on file, so that class is not scored.
- **Tests.** (a) D8 known break. (b) Injections at Primaries with the receiver set: persistent
  steps of 10% and 25% of the Primary's load, flat (the same MW at every half-hour) and with a
  profile (the Primary's own load shape, so the step scales with demand), onset on the first of an
  even or odd month in 2022 to 2024; temporary shifts of 25% for 14, 30, and 60 days, injected
  with the donor's mirror at the Primary with the highest correlation (the receiver gets +x, the
  donor -x). Controls: zero-size injections and the unmodified series. (c) The PV capacity steps of
  experiment 207: a boundary within 14 days of a month where experiment 207's step path at weight 3
  moves capacity by 0.5 MW or more is counted as explained by PV, and the count of such boundaries
  and of unexplained ones is reported.
- **Forecast gain.** For D8 after 2023-08-15 and for every correctly dated injected persistent step,
  a monthly refit of GAM_ONLY (the cut at the first of each month, 24-month window), with and
  without the offset: the mean in-sample residual over the days since the start of the latest
  regime, as of the cut. The break must have been confirmed by the cut, meaning at least `m` days
  have passed. Scored over the next month, for the six months after confirmation.
- **Compute.** CPU only, 4 cores, 16 GB memory cap, `OMP_NUM_THREADS=1` per worker. One fit takes
  about 0.3 seconds, so the whole experiment is a few hours at most.

**Planned contrasts (Bonferroni across the four, level 98.75%).**

- C1: D8 dating error against 14 days.
- C2: false breaks per Primary-year at the 14 controls against 1.
- C3: injected persistent-step accuracy against 0.8.
- C4: forecast error with the latest-regime offset minus without it, on rows after the break.

**Checks to report.** The median forecast minus the mean forecast per site (the GAM_ONLY fit has no
spread, so this is zero by construction and stated as such); the mean-forecast error beside the C4
score; which of the seven audit triggers fired.

**Advice followed or departed from.** The plan follows the advice of experiment 207 to use a total
variation weight near 3 (the penalty `beta` grid includes it) and to count false steps as net
change from one summer to the next as well as single changes. It follows experiment 274's advice to
score refits at the end date of a regime as well as at the first of a month. It departs from none.
