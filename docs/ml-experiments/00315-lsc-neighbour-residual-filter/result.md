---
id: "00315"
slug: lsc-neighbour-residual-filter
title: >-
  Do neighbouring substations' residuals cut the false alarms of an LSC detector?
theme: lsc-detectors
also_relevant_to: []
status: planned
kind: other
size: >-
  medium. Stored: private score tables under data/ml-experiments/00315-lsc-neighbour-residual-filter/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one defensible filter
  rule, fixed below before any alarm is scored. Callers: this folder's scripts only; they import
  experiment 309's folder and read the stored scores of experiments 308 and 309. Asset graph: not
  touched. The maintainer's brief limits the process to Sonnet: one plan review, one code review.
hypothesis: >-
  A real load shift moves load between neighbouring Primary substations, so the 24-hour mean
  residuals at the two substations have opposite signs, while weather and forecast error give the
  same sign at neighbours. Keeping a 24-hour mean residual alarm only when a neighbour does not
  contradict it therefore cuts the alarms per matched shift at equal recall.
prediction: >-
  Written before any alarm is scored. On the shifts of the long-and-deep stratum (19 of the 35 real
  load shifts over 1 day; at least 2 days long and moving at least 10% of load) whose logged
  counterpart Primary is metered (15 of them), the alarms per matched shift of the filtered 24-hour
  residual are at least 25% lower than the plain 24-hour residual's, at equal recall. The prediction
  fails if same-sign neighbour residuals are common on real shifts, or if the filter rarely fires.
decision_rule: >-
  Supported when the paired 95% interval (resampling whole shifts and whole calendar months, 20,000
  draws) of the ratio of filtered to plain alarms per match, in the stratum-with-counterpart group at
  the plain arm's recall in that group, lies wholly below 0.75. Not supported otherwise. The filter
  changes no model and no stored data, so any gain whose interval excludes 1 is cheap to port; the
  25% bar is the maintainer's. The experiment is exempt from the research-default `change` rule.
benchmark: the plain 24-hour mean residual detector of experiment 308 at the same recall
row_set_id: real record of experiment 308 (35 labelled real load shifts over 1 day, 15 Primaries)
metric: alarms per matched shift; matched shifts; megawatt-weighted share of matched shifts
---

# Do neighbouring substations' residuals cut the false alarms of an LSC detector?

**Status: planned. Nothing below has been scored.**

## Step 0: the ceiling (counted before any alarm)

- The neighbour list is the interconnection list of experiment 37 (the `ADJACENCY` table), limited
  to the 15 Primaries with a stored residual series. It is written to `neighbours.json` before any
  alarm is scored. It uses no shift label.
- All 35 real shifts have at least one interconnected metered neighbour. 22 of 35 have both
  Primaries of their logged pair metered (15 of the 19 in the long-and-deep stratum, 7 of the
  other 16). The other 13 have a pair member with no stored residual series (series 12 or a
  single-sided pair). The stop rule (fewer than about 10) does not fire.

## Plan

- **Signals.** The 24-hour mean residual (`r24`, experiment 308) and the band-position signal
  (`pit24`, experiment 309), each read from the stored scores. A flagged origin has an absolute
  score above `k` robust scales; flagged origins within 48 hours form one alarm (experiment 309's
  episode rule).
- **Filter rule, fixed.** At each flagged origin, compare the alarm series' score `a` with each
  metered interconnected neighbour's score `b` at the same origin (megawatts for `r24`; normal-score
  units for `pit24`). A neighbour confirms when `a*b < 0` and `|b| >= 0.25|a|`; it contradicts when
  `a*b > 0` and `|b| >= 0.25|a|`. Keep the flagged origin if some neighbour confirms, or if no
  neighbour has a finite score. Otherwise drop it when two or more neighbours contradict it; keep
  it when fewer than two do. No shift label enters the rule.
- **Arms.** Plain `r24`; filtered `r24`; `pit24` plain and filtered (separate arm, identical rule);
  the learned score `LR_b20` and RO-NC (`RO_b80`) of experiment 308 as stored.
- **Equal recall.** The threshold `k` of the plain arms is experiment 308/309's (set on odd months to
  the false alarm rate of `LR_b20` and of RO-NC). The filtered arm takes the largest `k` on the
  grid 1.0 to 20.0 (step 0.1) whose matched shifts are at least the plain arm's. The curve of alarms
  per match against matched shifts is also drawn over the whole grid. Alarms per match at recall of
  at least 0.5 (18 of 35) and at 0.33 (12 of 35) are read off each curve.
- **Alarms per match** is (false alarms in ordinary rows over all months + matched shifts) divided by
  matched shifts in the group, as in experiment 308. Alarms inside logged closures or on the probable
  unlogged events (D8's multi-day drops and the 17 block days) are not false alarms and are counted
  separately.
- **Groups.** (a) long-and-deep stratum and (b) the rest; (c) shifts with a metered counterpart and
  without; (d) D8 and D9 false alarms separately. Megawatt-weighted matches use experiment 308's
  weights.
- **Sensitivity (not the result).** The size threshold at 0.10 and 0.50; the rule that drops on one
  contradicting neighbour.
- **No injected worlds.**
- **Framing.** Experiment 312 showed that the forecast does not need a precise detector, so the
  filter's value is for event labels for NGED (aim 3), not for forecast cost.

## What each outcome changes

- Passes: the filtered residual is the cheap free detector for event labels.
- Fails: neighbour information is exhausted for the 24-hour residual, and the line stops.
