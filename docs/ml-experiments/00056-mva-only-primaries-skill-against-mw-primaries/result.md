---
id: "00056"
slug: mva-only-primaries-skill-against-mw-primaries
status: planned
size: simple
hypothesis: >-
  The best guess (experiment 33's N1 with the 14-day correction and the spread factor) forecasts the
  8 trial Primary substations metered in MVA clearly worse, relative to `manual_heuristic`, than the
  8 metered in MW, and the gap is largest at sunny middays, when embedded solar can push net flow
  through zero and an MVA meter reads the magnitude of that flow with no direction.
prediction: >-
  The sunny-midday skill against `manual_heuristic` is no more than 5 percentage points lower at
  the 8 MVA Primaries than at the 8 MW Primaries, so v1 needs no special handling. The share of
  sunny half-hours in which the magnitude of the power falls below 10% of its 99th percentile is
  within a factor of two between the two groups.
decision_rule: >-
  Skill is 100 x (1 - best-guess quantile score / `manual_heuristic` quantile score), pooled over
  the cases of a group. The decisive contrast is the skill at sunny middays at the MW Primaries
  minus the skill at the MVA Primaries, with whole months resampled (95% interval, 10,000 draws).
  `supported` (MVA clearly worse) when the lower end of the interval is above zero and the point
  estimate is 5 points or more. `refuted` (v1 can ignore the problem) when the upper end is under
  5 points. `inconclusive` otherwise, and the page says plainly that 8 series against 8 cannot
  tell. The 5-point threshold is set by what a special case costs: PVDIS-7 is research that NGED
  may not be able to support, so a gap has to be a meaningful share of the skill (the best guess
  is about 10% better than `manual_heuristic` overall) before it justifies the effort. A second
  interval resamples series within each group, to show how far the result rests on which 8 series
  are in each group.
benchmark: >-
  `manual_heuristic` on the same cases (experiment 33's stored per-case metrics).
row_set_id: >-
  fold_a_every5d_v1 (experiment 33's 73 runs for the 16 Primary series, July 2025 to June 2026,
  shared cases only, seeds averaged)
metric: >-
  Quantile score in points of capacity, as experiment 33 stores it per case.
---

# Experiment 56: do the Primaries metered only in MVA need special handling in v1?

*Plan; status `planned`. The write-up replaces this page.*

## Plan

**The question.** Eight of the 16 trial Primary substations are metered in MVA, so their series is
the magnitude of the power and carries no direction. The switching and solar design work expects v1
(the current forecast, which predicts that magnitude) to need no special handling there, but nobody
has measured it. This experiment re-scores experiment 33's stored forecasts. No weather model or
XGBoost model is trained.

**Method.**

1. Read the per-case quantile scores of the best guess (`n1_cs`) and `manual_heuristic` from
   experiment 33's `work/metrics/Primary/`, keep shared cases, and average the 3 seeds.
2. Read the units from the metadata table and the observed power (over capacity) from experiment
   33's `observed.parquet`.
3. Cut the cases four ways, each pre-registered: (a) sunny midday (valid time 10:00 to 13:30 UTC,
   April to September) against all other hours; (b) near zero net flow (the observed magnitude of
   power within 10% of capacity of zero) against the rest; (c) the two cuts crossed; (d) the whole
   year.
4. For each cut and each unit group, report the best guess's and `manual_heuristic`'s absolute
   scores, the skill, and the MW-minus-MVA gap in skill, with whole-month resampling.
5. Count, for each Primary, the sunny half-hours in which the magnitude of the power is below 10%
   of that series' 99th percentile, in the scored year and in all history from 2022. The count is
   reported for the MW series too, for comparison.
6. Label recall of experiment 41's labeller on MVA-side against MW-side pairs is optional and is
   left out of this plan; the write-up says so.

**Sizing (the five `plan-issue` triggers).** Stores: one private results table and 3 to 4 figures;
no predictions, because nothing is forecast. Serving path: untouched. Degradation rule: untouched.
Designs: the cuts and thresholds are fixed above, so one defensible design. Callers: reads two
stored tables and the metadata table; no code outside this folder. Size: simple.

**Earlier advice followed.** Experiment 44's advice to recompute the stored share of receiver is
not needed here, because no switching labels are used. The experiment reads experiment 33's
per-case table as experiment 44 does.

**Caveats written in advance.**

- Eight series against eight is a small sample, so unit group is confounded with which substations
  happen to be in each group (size, solar capacity, switching exposure).
- `manual_heuristic` forecasts the magnitude at the MVA series too, so skill against it measures
  the relative cost of MVA metering and not the absolute cost.
- Month resampling treats the series as fixed, so the series-resampled interval is the one to read
  for generalisation.
