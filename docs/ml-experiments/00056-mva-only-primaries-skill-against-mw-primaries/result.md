---
id: "00056"
slug: mva-only-primaries-skill-against-mw-primaries
status: done
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
verdict: inconclusive
best_arm: >-
  Not applicable (a re-scoring of one forecast). At sunny middays the best guess's skill against
  `manual_heuristic` is 15.9% at the 8 MW Primaries and 10.3% at the 8 MVA Primaries; the gap is
  5.6 points (95% interval -5.0 to 14.6, resampling months).
lessons: >-
  Nothing in the data says v1 needs special handling at the MVA Primaries at sunny middays, but
  8 series against 8 cannot rule a gap of up to 14 points out. The MVA Primaries' skill is lower
  all year (a gap of 10.5 points, 2.1 to 17.5), and at sunny middays it is higher than in other
  hours. One MW Primary with 30% skill carries most of the all-hours gap. Near-zero flow, the
  case an MVA meter hides, is rare at the MVA Primaries (12 sunny-midday half-hours in the year).
production_status: not-for-production
advice:
  - >-
    Do not build special handling for the MVA-only Primaries into v1 on the strength of this
    result: at sunny middays the best guess's skill is not lower than at the MW Primaries beyond
    the interval (gap 5.6 points, -5.0 to 14.6; figure 1).
  - >-
    Compare unit groups with one MW Primary's skill removed, or by series, because one MW Primary
    with 30% skill carries most of the all-hours gap (3.8 points without it; leave-one-out table).
  - >-
    Treat PVDIS-7 (MW at the MVA-only Primaries) as motivated by the solar disaggregation, not by
    forecast skill: near-zero flow at sunny middays is under 1.1% of half-hours at 7 of the 8 MVA
    Primaries (figure 3).
---

# Experiment 56: do the Primaries metered only in MVA need special handling in v1?

**The best guess forecasts the 8 trial Primary substations metered only in MVA no worse at sunny
middays than the 8 metered in MW, within what 8 series can show, so v1 can ignore the problem for
now; the experiment could not rule out a gap of up to 14 points.** Eight of the 16 trial
substations record only the size of the power (MVA), with no direction of flow, so a substation
where embedded solar pushes the net flow through zero could look odd to a forecast that predicts
that size. We took experiment 33's stored forecasts, measured how much better the best guess is
than a simple rule-based forecast (`manual_heuristic`) at each group of 8 substations, and split
the result by sunny midday and by near-zero flow. At sunny middays the best guess scored 15.9%
lower (better) than `manual_heuristic` at the MW substations and 10.3% lower at the MVA
substations, a 5.6-point gap whose 95% interval runs from -5.0 to 14.6 points.

## Result

**The verdict is `inconclusive`, by the rule fixed before the run.** The rule called MVA clearly
worse when the interval's lower end was above zero and the gap was 5 points or more, and called v1
safe to ignore when the upper end was under 5 points. The gap is 5.6 points, but its lower end is
below zero and its upper end is above 5.

**Skill is lower at the MVA Primaries all year, and not specifically at sunny middays.** Skill is
the percentage by which the best guess's quantile score is lower than `manual_heuristic`'s, pooled
over a group's cases (figure 1). In all hours, the MW group scores 14.97% and the MVA group 4.50%,
a gap of 10.5 points (2.1 to 17.5). Outside sunny middays the gap is 11.2 points (2.7 to 18.3), and at
sunny middays 5.6 points (-5.0 to 14.6). The MVA group's skill is higher at sunny middays (10.3%)
than in other hours (3.7%).

| Hours | Best guess, MW | `manual_heuristic`, MW | Best guess, MVA | `manual_heuristic`, MVA |
|---|---|---|---|---|
| All hours | 5.22 | 6.13 | 5.41 | 5.67 |
| Sunny midday | 7.23 | 8.60 | 7.47 | 8.34 |
| Other hours | 5.03 | 5.90 | 5.22 | 5.42 |

*Mean quantile score in points of capacity, July 2025 to June 2026, 16 Primaries.*

![Skill by unit group](figure_1_skill_by_unit.svg)

**The series are too few to separate the groups.** Resampling the series within each group, instead
of the months, gives a sunny-midday gap of 5.6 points with an interval of -2.8 to 14.0 (figure 2
shows each Primary). The all-hours gap's interval then runs from -2.0 to 20.2, which includes zero.

![Skill of each Primary at sunny midday](figure_2_skill_per_series.svg)

**Near-zero flow is rare at the MVA Primaries, so the worry has little to act on.** Only 106 of
403,712 MVA half-hours are within 10% of capacity of zero, 12 of them at sunny middays, against
2,392 of 364,361 for the MW group, 514 at sunny middays. At sunny middays the share of half-hours
with power under 10% of the series' 99th percentile is 0% at 5 of the 8 MVA Primaries and 8.7% at
the highest (series 26), against 22.9% and 19.8% at 2 MW Primaries (figure 3). The best guess's
skill at near-zero flow is 29.5% at the MVA group and 50.6% at the MW group (the MVA interval, 4.2
to 39.4, rests on 106 half-hours).

![Share of sunny-midday half-hours near zero flow](figure_3_near_zero_midday.svg)

## Post hoc check: one MW Primary

**One MW Primary (series 12, with 31% skill over all hours and 29.9% at sunny middays) carries
most of the MW group's edge.** Leaving it out, and nothing else, moves the sunny-midday gap from
5.6 points to -0.3 and the all-hours gap from 10.5 to 3.8. No other series moves the sunny-midday
gap by more than 1.4 points. This check was added after seeing the numbers, and the verdict does not
use it.

## Failures and caveats

- Eight series against eight cannot tell the groups apart beyond about 14 points at sunny middays,
  and the groups differ in size, solar capacity, and switching exposure as well as in meter unit.
- Skill is against `manual_heuristic`, which also forecasts the magnitude at the MVA Primaries, so
  the result measures the relative disadvantage of MVA metering and not an absolute one.
- Experiment 41's labeller recall on MVA-side against MW-side pairs, listed as optional in the
  brief, was not run.
- Series 7, 12, and 13 have shorter histories, so the all-history counts of low-magnitude
  half-hours (stored in the experiment's private folder) cover different periods; the figure uses
  the scored year only.

## Lessons for later experiments

**The MVA Primaries are no reason to delay v1, and no support for PVDIS-7 on forecast-skill
grounds.** The case for recovering MW at the MVA-only Primaries rests on separating solar from
demand, which the PVDIS series tests. A later comparison of unit groups should use more than 8
series or report each series, and should say which series carry the gap.
