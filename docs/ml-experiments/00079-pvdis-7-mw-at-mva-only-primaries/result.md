---
id: "00079"
slug: pvdis-7-mw-at-mva-only-primaries
status: planned
size: medium
hypothesis: >-
  At a Primary substation (a substation that steps 33 kV down to 11 kV) whose meter records only the
  magnitude of apparent power in megavolt-amperes (MVA), a least-squares fit of the magnitude of
  "calendar-and-temperature demand minus capacity times a solar shape" recovers the solar capacity
  and the direction of flow, as long as the power factor is near 1. The fit alternates between
  choosing a sign for each half-hour and fitting the signed model to the signed series. It is tested
  first on the 8 Primaries that are metered in MW, by hiding the sign of their readings (taking the
  absolute value) and recovering it.
prediction: >-
  On the Primaries that almost never go negative and carry no injection, the fit finds reverse flow
  in at most 2% of half-hours. With the output of a metered solar farm subtracted so that reverse
  flow is common (an injected capacity equal to the window's median absolute midday power, "level
  1.0", and 1.5 times that, "level 1.5"), the fit reproduces the capacity of the signed fit to a
  median relative error of 0.15 to 0.30, against 0.5 to 1.0 for the naive fit (the magnitude fitted
  as if it were signed), and recovers the direction of flow with a balanced accuracy (the mean of
  the share of reverse-flow half-hours and the share of forward-flow half-hours given the right
  sign) of 0.80 to 0.90. Level 0.5 has little reverse flow and serves as the no-signal check. A
  constant reactive-power floor that the fit does not know about lowers both by 0.05 to 0.15 at 20%
  of midday power, and breaks the direction at 40%. Probabilities of the verdicts: `supported` 0.30,
  `partly supported` 0.35, `refuted` 0.25, `inconclusive` 0.10.
decision_rule: >-
  Windows are one Primary in one season-year, April to September, as in experiment 00073. Cases are
  (Primary, farm, season-year) windows, and every interval is a two-stage cluster bootstrap: resample
  the Primaries, then resample the (farm, season-year) windows within each drawn Primary, 99%
  (a Bonferroni level for five contrasts), 4000 resamples, seed 79. With fewer than 6 distinct
  Primaries contributing to a contrast the verdict is `inconclusive`. The capacity truth is the
  capacity of the signed fit to the unhidden series, so capacity contrasts measure the unfolding
  step and not the real solar capacity. Relative capacity error is the absolute difference between
  the fitted and the signed-fit capacity, divided by the latter. The error against the injected
  capacity is reported beside it. The direction truth is the sign of the observed signed power; rows
  with an absolute power under 2% of the window's median absolute midday power are dropped. A window
  enters a sign score only with at least 50 reverse-flow and 50 forward-flow rows, and the number of
  windows dropped for that reason is reported per level. Balanced accuracy pools the counts of
  correct and incorrect rows over the windows of a Primary, then the Primaries are averaged. C1a and
  C1b are the median relative capacity error of the unfolding fit minus that of the naive fit, at
  level 1.0 and level 1.5. C2a and C2b are the balanced accuracy of the unfolding fit minus 0.5 (the
  all-positive guess), at the same levels. C3 is C1a on windows with a reactive-power floor of 20%
  of midday power. Capacity is `identified` when the median relative capacity error of the unfolding
  fit is at most 0.25 at both levels and the upper ends of C1a and C1b are below zero. The 25%
  limit equals the shortfall experiment 00039 already accepted, where a least-squares fit recovered
  0.77 of an injected capacity. Direction is `identified` when the balanced accuracy is at least
  0.80 at both levels, the lower ends of C2a and C2b are at least 0.25, and the median fitted
  reverse-flow share on the no-injection windows of the Primaries that almost never go negative (D5,
  D9, D11, D14, D16) is at most 0.02. That last clause is a null check: a fit that never reports
  reverse flow passes it, and the balanced-accuracy clause is what shows the fit can find reverse
  flow. `supported` when both parts are identified, `partly supported` when exactly one is,
  `refuted` when neither is and the self-tests pass, `inconclusive` otherwise. C3, the other
  reactive-power floors (10% and 40%), level 0.5, and the false reverse flow at level 0.5 are
  reported and never change the verdict. The fit is also applied to the 8 MVA-only Primaries,
  whatever the verdict. That output is descriptive: the fitted capacity, the fitted reverse-flow
  share, and the loss reduction against the naive fit, per Primary, with a week-block bootstrap
  interval. The write-up calls it a measurement only where the matching part was identified, and
  says that no signed series exists at those Primaries to check it against. The threshold for
  porting is not set, because the output is an offline table and the production forecast does not
  read it.
benchmark: >-
  The naive fit (the magnitude fitted as if signed), the all-positive sign guess (balanced accuracy
  0.5), and the fit to the unhidden signed series, whose capacity is the truth.
row_set_id: none
metric: >-
  Relative capacity error, balanced accuracy of the sign, and the fitted reverse-flow share.
code: ml-experiments/00079-pvdis-7-mw-at-mva-only-primaries/
production_status: not-for-production
adopted_in: none
---

# Experiment 79: can the direction of flow be recovered at Primaries metered only in MVA?

**Status: planned. The text below is the plan and is replaced by the write-up when the experiment
has run.**

## Plan

**Question.** Eight of the 16 trial-area Primaries have a meter that records only the magnitude of
apparent power (MVA). When embedded solar pushes net flow through zero, that reading bounces off
zero and shows no direction. NGED cannot supply real and reactive power at these 8, so the only route
to MW is the model. The fit assumes a power factor of 1.

**Method.** Each window's model is as in experiment 00073: power equals a calendar cell
(month, half-hour, day type) plus heating degree-days and cloud terms, minus a capacity `c >= 0`
times a solar shape (the fleet orientation mix of 0.2 east, 0.6 south, and 0.2 west at 25° tilt,
clipped at a DC:AC ratio of 1.15). The unfolding fit sees only `|power|`. It starts from a set of
sign patterns (all positive, and the patterns implied by capacities of 0.5, 1, 1.5, and 2 times the
window's median absolute midday power), alternates between fitting the model to `sign x |power|` and
resetting each sign to the sign of the fitted value, and keeps the start with the lowest squared
error. The naive fit is the first pass of the all-positive start.

**Known answers.** (1) Injected: the output of each metered solar farm (A to F, normalised to a 99th
percentile of 1) is scaled and subtracted from each of the 8 MW Primaries' power, at levels 0.5, 1.0,
and 1.5 times the window's median absolute midday power, then the sign is hidden. The truth is the signed
fit to the same series. (2) Reactive floor: the same at level 1.0, with `sqrt(power^2 + q^2)` taken
as the reading, where `q` is 0.1, 0.2, or 0.4 times the window's median absolute midday power, unknown
to the fit. Only 0.2 enters a contrast (C3). (3) No injection: the sign of the five MW Primaries that
almost never go negative is hidden, and the share of half-hours where the fitted flow is negative counts
false reverse flow. (4) Natural reverse flow: Wrangle T1 and T2 and Spilsby go negative in the
real data; the sign is hidden and recovered with no injection, where a window has at least 20
reverse-flow half-hours.

**Self-tests.** On a synthetic series with a known capacity and no noise, the fit recovers it to
1%. On series with capacity 0 and a small capacity that never nears zero, the unfolding fit returns
an all-positive sign pattern. A start that flips the signs wins only if it lowers the loss by more
than 0.5% of the all-positive start's loss. The run records, per window, the number of distinct
capacities the starts reach and the loss gap between the best and second-best start. Before any fit
runs, a script tabulates the reverse-flow rows per window and level, and the midday minimum of the
magnitude at each MVA-only Primary, to show whether level 1.0 resembles the real series.

**Size.** Medium. What gets stored: private tables only. Serving path: untouched. Degradation
rule: untouched. More than one defensible design: yes (alternating signs, a grid over capacity, or
a mixture fit), so the plan is reviewed once. Callers: the experiment copies experiment 00073's
fitting code, and nothing imports it.

**Earlier advice followed.** Experiments 00039 and 00056: leave the MVA Primaries out of a signed
capacity estimate (here they are fitted with the unfolding fit and reported apart), and report the
share of near-zero half-hours. Experiment 00039: injection into a linear fit tests the shape only;
the magnitude link is non-linear, so injection tests the fit here. Experiment 00045: use the
season-year constant capacity, which this experiment does. No earlier advice item is departed from.

## Result

Not yet run.
