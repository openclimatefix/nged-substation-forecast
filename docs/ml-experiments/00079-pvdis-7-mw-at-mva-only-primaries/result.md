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
  Where reverse flow is rare or absent, the fit finds none: on the Primaries that never go negative
  and carry no injection, the fitted flow is negative in at most 2% of half-hours. With the output of
  a metered solar farm subtracted so that reverse flow is common (an injected capacity equal to the
  window's median midday demand, "level 1.0"), the fit recovers the solar capacity to a median
  relative error of 0.15 to 0.30, against 0.5 to 1.0 for fitting the magnitude as if it were signed
  (the naive fit), and recovers the direction of flow with a balanced accuracy (the mean of the shares
  of reverse-flow and forward-flow half-hours given the right sign) of 0.80 to 0.90. A constant
  reactive-power floor, which the fit does not know about, lowers both by 0.05 to 0.15. At the 8
  MVA-only Primaries the fit finds reverse flow in under 1% of half-hours at most of them, so that MW
  equals MVA there. Probabilities of the verdicts: `supported` 0.30, `partly supported` 0.35,
  `refuted` 0.25, `inconclusive` 0.10.
decision_rule: >-
  Windows are one Primary in one season-year, April to September, as in experiment 00073. Cases
  are (Primary, farm) pairs, and every interval is a cluster bootstrap over pairs, 98.33% (a
  Bonferroni level for three contrasts), 4000 resamples, seed 79. With fewer than 6 distinct
  Primaries or 20 pairs in a contrast the verdict is `inconclusive`. C1 is the median relative
  capacity error (the absolute difference between the fitted capacity and the capacity of the fit
  to the unhidden signed series, divided by the latter) of the unfolding fit minus that of the naive
  fit, on injected windows at level 1.0. C2 is the balanced accuracy of the unfolding fit's sign
  minus 0.5 (the all-positive guess), on the same windows. C3 is C1 on the windows with the
  reactive-power floor. Capacity is `identified` when the median relative capacity error of the
  unfolding fit is at most 0.25 at levels 1.0 and 1.5 and the upper end of C1 is below zero. The
  25% limit equals the shortfall experiment 00039 already accepted, where a least-squares fit
  recovered 0.77 of an injected capacity. Direction is `identified` when the median balanced
  accuracy is at least 0.80, the lower end of C2 is at least 0.25, and the median fitted
  reverse-flow share on the no-injection windows of the Primaries that never go negative (D5, D9,
  D11, D14, D16) is at most 0.02. `supported` when both are identified, `partly supported` when
  exactly one is, `refuted` when neither is and the self-tests pass, `inconclusive` otherwise. C3 is
  reported and never changes the verdict. The fit is applied to the 8 MVA-only Primaries whatever the
  verdict, but the write-up calls its output a measurement only where the matching part was
  identified. The threshold for porting is not set, because the output is an offline table and the
  production forecast does not read it.
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
window's median midday demand), alternates between fitting the model to `sign x |power|` and
resetting each sign to the sign of the fitted value, and keeps the start with the lowest squared
error. The naive fit is the first pass of the all-positive start.

**Known answers.** (1) Injected: the output of each metered solar farm (A to F, normalised to a 99th
percentile of 1) is scaled and subtracted from each of the 8 MW Primaries' power, at levels 0.5, 1.0,
and 1.5 times the window's median midday demand, then the sign is hidden. The truth is the signed
fit to the same series. (2) Reactive floor: the same at level 1.0, with `sqrt(power^2 + q^2)` taken
as the reading, where `q` is 0.2 times the window's median midday demand (a power factor of about
0.98 at typical load), unknown to the fit. (3) No injection: the sign of the five MW Primaries that
never go negative is hidden, and the share of half-hours where the fitted flow is negative counts
false reverse flow. (4) Natural reverse flow: Wrangle T1 and T2 and Spilsby go negative in the
real data; the sign is hidden and recovered with no injection, where a window has at least 20
reverse-flow half-hours.

**Self-tests.** On a synthetic series with a known capacity and no noise, the fit recovers it to
1%. On a series that never nears zero, the unfolding fit equals the naive fit.

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
