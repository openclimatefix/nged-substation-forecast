---
id: "00079"
slug: pvdis-7-mw-at-mva-only-primaries
theme: solar-disaggregation
also_relevant_to: []
status: done
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
  (Primary, farm, season-year) windows, and every interval is a two-stage cluster bootstrap:
  resample
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
verdict: partly supported
verdict_note: >-
  Read by the pre-registered rule. Capacity is identified. Direction is not, because the lower ends
  of C2a and C2b (0.157 and 0.184) are under the 0.25 limit and the balanced accuracy at level 1.0
  (0.779) is under 0.80.
best_arm: >-
  The unfolding fit. At injection level 1.0 its median relative capacity error is 0.092 against
  0.290 for the naive fit, and its balanced accuracy of flow direction is 0.779.
headline: >-
  C1a (unfolding minus naive median capacity error, level 1.0) is -0.198 (99% interval -0.409 to
  -0.068); C1b (level 1.5) is -0.887 (-0.920 to -0.754). C2a (balanced accuracy minus 0.5) is
  +0.279 (+0.157 to +0.337); C2b is +0.311 (+0.184 to +0.372). Without Wrangle T1 the balanced
  accuracy is 0.819 (0.791 to 0.843) at level 1.0 and 0.856 (0.831 to 0.874) at level 1.5.
take_home_message: >-
  Hiding the sign of the MW series and recovering it works for the solar capacity (median error
  0.09 to 0.11 of the signed fit's, against 0.29 to 1.00 for the naive fit) but not to the pre-
  registered level for the direction of flow (balanced accuracy 0.78 to 0.81, rule 0.80), so the
  verdict is partly supported. At 7 of the 8 Primaries metered only in MVA, the fit found no
  reverse flow.
lessons: >-
  A fit of the magnitude of demand minus solar recovers the solar capacity through zero flow, and
  finds the direction of flow with a balanced accuracy of 0.78 to 0.86, except where reverse flow is
  the
  usual state, as at Wrangle T1. A reactive-power floor of 20% of midday power already lowers the
  capacity recovery to a median error of 0.31 and the balanced accuracy to 0.67. On the real
  series the fit finds no reverse flow at 7 of the 8 MVA-only Primaries, and finds a flow that is
  negative on 75% to 87% of daylight half-hours at Stickney from 2021, which cannot be checked.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Read the MVA reading as a stand-in for MW at the 7 MVA-only Primaries other than Stickney only
    with the caveat that the fit finds no reverse flow there at a recall of about 0.69, and that
    Ingoldmells has a bootstrap 95th percentile of 9.5% negative half-hours (figure 4).
  - >-
    Ask NGED for the direction of flow, or a few weeks of signed MW, at Stickney before using any
    fitted capacity or sign there, because the fit's answer there cannot be checked and the fit
    fails at the one MW Primary where reverse flow is the usual state (post hoc section).
  - >-
    Test a new starting sign pattern on Wrangle T1 before fitting a Primary whose reverse flow may
    be the usual state, because both the registered starts and the post hoc threshold starts return
    capacity 0 there (figure 3, post hoc section).
  - >-
    Do not apply the sign recovery where reactive power may exceed about 10% of midday power, a
    conservative reading between the 10% and 20% runs, because 20% already cuts the balanced
    accuracy at three of eight Primaries to 0.50 to 0.61 (reactive-power rows, figure 3).
  - >-
    Report capacity from the magnitude against the signed fit of the same series and not against
    the injected farm alone, because the two differ by the signed fit's own error (0.092 against
    0.151 at level 1.0, and 0.113 against 0.073 at level 1.5).
---

# Experiment 79: can the direction of flow be recovered at Primaries metered only in MVA?

**Hiding the sign of a Primary's power and recovering it works for the solar capacity, and finds the
direction of flow with a balanced accuracy of 0.78 (0.5 is a guess), but the 8 Primaries metered
only in MVA show almost no reverse flow to recover, and the one that might (Stickney) cannot be
checked.** A Primary is a substation that steps 33 kV down to 11 kV. At 7 of the 8 Primaries metered
only in megavolt-amperes (MVA), the MVA reading can stand in for megawatts (MW) from April to
September, so no fit is needed there. Stickney is the exception, and the next step is a few weeks of
signed power from NGED for Stickney. The verdict is `partly supported`: the capacity part passed the
pre-registered rule and the direction part did not.

We asked whether the direction of the power flowing through a Primary, and the capacity of the solar
panels behind it, can be found from a meter that records only the size of the flow, not its
direction. First we tested the method where the answer is known. We took the 8 Primaries whose
meters record signed power in MW. We subtracted the recorded output of a real solar farm, scaled to
the Primary's typical midday power (level 1.0) or to 1.5 times that (level 1.5), so that the flow
reversed on sunny middays. Then we threw the sign away and asked the fit to recover the capacity and
the direction. Last, we ran the same fit on the real readings of the 8 other Primaries, which are
metered only in MVA. The median error of the fitted capacity was 0.09 to 0.11 of the capacity found
by fitting the unhidden signed power. A fit that treated the magnitude as if it were signed (the
naive fit) had median errors of 0.29 to 1.00. The fit gave the right direction of flow with a
balanced accuracy of 0.78 to 0.81 (the mean of the share of reverse-flow half-hours and the share of
forward-flow half-hours labelled correctly), or 0.82 to 0.86 after leaving out one Primary where
reverse flow is the usual state. The balanced accuracy fell to 0.67 when the reading also held
reactive power equal to 20% of midday power, which the fit does not know about. At 7 of the 8 real
Primaries metered only in MVA, the fit found no reverse flow.

## Result

**Capacity is identified, and direction is not, by the rule fixed before the run.** In this rule,
capacity needed a median relative error of at most 0.25 and an upper interval end below zero for
the difference from the naive fit (contrast C1). Direction needed a balanced accuracy of at least
0.80 and a lower interval end of at least 0.25 above the all-positive guess of 0.5 (contrast C2).
A window is one Primary in one April-to-September season, and the years run from September to
August as in experiment 00073. Intervals are 99% from a bootstrap that resamples Primaries and then
windows. The unfolding fit is the method under test: it alternates between choosing a sign for each
half-hour and fitting the signed model to the signed series.

| Injected solar | Capacity error, unfolding fit | Capacity error, naive fit | C1: unfolding minus naive | Balanced accuracy | C2: accuracy minus 0.5 |
|---|---|---|---|---|---|
| Level 1.0 (227 windows) | 0.092 | 0.290 | -0.198 (-0.409 to -0.068) | 0.779 | +0.279 (+0.157 to +0.337) |
| Level 1.5 (227 windows) | 0.113 | 1.000 | -0.887 (-0.920 to -0.754) | 0.811 | +0.311 (+0.184 to +0.372) |

*Median relative capacity error against the signed fit, and balanced accuracy of the fitted flow
direction, 8 MW Primaries and 6 farms. Four windows at level 1.0 and one at level 1.5 had too few
reverse-flow half-hours to score for sign.*

![Recovered capacity against the signed fit's capacity](figure_2_capacity_recovery.svg)

**Against the capacity of the injected farm alone, the median error is 0.151 at level 1.0 and 0.073
at level 1.5,** where against the signed fit it is 0.092 and 0.113, so the signed fit's own error is
part of every figure in the table.
Level 0.5 has little reverse flow at 7 of the 8 Primaries and is a no-signal check. The median
relative capacity error is 0.004 for the unfolding fit and 0.001 for the naive fit, and the median
fitted negative share is 0 (single windows reach 0.21). On the
30 windows of the 5 Primaries that almost never go negative, with no injection, the median fitted
negative share is 0.0000, so the fit does not invent reverse flow. The no-injection check passes
trivially for a fit that never finds reverse flow.

**A reactive-power floor the fit cannot see lowers both parts quickly.** The reading was
`sqrt(power² + q²)` with `q` equal to 10%, 20%, or 40% of midday power (level 1.0). The unfolding
fit's capacity error is 0.130, 0.309, and 0.529 against the naive fit's 0.331, 0.404, and 0.539, and
its balanced accuracy is 0.756, 0.666, and 0.561. At 20%, three Primaries (Horncastle, Trusthorpe,
and Donington) fall to a recall of reverse flow of 0.01 to 0.23. The reactive-power result is
descriptive and did not change the verdict. A power factor of 1 is therefore a strong assumption for
the direction, and
nothing in this experiment measures the real reactive power at any Primary.

![Balanced accuracy of the fitted flow direction](figure_3_direction.svg)

## Post hoc: Wrangle T1 and Stickney

**One MW Primary, Wrangle T1, drags the direction score below its limit, and the same weakness may
hide Stickney.** Wrangle T1 is net generation on 59% to 94% of April-to-September daylight
half-hours in its two seasons. With
level 1.0 injected, the fit returns capacity 0 and no negative flow there, against a signed capacity
of 1.73 MW, so its balanced accuracy is 0.5. Without it, the balanced accuracy is 0.819 (0.791 to
0.843) at level 1.0 and 0.856 (0.831 to 0.874) at level 1.5, with 0.69 and 0.77 of reverse-flow
half-hours found and 0.95 and 0.94 of forward-flow half-hours kept. The split between Wrangle T1 and
the other Primaries was chosen after seeing the result and does not change the verdict.

![Raw readings at the 8 MVA-only Primaries](figure_1_raw_profile.svg)

**Stickney's reading rises on sunny middays, which is what reverse flow would look like, and the
registered fit does not find reverse flow there.** Stickney's median midday reading is about 2.4
times its typical reading on sunny days and about 0.8 times on cloudy days (figure 1). Ingoldmells
is flat, and the other six MVA-only Primaries fall on sunny days by 1% to 38% of their typical
reading. Experiment 00039 found a negative capacity at Stickney in every
year. With extra starting sign patterns that flip every half-hour above a solar-output threshold (a
post hoc change, not tried on the known answers) the fit reaches a capacity of 1.3 to 7.2 MW with a
negative flow in 75% to 87% of daylight half-hours from 2021, and lowers the loss by 9% to 18% in 4
of 5 windows. Because the same fit fails at Wrangle T1, where the answer is known, the Stickney
capacity from the post hoc starts is not evidence.

## The real series

**At 7 of the 8 MVA-only Primaries the fit finds no reverse flow, and its capacity equals the naive
fit's** (figure 4). The median fitted negative share is 0.0% at all of them except Stickney, where
the registered fit gives 0.7% and a capacity of 0. The fitted capacities are 6.0 MW at Warth Lane
Skegness, 2.5 at Sleaford Road Boston, 1.4 at Langrick, 1.1 at Alford, Chapel St Leonards, and Marsh
Lane, and 0.7 at Ingoldmells. These capacities come from a fit that cannot be checked against signed
power, and
a load that tracks sunshine would give the same numbers. The 95th percentile of the negative share
over resampled weeks is 0.095 at Ingoldmells, 0.054 at Stickney, and under 0.001 at the others. The
5th to 95th percentile of the fitted capacity runs from 0 to 3.4 MW at Ingoldmells and from 0 to 2.4
MW at Marsh Lane. The lowest midday readings are 0.017 MW at
Stickney, 0.026 MW at Marsh Lane, and 0.099 MW at Langrick; the other five stay above 0.87 MW.

![Fitted capacity and reverse-flow share at the MVA-only Primaries](figure_4_mva_primaries.svg)

## Failures and caveats

- All 6 solar farms (A to F) are used for the injection, because the fit has no tuning parameter
  that
  could be chosen to favour a farm. The 8 MW Primaries are 8 independent clusters in the bootstrap,
  so every conclusion about direction rests on 8 Primaries, one of them an outlier.
- The capacity truth is the signed fit, so a capacity error here measures the unfolding step and not
  the real solar capacity. Real capacity at the MVA-only Primaries has no truth.
- Natural reverse flow, with no injection, gave only 5 eligible windows (Spilsby in three
  season-years and Wrangle T1 in two), so known answer (4) of the plan says little. Spilsby's
  recall of reverse flow is 0.35.
- The unfolding fit's starting sign patterns are sized from the magnitude reading alone, and not
  from the scale that sizes the injection, so the fit never begins at the answer.
- The fit's capacity at the MVA-only Primaries uses the fleet shape (0.2 east, 0.6 south, 0.2 west
  at
  a DC:AC ratio of 1.15) and the nuisance terms of experiment 00073. The capacity is not corrected
  for the 0.77 recovery that experiment 00039 found.

## Lessons for later experiments

**The sign can be recovered where reverse flow is sometimes present, and the capacity follows, but
the fit has no validated answer for a Primary that exports power on most sunny days.** Wrangle T1 is
the known answer where the fit fails, and Stickney is the real Primary that looks like Wrangle T1.
The registered starts and the post hoc starts both need a known answer in that regime before any
Stickney value is quoted.

## What the next experiment should answer

**Does a starting pattern that flips every daylight half-hour of the clearest days recover the
capacity and direction at Wrangle T1, and, if so, does the pattern give a consistent answer at
Stickney?** The threshold-flip starts of the post hoc section already return capacity 0 at Wrangle
T1 in both seasons. The test should
use injection at levels where reverse flow is the usual state (level 2.0 and above), keep the 50-row
sign rule, and add the post hoc starts to the self-tests. The test should also check that the other
seven MVA-only Primaries, which show no reverse flow, still return the naive fit. Signed power from
NGED for Stickney on even a few days would settle the question without a fit.

## The plan as registered

**Question.** Eight of the 16 trial-area Primaries have a meter that records only the magnitude of
apparent power (MVA). When embedded solar pushes net flow through zero, that reading bounces off
zero and shows no direction. NGED cannot supply real and reactive power at these 8, so the only
route
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
percentile of 1) is scaled and subtracted from each of the 8 MW Primaries' power, at levels 0.5,
1.0,
and 1.5 times the window's median absolute midday power, then the sign is hidden. The truth is the
signed
fit to the same series. (2) Reactive floor: the same at level 1.0, with `sqrt(power^2 + q^2)` taken
as the reading, where `q` is 0.1, 0.2, or 0.4 times the window's median absolute midday power,
unknown
to the fit. Only 0.2 enters a contrast (C3). (3) No injection: the sign of the five MW Primaries
that
almost never go negative is hidden, and the share of half-hours where the fitted flow is negative
counts
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
