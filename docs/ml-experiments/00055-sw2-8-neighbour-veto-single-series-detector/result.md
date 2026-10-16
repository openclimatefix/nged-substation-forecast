---
id: "00055"
slug: sw2-8-neighbour-veto-single-series-detector
status: done
size: medium
hypothesis: >-
  Keeping every alarm of the single-series detector D0 at the 24-hour window, except those vetoed
  because at least two neighbouring Primary substations stepped the same way and none stepped the
  opposite way, lets the alarm threshold fall at the same rate of unexplained alarms, and the
  detector then finds more injected load transfers than D0 alone.
prediction: >-
  Experiment 43's D0 24-hour alarms are the benchmark. At a veto of 2.0 noise units the veto removes
  14 of the 80 unexplained odd-month alarms (18%) and 16 of the 137 alarms that a logged closure
  explains. The matched 24-hour threshold falls by 0.05 to 0.20 from 4.00, so the gain through the
  threshold is small. C1 (24-hour window, injected transfers of 25% to 40% of the receiving load
  lasting over 1 day, starting in even months) rises by 0.00 to 0.02 from about 0.35, with a point
  value near +0.01. Experiment 47 saw the odd-month share found change by -0.019 when its threshold
  rose by 0.22, which puts the gain from a fall of 0.05 to 0.20 near +0.005 to +0.015. The veto
  removes the alarm that found fewer than 10% of the transfers the benchmark finds. Of the alarms on
  the real residual, the confirmed tier (an opposite neighbour step that balances the alarm's step)
  is explained by a logged closure more often than the unconfirmed tier.
decision_rule: >-
  The headline is C1, D2 minus D0M in the share of injected transfers found, paired by transfer, on
  the 24-hour window and the headline band, over transfers that start in even months, read at 98.75%
  (Bonferroni across C1 to C4) from 20,000 resamples of whole calendar months. D2's veto strength
  and threshold are chosen on odd months and on the injected transfers that start in odd months,
  never on even months. `supported` when the C1 interval lies wholly above zero, the C1 point value
  is at least +0.01, and D2's even-month unexplained-alarm count is no more than 10% above D0M's.
  `refuted` when the upper end of the C1 interval lies below +0.01. `inconclusive` otherwise. C2 to
  C4 and the real-event counts are reported without changing the verdict.
benchmark: >-
  D0M, D0 with its threshold chosen by the search D2 uses and no veto (so that neither detector
  gets a threshold step the other was not offered); D0 at experiment 43's thresholds (4-hour window
  6.0, 24-hour window 4.0) through experiment 43's harness, whose 600 injected transfers and
  outcomes the harness check must reproduce; D1 at its experiment 43 thresholds for reference; and a
  detector that raises no alarm.
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled transfers, as
  experiment 43 does, not forecasts)
metric: >-
  The share of injected transfers found (start within 2 hours and end within the larger of 6 hours
  and 10% of the duration), event F1 at the two precision bounds of experiment 43, the
  even-month unexplained-alarm count, and the share of found transfers whose receiver the alarm
  names.
verdict: inconclusive
best_arm: >-
  No arm wins. D2 (D0 with the neighbour veto, strength 1.5 noise units, matched 24-hour threshold
  3.70) finds 0.363 of the 620 headline-band transfers that start in even months against 0.355 for
  D0M, a difference (C1) of +0.008 (98.75% interval -0.003 to +0.022). The rule needs an interval
  wholly above zero and a point value of at least +0.01.
lessons: >-
  The veto is almost free: it loses 2 of the 220 transfers D0M finds and gains 7, raises the
  precision of the real alarms by about 3 points at 24 hours and 6 to 7 points at 4 hours, and the
  alarms that find a transfer with a metered receiver name the donor or a receiver in 88% of the
  cases. In this population its gain is unlikely to exceed 0.022, because it removes only about a
  third of the unexplained alarms and the threshold then falls by only 0.30. The veto removes 35% of
  the unexplained 24-hour alarms and 20% of the alarms a logged closure explains (a ratio of 1.8,
  under the 2 that would make it useful). Of experiment 41's 148 labelled alarms, 5 of 79
  confirmed, 4 of 49 unconfirmed, and 0 of 20 vetoed were visible transfers, so the confirmed tier
  is not a better mask for SW2-12.
advice:
  - >-
    Do not expect neighbour logic that only removes or re-ranks single-series alarms to lift the
    share of transfers found by more than about 0.02 at the 24-hour window, in the 620 injected
    transfers scored here (C1 interval).
  - >-
    Work on the transfers with no metered receiver, which are 44% of the headline band and which D0
    finds 0.17 of against 0.50 for the others, because they set the ceiling (C3 and C4).
  - >-
    Do not use the confirmed tier as a filter for training masks on this evidence: 5 of 79
    confirmed and 4 of 49 unconfirmed labelled alarms were visible transfers (tier figure).
  - >-
    Check the veto on transfers that start within 1 day of a bank holiday before relying on it,
    since there it lost 6 of 94 transfers and gained 2, a net loss of 4 of 300 (holiday set).
production_status: not-for-production
adopted_in: none
pull_request: https://github.com/openclimatefix/nged-substation-forecast/pull/1282
code: ml-experiments/00055-sw2-8-neighbour-veto-single-series-detector/
---

# Experiment 55: does a neighbour veto help the single-series switching detector?

**A neighbour veto makes the single-series detector slightly more precise but cannot make it find
many more transfers: the most it adds is about 2 points of the share found, so the next gain has to
come from the transfers that have no metered receiver.** We kept every alarm of the single-series
detector D0, which raises an alarm when the 24-hour mean of one Primary substation's residual
(measured minus predicted power) steps up or down. We then removed the alarms that look like a
weather or calendar miss, because at least two neighbouring Primary substations stepped the same
way and none stepped the opposite way. With those alarms gone the threshold could fall from 4.00 to
3.70 noise units at the same rate of unexplained alarms, and the detector found 0.363 of 620
injected transfers against 0.355 without the veto (+0.008, 98.75% interval -0.003 to +0.022). The
verdict is inconclusive. In the 620 injected transfers scored here, the veto's gain is unlikely to
exceed 0.022, so refining the neighbour logic is unlikely to lift recall much. D2 denotes D0 with
the veto, and the transfers with no metered receiver are 44% of the headline band, of which D0 finds
0.17 and D2 finds 0.19. Those transfers are where the share found has room to rise.

![The matched threshold falls from 4.00 to 3.70 with the veto](figure_1_operating_points.svg)

## Verdict

**C1, D2 minus D0M in the share of transfers found, is +0.008 (98.75% interval -0.003 to +0.022).**
D0M is D0 with its threshold chosen by the same search as D2's and no veto, so that neither
detector gets a threshold step the other was not offered. The search returned D0's own thresholds
(4.00 and 6.00), so D0M and D0 at experiment 43's thresholds are the same detector. The population
is the 620 injected transfers that move 25% to 40% of the receiving load for over 1 day and start in
even months: 89 of experiment 43's 160 and 531 of the 1,000 new ones. The rule asked for an
interval wholly above zero and a point value of at least +0.01. The rule called the hypothesis
refuted only if the upper end fell below +0.01. The verdict is therefore inconclusive. Of the 620
transfers, D2 finds 7 that D0M misses
and misses 2 that D0M finds.

![C1 to C4 with intervals](figure_3_contrasts.svg)

| Contrast | Window, transfers | D2 | D0M | D2 minus D0M | 98.75% interval |
|---|---|---|---|---|---|
| C1 | 24 h, all (620) | 0.363 | 0.355 | +0.008 | -0.003 to +0.022 |
| C2 | 4 h, all (620) | 0.237 | 0.231 | +0.007 | -0.004 to +0.015 |
| C3 | 24 h, metered receiver (346) | 0.500 | 0.503 | -0.003 | -0.007 to +0.000 |
| C4 | 24 h, no metered receiver (274) | 0.190 | 0.168 | +0.022 | +0.000 to +0.041 |

**The gain sits in the transfers that have no metered receiver.** For those, the lower threshold
finds 6 more of 274. For transfers with a metered receiver, the veto changes nothing (-0.003). D0
finds 0.50 of the transfers with a metered receiver and 0.17 of those without, possibly because two
substations stepping give two chances to raise an alarm.

## What held in the prediction and what did not

- **Held: C1 lay in the predicted 0.00 to 0.02** (+0.008, point value just under the +0.01 bar).
- **Held: the veto cost few found transfers.** It removed 2 of the 220 transfers D0M finds, under
  the predicted 10%.
- **Partly held: the veto separates unexplained alarms from logged ones weakly.** The prediction
  gave 18% of the unexplained odd-month alarms and 12% of the logged ones at a veto of 2.0 on the
  original thresholds. At the matched threshold of the chosen veto (1.5) it removes 37% of the
  unexplained 24-hour alarms in odd months, 35% of those in all months, and 20% of the alarms a
  logged closure explains, a ratio of 1.8. At 4 hours the ratio is 1.5 (38% to 25%). The 35 vetoed
  24-hour alarms that a logged closure explains are spread over 12 of the 16 series.
- **Held: the confirmed tier is explained by a logged closure more often.** At 24 hours 64% of
  confirmed alarms, 46% of unconfirmed ones, and 30% of vetoed ones lie within 24 hours of a logged
  closure.
- **Held at veto 2.0, exceeded at the chosen strength: the matched threshold fell by 0.15 at 2.0
  (predicted 0.05 to 0.20) and by 0.30 at the chosen 1.5.** Odd months chose the strongest
  veto, 1.5 noise units (odd-month share found 0.304, against 0.287 at 2.0 and 0.274 at 3.0). The
  threshold fell by 0.15 at 2.0 and 0.00 at 3.0. The strongest veto was the edge of the grid, so a
  stronger one was not tried.

![Share of alarms vetoed, by class of alarm](figure_2_veto_rates.svg)

## Precision, and what the confirmed tier tells us

**The veto raises the precision of the real alarms by about 3 points at 24 hours and 6 to 7 points
at 4 hours.** In even months, the lower
precision bound (alarms within 24 hours of a logged closure over all alarms) of the 24-hour detector
rises from 0.441 to 0.474 and event F1 (the harmonic mean of the share found and the precision) from
0.393 to 0.411. At 4 hours the bounds rise from 0.471 and 0.528 to 0.538 and 0.588, and F1 from
0.310 and 0.321 to 0.329 and 0.338. D2 raises 137 24-hour alarms in even months against 143 for D0M,
and 72 unexplained ones against 80, so D2 sits at a slightly stricter operating point, which biases
the comparison against D2.

**The confirmed tier is not more often a visible transfer.** Of experiment 41's 148 labelled alarms,
5 of the 79 confirmed, 4 of the 49 unconfirmed, and 0 of the 20 vetoed were labelled visible
transfers. The sample is too coarse to separate the tiers, so the 148 labelled alarms do not support
using the confirmed tier as a training mask for the normal-running-arrangement forecast.

![Confirmed alarms match logged closures more often, but not visible transfers](figure_5_tiers.svg)

**D2 finds 88% of the transfers with a metered receiver through an alarm that names the donor or a
receiver.** Of the 360 such transfers that D2 finds at 24 hours, over all months, 316 are found by
such an alarm.

## Near bank holidays

**The veto loses transfers near bank holidays: of 300 transfers that start within 1 day of one,
D0M finds 94 and D2 finds 90.** D2 loses 6 and gains 2. The holiday set is drawn with logged
closures only, and 206 of its 300 onsets lie within 2 days of an alarm of experiment 43's four
settings. Under experiment 43's full quiet mask, only 94 of the 300 would have been kept (a separate
count from the 94 that D0M finds). Experiment 43's 160 transfers are found at 47 (D0M) and 53 (D2),
and the new 1,000 at 321 and 336.

![Share found by set](figure_4_found_by_set.svg)

## Real transfers

D2 finds 2 of the 14 labelled real transfers of 25% or more of the receiving load that last over 1
day, against 1 for D0M, and 3 of the 35 labelled transfers over 1 day against 2. At 4 hours the
counts are 0 of 14 and 1 of 35 for both detectors. The real-transfer counts carry no verdict.

## Limits

- **The veto strength sits at the edge of the grid.** The strongest value, 1.5 noise units, won on
  odd months.
- **The injected transfers sit away from alarms.** Experiment 43's quiet mask keeps the original and
  new sets at least 2 days from an alarm of its four settings on every series, which understates how
  often a wrong veto happens. The holiday set drops that condition.
- **D0M equals D0.** The matching search returned D0's thresholds, so comparing with D0M adds
nothing beyond comparing with D0.
- **The residual needs ERA5.** ERA5 (the ECMWF Reanalysis, version 5) arrives about 5 days after the
event, so the detectors run retrospectively.

## What the next experiment should answer

**The next experiment should raise the share of transfers found among those with no metered
receiver, which D0 finds at 0.17.** In this experiment, neighbour logic reaches only the transfers
that a neighbour can see, and those are already found at 0.50. The shape-aware switching detector
planned as SW2-7 models the moved load as a share of the donor's expected daily profile, which gives
a single series a better step statistic than a constant shift. The plan for SW2-7 should score the
C3 and C4 split first, take several hundred headline-band transfers in even months as here, and keep
the veto of this experiment as an optional layer on top, since it costs 2 of 220 found transfers and
raises precision.

Code: `ml-experiments/00055-sw2-8-neighbour-veto-single-series-detector/`, with every number
printed by `e00055_report.py`.
