---
id: "00071"
slug: sw2-18-refit-offset-and-gap-bridging
status: done
size: medium
hypothesis: >-
  The detector D0M with the end rule PF14 (experiment 69's benchmark) cannot raise an alarm, or
  accept a return step, within 24 hours of the monthly refit of the residual's XGBoost model.
  Computing the step statistic from whichever of two monthly-refit models, refitted half a month
  apart, did not refit within 24 hours of the stamp removes that blind spot, and finds more of the
  moved MW than PF14 at the same number of unexplained alarms. Bridging data gaps is a side check
  with no verdict, because the panel holds almost no gaps to bridge.
prediction: >-
  Written before any model is fitted or any transfer scored. Alarms are suppressed on 2 of every 30
  days (24 hours either side of the refit), so about 6.7% of transfer onsets fall where PF14 cannot
  raise an alarm, and PF14 finds about 0.67 of the moved MW elsewhere. Removing the blind spot
  gains about +0.045 from onsets and a little more from return steps, but un-suppressing the zones
  adds unexplained alarms, so holding the count at 80 raises the threshold above 4.0 and gives some
  of the gain back. We expect C1 (X minus PF14) of +0.01 to +0.04, with an interval about 0.05 wide
  (experiment 69's paired interval for a similar contrast was 0.054 wide), so `inconclusive` is a
  likely verdict. The zone split (below) is where the mechanism should show: X should gain +0.3 or
  more on transfers whose onset is in a zone and about 0 on the others. Arm G changes nothing worth
  reading (C2 within +-0.003), because the residual panel holds only 3 gaps of a day or longer in
  the 16 series. The six real transfers that experiment 69 sorted as "data gap" all lie on series
  12 and 13, which have no power telemetry at all before December 2025 and February 2026, so no
  bridging can reach them. X finds the onset of at least 2 of the 3 real transfers that experiment
  69 sorted as "monthly refit".
decision_rule: >-
  Three planned contrasts, each the paired difference in MW-weighted recall (the share of the moved
  MW that sits in found transfers), pooled with equal weight over the injected shapes S2, S3, and
  S4, on the fresh headline-band draw, for transfers that start in even months. Intervals are
  98.33% (Bonferroni across the three contrasts) from 20,000 resamples of whole calendar months.
  C1 is X minus PF14, C2 is G minus PF14, and C3 is XG (both changes) minus PF14. Each arm's
  threshold is retuned on odd months to the largest alarm count not above 80 unexplained alarms, on
  the 0.05 grid, as experiment 69 did for PF14. The verdict concerns C1 and is a judgement of cost,
  not a significance test. It is `supported` when the lower end of C1 lies above zero and its point
  value is at least +0.01. Otherwise, in this order, it is `small gain` when the lower end lies
  above zero and the point value is under +0.01; `refuted` when the lower end is at or below zero
  and the upper end is under +0.01; and `inconclusive` in every other case. The threshold of +0.01
  (1% of the moved MW) is set by the cost of the change: one more XGBoost fit per series per month
  in a detector that serves only training-time masks and the event table, which is cheaper than the
  pairing logic that experiment 69's threshold of +0.02 paid for. C2 and C3 carry no verdict,
  because G is expected to be null. The fresh long draw, the zone split, the zone-oversampled
  draw, the labelled real transfers, and the scaled onset tolerance are reported and decide
  nothing.
benchmark: >-
  PF14 on the stored residual (experiment 69's benchmark: D0M at 4.0 noise units, with the end at
  the first opposite step of at least half the onset step), at 80 odd-month unexplained alarms.
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled real transfers, as
  experiments 43, 55, 58, 60, 61, 66, and 69 do, not forecasts)
metric: >-
  MW-weighted recall (headline), pooled over S2, S3, and S4 on even months at 80 odd-month
  unexplained alarms. Unweighted recall, the share of the moved MW missed (no alarm at the onset,
  and a wrong end), recall by donor step in noise units split by metered receiver, and the count of
  the 35 labelled real transfers over 1 day found follow.
code: ml-experiments/00071-sw2-18-refit-offset-and-gap-bridging/
verdict: supported
best_arm: >-
  X, the step statistic taken from whichever of two monthly-refit residual models did not refit
  within 24 hours of the stamp. It finds 0.715 of the moved MW on the fresh headline-band transfers
  (98.33% interval 0.591 to 0.818), against 0.665 for PF14 (0.543 to 0.750). C1, the paired
  difference, is +0.050 (+0.012 to +0.107). The control L, which only lifts the suppression, finds
  0.720.
lessons: >-
  The detector's blind spot at a monthly refit cost about 5% of the moved MW, and all of that loss
  sat in the 13% of transfers that start or end within 24 hours of a refit (+0.431 there, -0.005
  elsewhere). Lifting the suppression recovers it, with or without the second model. On injected
  transfers the second model adds nothing to the lift (X minus L is -0.005); on the 3 real
  transfers that start at a refit, X finds all 3 and the lift alone finds 2. Gap bridging changes
  the outcome of none of 1,860 transfers, because the residual panel holds 3 gaps of a day or
  longer and the 6 real "data gap" transfers lie on series with no earlier telemetry.
advice:
  - >-
    Stop suppressing alarms near the monthly refit of the residual model and read the cost in
    unexplained alarms instead: the unexplained-alarm target of 80 then moves the threshold from
    4.0 to 4.05 or 4.1, and recall on injected transfers rises by +0.050 to +0.056 (C1, control L).
  - >-
    Fit a second residual model with refits half a month off only if real transfers matter more than
    injected ones; on injected transfers it adds -0.005 (-0.010 to -0.002) to the plain lift, and
    on the 3 real refit transfers it finds 3 against 2 (X minus L, real transfers table).
  - >-
    Do not build gap bridging for the 16-series Primary panel: it holds 3 gaps of a day or longer,
    and arm G changes the outcome of 0 of 1,860 transfers (C2).
  - >-
    Test future end rules and detectors on a draw that forces onsets into the refit zones as well
    as the ordinary draw, because the ordinary draw puts only 13% of transfers there and dilutes a
    +0.43 gain to +0.05 (zone split, figure 3).
  - >-
    Look for the remaining misses among the 8 sharp steps of 2 to 4 noise units and the 3 alarms
    2 to 12 hours from the labelled start, not among refits and gaps; none of those 11 transfers
    changed under any arm (real transfers table).
production_status: candidate
adopted_in: none
---

# Experiment 71 (SW2-18, part of the switching-event detector series): do the mechanical misses of the end rule come from refits and gaps?

**Lifting the detector's blind spot at the monthly refit recovers about 5% of the moved power, and
nearly all of that recovery needs only the suppression to be lifted. A second residual model refitted
half a month apart adds little on injected transfers, and bridging data gaps changes nothing.** A
load transfer is a switching operation that moves part of one Primary substation's demand to
neighbouring substations for some hours or days. The detector D0M raises an alarm when the 24-hour
average of a substation's residual (measured power minus an XGBoost forecast that uses no past
power) steps by 4.0 noise units or more, and PF14 ends each alarm interval at the first opposite
step of at least half the onset step. The XGBoost model is refitted on the 1st of every month, and
the detector ignores every alarm within 24 hours of a refit, so a transfer that starts there is never
found. We tested a second residual model refitted on the 16th, from which the step statistic is taken
whenever the first model refits within 24 hours, and a step statistic that bridges data gaps. On
injected transfers that start in even-numbered months, the two-model statistic found 0.715 of the
moved power against 0.665 for PF14, a gain of +0.050 (98.33% interval +0.012 to +0.107) at the same
number of false alarms. The whole gain lies in the transfers that start or end within 24 hours of a
refit. Of the 3 real transfers that start at a refit, the two-model statistic found all 3, where PF14
found none.

![MW-weighted recall by arm](figure_1_recall_by_arm.svg)

PF14, X, and the other arms use up to 14 days of hindsight, so they serve training-time masks and the
event table for National Grid Electricity Distribution (NGED), not live forecast inputs.

## Verdict

**`supported` by the rule fixed in the plan: the lower end of C1 lies above zero and its point value
is at least +0.01.** The threshold is a judgement of cost, not a significance test.

| Contrast (even months, S2, S3, and S4 pooled, headline band) | Change in MW-weighted recall |
|---|---|
| C1, X (two models) minus PF14 | +0.050 (+0.012 to +0.107) |
| C2, G (gap bridging) minus PF14 | +0.000 (+0.000 to +0.000) |
| C3, XG (both) minus PF14 | +0.050 (+0.012 to +0.107) |
| Control L (suppression lifted, no second model) minus PF14 | +0.056 (+0.016 to +0.113) |
| X minus L (what the second model adds) | -0.005 (-0.010 to -0.002) |
| Long transfers (15.6 to 60 days), X minus PF14 | -0.018 (-0.057 to +0.017) |

![The planned and descriptive contrasts](figure_2_contrasts.svg)

**Each arm finds this share of the moved MW, with the share it misses.** The missed share is one
minus the recall. For PF14, 0.301 of the moved MW sits in transfers with no alarm at the start and
0.035 in transfers whose start was found and whose end was wrong. For X the two shares are 0.265 and
0.019. Each arm's threshold is the lowest on a 0.05 grid whose odd-month count of unexplained
alarms (alarms more than 24 hours from any closure in NGED's switching log) is at most 80.

| Arm | Threshold | Unexplained alarms, odd / even months | Headline band | Long set |
|---|---|---|---|---|
| PF14 (benchmark) | 4.0 | 80 / 80 | 0.665 (0.543 to 0.750) | 0.400 (0.079 to 0.569) |
| X, two models | 4.1 | 75 / 80 | 0.715 (0.591 to 0.818) | 0.382 (0.082 to 0.538) |
| G, gaps bridged | 4.0 | 80 / 80 | 0.665 (0.543 to 0.750) | 0.400 (0.079 to 0.569) |
| XG, both | 4.1 | 75 / 80 | 0.715 (0.591 to 0.818) | 0.382 (0.082 to 0.538) |
| L, suppression lifted | 4.05 | 79 / 84 | 0.720 (0.597 to 0.822) | 0.388 (0.082 to 0.546) |

*Absolute MW-weighted recall on even months with 98.33% intervals. An onset tolerance of the larger
of 6 hours and 10% of the duration, instead of 2 hours, raises every arm by about 0.03 (PF14
0.694, X 0.741) and does not separate the arms.*

## The gain sits in the refit zones

**The gain comes from transfers that start or end within 24 hours of a refit, and the 13% of
transfers in that zone carry all of it.** A zone is the 24 hours either side of a refit of the model
refitted on the 1st. Of the 615 headline-band transfers per shape that start in even months, 81 start
or end in a zone. PF14 finds 0.177 of the moved MW there and X gains +0.431 (+0.182 to +0.696). Outside
the zones X is slightly worse, -0.005 (-0.011 to -0.001), because the threshold rises from 4.0 to
4.1 to hold the alarm count. A further 300 transfers with onsets forced into a zone (149 per shape
start in even months) show the mechanism cleanly: PF14 finds 0.000 and X finds 0.689, a gain of +0.689
(+0.420 to +0.845).

![The change in recall, split by whether a transfer touches a refit](figure_3_zone_split.svg)

**The second model does not improve on lifting the suppression for injected transfers.** The control
L only stops ignoring alarms near a refit and keeps the first model throughout. L finds 0.720 and X
0.715. L raises 84 unexplained alarms on even months where X raises 80, and 73 of its 311 real alarms
lie within 48 half-hours of a splice stamp or refit, against 66 of 297 for X and 46 of 297 for PF14.
Injected transfers sit on the real residual, so these counts show the refit jump producing a few
extra alarms in L, but the extra alarms are too few to cost L any recall.

## Gaps, and the real transfers

**Gap bridging changes no outcome, because the panel has nothing to bridge.** The residual panel
holds 188 gaps inside its 16 series. Only 7 last 13 half-hours or longer, and only 3 last a day or
longer. G differs from PF14 in the outcome of 0 of 1,860 transfers. The 6 real transfers that
experiment 69 called "data gap" all lie on series 12 and 13, which have no power telemetry before
December 2025 and February 2026, so no bridging can reach them.

**On the 35 real transfers over 1 day, X finds 10 against 8 for PF14, and its MW-weighted recall is
0.350 against 0.156.** The 3 real transfers that start at a refit are 3.9, 2.5, and 2.0 MW; X raises
an onset alarm for all 3, and L for 2 of them. X also loses the onset alarm of two small transfers
(0.6 and 0.5 MW) when the threshold rises to 4.1. The other 21 of the 24 transfers with no PF14 alarm
at the start are unchanged under every arm: 6 in "data gap", 3 with an alarm 2 to 12 hours from the
start, 8 with a sharp step of 2 to 4 noise units, 2 with a spread or late step, and 2 with no
visible step. The counts are too small for an interval.

| Class of the 24 transfers with no PF14 onset alarm | Transfers | Onset alarm under X | Found under X |
|---|---|---|---|
| Monthly refit | 3 | 3 | 3 |
| Data gap | 6 | 0 | 0 |
| Alarm outside the 2-hour tolerance | 3 | 0 | 0 |
| Sharp step of 2 to 4 noise units | 8 | 0 | 0 |
| Spread or late step of 2 to 4 noise units | 2 | 0 | 0 |
| No step visible | 2 | 0 | 0 |

## Recall by donor step

**X finds more of the large steps with a metered receiver and more of the 4 to 6 unit steps with no
metered receiver.** The donor is the substation that loses load and a receiver is a substation that
gains it; a receiver is metered when its power is in the telemetry. The table counts transfers found
out of transfers, summed over S2, S3, and S4, on the headline-band transfers that start in even
months, by the donor's step in 24-hour noise units. The printed report also gives G, XG, and L.

| Donor step | Receiver | Transfers | PF14 | X |
|---|---|---|---|---|
| 2 to 4 | metered | 135 | 85 | 90 |
| 4 to 6 | metered | 136 | 111 | 110 |
| 6 to 10 | metered | 184 | 173 | 178 |
| 10 or more | metered | 374 | 334 | 363 |
| 2 to 4 | unmetered | 537 | 68 | 72 |
| 4 to 6 | unmetered | 257 | 125 | 138 |

## Do the two models disagree?

**The two models' forecasts differ by about 2% of the substation's median power, with no jump at
either refit, so splicing the two statistics adds no visible step.** Where both statistics are clean,
the median absolute difference between them is 0.136 noise units (0.211 for the series with the
largest difference), and the 99th percentile of the difference is 0.889 noise units (median over the
16 series). That is below the 4.0 noise units that raise an alarm. The printed report does not count
return steps near the splice stamps; it counts alarms only.

![Difference between the two models' forecasts around a refit](figure_4_refit_profile.svg)

## What held in the prediction and what did not

- **Held:** the zone split (+0.431 where a gain of at least +0.3 was predicted, and -0.005 where
  about 0 was predicted); C2 within +-0.003 (it is +0.000); at least 2 of the 3 real refit transfers
  found (3 are); the threshold rising above 4.0 (it is 4.1).
- **Not held:** C1 of +0.01 to +0.04 (it is +0.050, above the range), and an interval about 0.05 wide
  (it is 0.095 wide).

## Limits

- **The detector uses up to 14 days of hindsight,** so it serves training-time masks and the event
  table, not live inputs.
- **Injected transfers do not distinguish X from L.** Injection adds the same power to both models'
  residuals on top of the real refit jumps, and the control L, which keeps the first model's jumps,
  loses no recall to them. Only the 3 real refit transfers separate the two arms.
- **The real transfers number 35 and the refit class 3,** so "3 against 2" is not a result with an
  interval.
- **The tracked relative residual,** which fixes the end of an interval when no return step is
  found, switches to the second model inside the zone, and that switch can itself change the level
  by the difference between the two models.
- **The scaled onset tolerance is scored on injected transfers only.**
- **A fresh draw is new in the placement of transfers and in the injection noise only.** It reuses
  the same residual table, series, months, and set of 25 long operations as experiments 66 and 69.

## Reviews

Reviews of the written page follow below once they have run.

## What the next experiment should answer

**Two remaining groups of misses are not mechanical: 8 real transfers with a sharp step of 2 to 4
noise units, and 3 with an alarm 2 to 12 hours from the labelled start.** Neither changed under any
arm here. A threshold lowered to reach the sharp small steps raises the unexplained-alarm count,
and experiment 69's weak-alarm rule did not reach them without losing recall elsewhere. The
scaled onset tolerance did not separate
the arms on injected transfers, so the 3 timing cases need a score on real transfers with a
duration-scaled tolerance to say whether they are a scoring artefact.

Code: `ml-experiments/00071-sw2-18-refit-offset-and-gap-bridging/`, with every number printed by
`e00071_report.py`.
