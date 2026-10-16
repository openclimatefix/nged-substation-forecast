---
id: "00071"
slug: sw2-18-refit-offset-and-gap-bridging
status: planned
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
---

# Experiment 71 (SW2-18, part of the switching-event detector series): do the mechanical misses of the end rule come from refits and gaps?

**Status: planned. This page is the plan; the write-up replaces it when the experiment has run.**

## Question and approach

**We ask whether the load transfers that the detector cannot see for mechanical reasons can be
found by fixing those reasons, without raising the number of false alarms.** A load transfer is a
switching operation that moves part of one Primary substation's demand to neighbouring substations
for some hours or days. The detector D0M raises an alarm when the 24-hour average of a substation's
residual (measured power minus an XGBoost forecast that uses no past power) steps by 4.0 noise
units or more, and PF14 ends each alarm interval at the first opposite step of at least half the
onset step. Experiment 69 found that 24 of 35 labelled real transfers lasting over 1 day have no
alarm at their start. Of those 24, 3 start where the monthly refit of the residual's XGBoost model
suppresses the alarm, and 6 start in a "data gap". The remaining 15 are small steps or timing
errors, which this experiment does not address.

- **Arm X, two models offset by half a month.** The stored residual comes from an XGBoost model
  refitted on the 1st of each month. We fit a second model, identical in every setting, that
  refits on the 16th. A step statistic computed across a refit mixes the jump of the refit with
  the step of the transfer. At each stamp we therefore take the statistic of the first model unless
  that model refits within 24 hours of the stamp, and the statistic of the second model otherwise.
  The statistic is never computed across a refit, so no alarm is suppressed and no return step is
  blocked. The noise scale is the odd-month spread of the combined statistic, computed once on the
  uninjected residual. The second model trains on the rows before its refit date minus 7 days,
  from 2019-09-01, with the same columns, weights, parameters, and seed as the first, so it is
  free of lookahead in the same way; the harness refits the first model for one origin and
  requires the stored predictions back. The statistic of the first model and of the second model
  differ where the two forecasts differ, which makes a small step at each of the four splice
  stamps per month. The write-up reports that difference in noise units and the alarms and return
  steps within 48 half-hours of a splice. Arm L, a control, is the stored residual with the
  suppression lifted and no splice, which shows how many extra alarms the refit jumps produce.
- **Arm G, gap bridging.** For each gap of a day or longer (48 half-hours or more) in a series'
  residual, we set the missing statistic at one stamp, the middle of the gap, to the mean residual
  in the 24 hours after the gap minus the mean in the 24 hours before it. Each of the two flanks
  must hold at least 75% of its stamps, and no refit of that model may fall between the start of
  the before-flank and the end of the after-flank, otherwise the statistic stays missing. Arm XG
  bridges each model separately before the two are spliced. The noise scale stays the one of the
  unbridged statistic. The alarm sits at the middle of the gap, so the 2-hour onset tolerance only
  matches a transfer that starts within 2 hours of that stamp; the write-up counts the gaps of 6.5
  hours or longer and the injected transfers that start inside one, and says whether arm G is
  testable at all.
- **Arm XG** applies both changes. **PF14** is the benchmark and is not changed.
- **The six "data gap" transfers cannot be reached by arm G.** Series 12 and 13 have no power
  telemetry before December 2025 and February 2026, so there is no data on either side of those
  transfers. The residual panel holds only 3 gaps of a day or longer in its 16 series (series 15
  once and series 17 twice). Arm G is in the experiment because the plan asked for it, and its
  expected result is no change; the write-up counts how many of the 1,160 injected transfers an
  arm G alarm touches.

## What is scored

**Every arm is tuned and scored with experiment 69's harness.** The harness adds 1,160 injected
transfers in the headline band (25% to 40% of the receiving load, lasting over 1 day) and 400
transfers lasting 15.6 to 60 days to the stored power, under five injection shapes. The headline
pools the three shapes (S2, S3, and S4) that no detector in this series assumes: a constant MW, a
share of each receiver's profile, and a mix fitted to the labelled real transfers. A fresh draw
uses new seeds (6900 and 7900 in experiment 69; 7100 and 8100 here), and the harness check also
reproduces experiment 69's own draw. An injected transfer adds the
same power to both models' residuals, so the two models see the same transfer. The harness check
reproduces experiment 69's PF14 flags on its transfers before any new arm is scored.

- **Contrast reading.** C1, C2, and C3 are read on transfers that start in even months, because
  every threshold is tuned on odd months. Each arm's unexplained-alarm count is reported for odd
  and even months beside the contrasts.
- **Zone split.** C1 is also reported separately for transfers whose onset or end lies within 24
  hours of a refit of the first model and for the rest, with the counts. A gain outside the zones
  would come from the retuned threshold or the changed noise scale, not from the splice.
- **Zone-oversampled draw.** A further 300 headline-band transfers whose onsets are forced into
  the zones are scored as a test of the mechanism, and decide nothing.
- **Standard tables.** Each arm gets recall by donor step in noise units split by metered
  receiver, the unweighted recall, and the share of the moved MW missed, split into no alarm at
  the onset and a wrong end.
- **Real transfers.** The 35 labelled real transfers over 1 day are scored for every arm, with the
  3 "monthly refit" transfers and the 3 "alarm outside the 2-hour tolerance" transfers reported
  separately.
- **Scaled onset tolerance, as a scoring sensitivity.** Every arm is also scored with an onset
  tolerance equal to the larger of 6 hours and 10% of the transfer's duration (the end
  tolerance), which decides nothing.
- **Hindsight.** PF14 uses up to 14 days of hindsight and the second model uses the whole history
  before each refit, so the detector serves training-time masks and the event table for National
  Grid Electricity Distribution (NGED), not live forecast inputs.

## Size, and the earlier advice

**The size is medium.** The five triggers: (1) nothing is stored beyond private tables and
figures; (2) the production serving path is not touched; (3) no degradation rule is touched; (4)
more than one design would test the hypothesis, since the two models' statistics could be
averaged instead of chosen, so a Sonnet reviewer read the plan for that choice; (5) the
experiment copies experiment 69's harness and the callers are known. It gets a Sonnet plan review,
a Sonnet code review, a Sonnet fact check, and a Sonnet prose review.

**The plan follows the advice of experiments 61, 65, 66, and 69** to use PF14 as the end rule, to
fix the monthly-refit suppression and test a longer onset tolerance before lowering any threshold,
to count the data-gap transfers as unobserved, and to hold the alarm count fixed when comparing
alarm rules. It departs from none of it.

## Compute

Slot C: 4 cores, no GPU. The offset model is 16 series times about 55 monthly refits of the
settings experiment 17 used (half-life 0.5 years, 500 rounds), run as 4 worker processes with one
thread each, one output file per series.
