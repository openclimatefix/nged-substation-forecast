---
id: "00060"
slug: sw2-7-profile-weighted-step-detector
status: planned
size: medium
hypothesis: >-
  A step statistic that weights each half-hour by the series' expected power (the daily profile of
  the fitted weather model), D3, finds more of the MW moved by injected load transfers than the
  constant-step statistic, D0M, at the same rate of unexplained alarms. The gain holds on transfers
  whose shape D3 does not assume (a constant MW, a share of the receivers' own profiles, and a mix
  of the two fitted to real transfers), not only on a share of the donor's profile, which is how
  the injection built every transfer in experiments 43, 55 and 58.
prediction: >-
  MW-weighted recall (the share of the moved MW that sits in found transfers), 24-hour window, even
  months, pooled over the three non-matching shapes, rises by +0.01 to +0.04 over D0M (C1). On the
  matching shape (a share of the donor's profile) the gain is larger, +0.03 to +0.10 (C2). On the
  constant-MW shape D3 is level with D0M or up to 0.03 worse, because the weights only add noise
  there. Experiment 58 found 0.368 of 620 transfers found by the best veto arm and 0.355 by D0M, so
  D0M's MW-weighted recall is expected near 0.40 to 0.55, with large transfers found more often.
decision_rule: >-
  Three planned contrasts, read at 98.33% (Bonferroni across C1 to C3) from 20,000 resamples of
  whole calendar months. C1 (the verdict) is the paired difference D3 minus D0M in MW-weighted
  recall, averaged with equal weight over the three non-matching shapes, over experiment 58's 620
  headline-band transfers that start in even months. `supported` when the C1 interval lies wholly
  above zero, the C1 point value is at least +0.01, the even-month unexplained-alarm count of D3 is
  no more than 10% above D0M's, and no non-matching shape (S2, S3, S4) has an interval wholly
  below zero. `partly supported` when the first three hold and at least one non-matching shape has an
  interval wholly below zero. The MW weight of a transfer is its mean moved MW under S1, the same
  in every shape and for both detectors; a transfer that moves no MW has weight zero. `refuted` when the upper end of the C1 interval lies below +0.01.
  `inconclusive` otherwise. The threshold +0.01 is the one experiments 55 and 58 used, and it is
  low because D3 replaces the step statistic and adds no pipeline step. C2 is the same contrast
  on the matching shape and C3 is the paired count of labelled real transfers of more than 1 day
  found by each detector; both are reported without changing the verdict. Each shape is also read
  alone, with the same interval level; those per-shape reads are descriptive and add no verdict.
  S5 (below) is a guard, reported and outside C1.
benchmark: >-
  D0M, which is D0 at a 24-hour window and experiment 43's threshold of 4.0 noise units, whose
  odd-month unexplained-alarm count (80) D3's threshold is matched to.
row_set_id: >-
  none (this experiment scores detector alarms against injected transfers, as experiments 43, 55
  and 58 do, not forecasts)
metric: >-
  MW-weighted recall (headline): the share of the MW moved by injected transfers that sits in
  transfers the detector finds, where a transfer is found when an alarm interval starts within 2
  hours and ends within the larger of 6 hours and 10% of the duration. Unweighted recall is
  secondary.
verdict: null
code: ml-experiments/00060-sw2-7-profile-weighted-step-detector/
---

# Experiment 60: does a step that weights each half-hour by the daily profile find more of the moved load?

*Plan, committed before any code runs. SW2-7 of the switching line.*

## What we ask and how

**We ask whether a detector that expects a load transfer to follow the daily shape of the load
finds more of the moved power than a detector that expects a constant step.** Both detectors raise
an alarm when the 24-hour mean of one Primary substation's residual (measured minus predicted
power) differs from the mean of the previous 24 hours. The constant-step detector, D0M, takes a
plain mean. The profile-weighted detector, D3, weights each half-hour by the predicted power of the
same series, so that the high-load hours, where a transfer that is a share of the load moves the
most MW, count most. We inject known load transfers into the power history, with five different
shapes, and compare the share of the moved MW each detector finds. Experiments 43, 55 and 58
injected every transfer as a share of the donor's profile, which is exactly what D3 assumes, so a
gain on that shape alone would be circular. The verdict therefore rests on the three other shapes.

## Design

**D3 changes only the step statistic.** For series *s* and stamp *t*, D3's step is the mean of the
residual over the 24 hours from *t*, weighted by the series' expected power, minus the same
weighted mean over the 24 hours before *t*. Expected power is floored at 10% of its median, as in
`relative_residual`. Everything after the statistic is D0's: the noise scale is the robust spread
(1.4826 times the median absolute deviation) over odd months, the alarms are local peaks of the
step in noise units, alarms next to a monthly refit are dropped, and intervals are built by the
same `build_intervals`. D3's threshold is the lowest on a 0.05 grid whose odd-month count of alarms
that no logged closure explains is at most D0M's 80, found by experiment 58's `tune_d2` search with
the veto switched off.

**Transfers: the 620 headline-band transfers of experiment 58 that start in even months.** They are
25% to 40% of the receiving load for over 1 day (the original 160 and the extra 1,000 sets, without
the holiday set), with the donor, receivers, weights, onsets, and durations of experiment 58. Each
transfer is injected five times, once per shape, with the slow drift between 0.5 and 2 times the
start share that the earlier experiments used, and every shape is rescaled so that the mean moved MW of each transfer equals S1's.

| Shape | Moved MW at each half-hour | Matches D3's model? |
|---|---|---|
| S1, donor profile | start share × drift × the donor's power (the earlier harness) | Yes |
| S2, constant MW | start share × drift × the donor's mean power | No |
| S3, receiver profile | each metered receiver gains its weight × the same MW level × its own power relative to its mean; the donor loses the total, with the unmetered remainder flat | No |
| S4, fitted mix | S2's level with a share λ of the donor's profile and 1 − λ constant, λ drawn per transfer from the λ fitted to the labelled real transfers | No |
| S5, profile shifted 12 h | S1's profile moved by 12 hours, so the load peak falls at the donor's trough | No, and works against D3's weights |

A transfer with no metered receiver has no receiver profile, so S3 moves a constant MW for it, as S2 does. S3 is therefore also reported for the transfers with a metered receiver alone. The donor's and receivers' profiles are all daily shapes of Primary demand and so resemble each other, which is why S2 is the cleanest non-matching shape and S5 is the guard. The write-up reports the correlation between each shape's moved-MW profile and the donor's expected-power profile.

For S4, λ is fitted for each labelled real transfer (visibility `conserved` or `one_sided`) of at
least 6 hours with a known donor: least squares of the donor's residual during the transfer on a
constant and the donor's expected power relative to its mean, and λ is the profile term's share of
the fitted mean, clipped to 0 to 1 (a transfer whose fitted mean is not positive is dropped). The fit uses the same real transfers that C3 scores, so S4 and C3 are in-sample with respect to each other. If fewer than 10 transfers have a usable fit, S4 instead draws
λ uniformly from 0 to 1, and the write-up says so. Thresholds and noise scales come from the
uninjected residual and odd months only, and scoring uses even months only.

**What we report.** For each shape and detector: recall and MW-weighted recall with month-resampled
intervals, the share of the moved MW that is missed (1 minus the MW-weighted recall), and a table
of recall by the donor's step in noise units (bins below 2, 2 to 4, 4 to 6, 6 to 10, and 10 or
more), split by whether any receiver is metered. For the labelled real transfers of more than 1
day, the paired counts of transfers found by each detector, with MW-weighted recall using the
labelled median MW. The stronger neighbour veto is not rerun: experiment 58 already ran it, and its
result is quoted.

## Sizing

**Medium.** D3's threshold is matched by alarm count, and the grid may settle below the target; the write-up reports D3's odd-month and even-month counts beside D0M's. D3's gain is a signal-to-noise gain, because the weighted mean of a constant step is unbiased. The five `plan-issue` triggers:

- **What gets stored:** only private tables and the experiment's figures; nothing the production
  pipeline reads.
- **The production serving path:** not touched.
- **A degradation rule:** not touched.
- **More than one defensible design:** yes. The weights could be the expected power or its square,
  and the non-matching shapes could be drawn differently. Both are fixed here before any fit.
- **Code whose callers cannot be named without searching:** no. The harness is a copy of
  experiment 58's, and each function's callers are in the experiment folder.

## Earlier advice followed

- Experiment 58: score on shapes the detector does not assume, on the labelled real transfers, and
  by bins of step in noise units (followed; this is the design).
- Experiment 58: look at why about half of the transfers stepping by 4 noise units or more are
  missed (followed in part: the step-by-shape table shows whether the misses depend on shape).
- Experiment 58: do not build a better step statistic to close the no-metered-receiver gap
  (not in conflict: this experiment tests the MW-weighted headline, not that gap).
- Experiment 55: threshold of +0.01 for a pipeline step (followed).
