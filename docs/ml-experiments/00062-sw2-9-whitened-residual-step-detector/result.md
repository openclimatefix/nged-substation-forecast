---
id: "00062"
slug: sw2-9-whitened-residual-step-detector
status: planned
size: medium
hypothesis: >-
  Removing the half-hour-to-half-hour memory of the residual before the step statistic is taken
  (a first-order autoregressive filter, fitted per series and half-hour of day on odd months)
  makes the switching detector find a larger share of the MW moved by injected load transfers than
  the same detector on the unfiltered residual (D0M), at the same rate of unexplained alarms. The
  gain holds on transfers whose shape the detector does not assume.
prediction: >-
  MW-weighted recall (the share of the moved MW that sits in found transfers), 24-hour window, even
  months, pooled over the three non-matching shapes, rises by +0.00 to +0.06 over D0M (C1). The
  prediction is wide because the residual's memory cuts both ways: filtering shrinks a slow step
  to (1 minus the filter coefficient) of its size, and the noise shrinks by an amount that depends
  on the noise spectrum, which this experiment is the first to measure on this residual. Experiment
  58 found D0M's MW-weighted recall near 0.44 to 0.47. The ceiling on any gain is low, because
  about a third of the large transfers that D0M misses have the right start and a wrong end
  (experiment 60), and the interval ends come from the unfiltered residual in both detectors. A
  gain of +0.02 or more is therefore not the expected outcome, and `refuted` is the likeliest
  verdict.
decision_rule: >-
  Three planned contrasts, read at 98.33% (Bonferroni across C1 to C3) from 20,000 resamples of
  whole calendar months. C1 (the verdict) is the paired difference W0 minus D0M in MW-weighted
  recall at the 24-hour window, averaged with equal weight over the three non-matching shapes (S2,
  S3, S4), over experiment 58's 620 headline-band transfers that start in even months. `supported`
  when the C1 interval lies wholly above zero, the C1 point value is at least +0.02, the even-month
  unexplained-alarm count of W0 is no more than 10% above D0M's, and no non-matching shape has an
  interval wholly below zero. `partly supported` when the first three conditions hold and at
  least one non-matching shape has an interval wholly below zero. A C1 that passes while W0's
  alarm count is more than 10% above D0M's is `inconclusive`. `refuted` when the upper end of the C1
  interval lies below +0.02. `inconclusive` otherwise. The threshold is +0.02, not the +0.01 of
  experiments 55, 58 and 60, because the filter adds 48 fitted coefficients per series to the
  detector. The extra bar is a charge for that complexity; the experiment does not touch the
  serving path. C2 is the same contrast at the 4-hour window,
  and C3 is the paired count of labelled real transfers of more than 1 day found by each detector;
  both are reported without changing the verdict. Each shape is also read alone at the same level;
  those reads are descriptive. S1 (the shape of experiments 43, 55, and 58) and S5 (profile shifted 12
  hours) are reported and outside C1. The headline pools S2, S3, and S4 because the switching
  line's plan asks for the headline on shapes that the earlier harness did not use.
benchmark: >-
  D0M, which is D0 at a 24-hour window and experiment 43's threshold of 4.0 noise units, whose
  odd-month unexplained-alarm count (80) W0's threshold is matched to. At 4 hours, D0M at 6.0 noise
  units and 120 alarms.
row_set_id: >-
  none (this experiment scores detector alarms against injected transfers, as experiments 43, 55,
  58 and 60 do, not forecasts)
metric: >-
  MW-weighted recall (headline): the share of the MW moved by injected transfers that sits in
  transfers the detector finds, where a transfer is found when an alarm interval starts within 2
  hours and ends within the larger of 6 hours and 10% of the duration. Unweighted recall is
  secondary.
verdict: null
code: ml-experiments/00062-sw2-9-whitened-residual-step-detector/
---

# Experiment 62: does removing the residual's half-hour-to-half-hour memory help the switching detector?

*Plan, committed before any code runs. SW2-9 of the switching line.*

## What we ask and how

**We ask whether filtering out the residual's short memory lets the detector find more of the power
moved by a load transfer.** A load transfer is a switching operation that moves part of one Primary
substation's demand onto neighbouring substations. The detector raises an alarm when the 24-hour
mean of a substation's residual (measured minus predicted power) differs from the mean of the
previous 24 hours. Residuals are not independent from one half-hour to the next, which may inflate
the noise in that comparison. The whitened detector, W0, first subtracts from each residual the part
that the previous half-hour's residual predicts, and then applies the same 24-hour comparison. We
inject known load transfers into the power history and compare the share of the moved MW that each
detector finds (the MW-weighted recall), at the same number of alarms that no logged line closure
explains. A noise unit is the typical size of the step on a quiet day, estimated per series on odd
months.

## Design

**W0 changes only the residual that the step statistic reads.** For series *s* and half-hour of day
*h* (48 per day, in UTC), the coefficient is the least-squares slope of the residual at half-hour
*t* (in slot *h*) on the residual at *t* − 1, through the origin, over odd-month pairs where both are
present, clipped to 0 to 0.98. The whitened residual is *r*(*t*) minus the coefficient times
*r*(*t* − 1), and is missing where either residual is missing. The coefficients are fitted once, from the
uninjected residual of odd months only, and are not refitted for the even months. The fit does not
mask logged closures or labelled transfers, so a persistent real step can push a coefficient
towards the clip; the report counts the coefficients at the clip. An injected residual is whitened with the same
coefficients. Every step after the filter is the same as in D0: the step is the mean over the next 24 hours minus
the mean over the previous 24 hours, the noise scale is 1.4826 times the median absolute deviation
over odd months (of the whitened step), alarms are local peaks, alarms next to a monthly refit are
dropped, and intervals come from `build_intervals` run on the unfiltered relative residual. W0
therefore differs from D0M only in which stamps raise an alarm. A whitened persistent step
keeps its full size at the onset half-hour, so the 24-hour step still peaks at the onset and the
2-hour onset tolerance is fair to both detectors. W0's threshold is the lowest on a
0.05 grid whose odd-month count of alarms that no logged closure explains is at most D0M's
(experiment 58's `tune_d2` with the veto switched off), for the 24-hour and the 4-hour window
separately.

**The transfers and the shapes are the ones experiment 60 used.** The 620 headline-band transfers of experiment 58 that
start in even months (25% to 40% of the receiving load, over 1 day), each injected in five shapes:
S1 a share of the donor's profile (the earlier experiments' harness), S2 a constant MW, S3 a share
of each metered receiver's own profile, S4 a mix fitted to the labelled real transfers, and S5 the
donor's profile 12 hours ahead. All shapes move the same mean MW per transfer as S1. The headline
reads S2, S3 and S4, the shapes that experiments 43, 55 and 58 did not use. The harness check
requires D0M under S1 to reproduce experiment 58 transfer by transfer.

**What we report.** For each shape and detector: MW-weighted recall with month-resampled intervals,
unweighted recall, both detectors' odd-month and even-month alarm counts, and
the recall by the donor's step in noise units (bins below 2, 2 to 4, 4 to 6, 6 to 10, 10 or more,
in D0M's noise units for both detectors), split by whether any receiver is metered. For the
labelled real transfers of more than 1 day, the paired counts found, with MW-weighted recall using
the labelled median MW. Post hoc and labelled as such, per series: the lag-1 and lag-48 autocorrelation of the residual
before and after filtering, the ratio of the whitened step's noise scale to the raw step's, the
resulting signal-to-noise gain for a constant step, and the share of transfers with the right onset
and the wrong end. These numbers predict the verdict without injecting anything, and the write-up
says whether they did.

## Sizing

**Medium.** The five `plan-issue` triggers:

- **What gets stored:** private tables and the experiment's figures; nothing the production
  pipeline reads.
- **The production serving path:** not touched.
- **A degradation rule:** not touched.
- **More than one defensible design:** yes. The filter could be of order 2, per local half-hour, or
  fitted on all months. The first-order, per-UTC-half-hour, odd-month design is fixed here, as the
  switching line's plan names (experiment 37's F4). A coefficient per series, or a lag-48 filter, is
  left for a later experiment if this one shows memory at lag 48.
- **Code whose callers cannot be named without searching:** no. The harness is a copy of
  experiment 60's, and each function's callers are in the experiment folder.

## Earlier advice followed

- Experiment 60: report recall by donor step in noise units, split by metered receiver (followed).
- Experiment 60: score on shapes the detector does not assume, plus the labelled real transfers of
  more than 1 day (followed).
- Experiment 60 advises working on the interval end estimate before the step statistic. This plan
  departs from the advice: the brief for this experiment fixes it as written, and the end-estimate
  misses are reported, not addressed.
- Experiment 58 used a threshold of +0.01 for a pipeline step. This plan departs from that
  threshold and uses +0.02, with the reason in the decision rule.
