---
id: "00061"
slug: sw2-6-paired-step-intervals
status: planned
size: medium
hypothesis: >-
  A detector that looks for an onset step and its opposite return step together, scoring the pair
  and choosing the return stamp with a prior on the duration of real transfers, finds more of the MW
  moved by injected load transfers than the best single-window detector (D0M, the 24-hour step with
  a tracked end), at the same rate of unexplained alarms. The gain comes from interval ends: about a
  third of D0M's transfers with a donor step of 6 noise units or more have the right start and a
  wrong end (experiment 60).
prediction: >-
  MW-weighted recall (the share of the moved MW that sits in found transfers), 24-hour window, even
  months, pooled over the three shapes S2, S3 and S4 of experiment 60, rises by +0.03 to +0.08 for
  the paired detector with the labelled-duration prior (PL) over D0M (C1). The flat-prior detector
  (PF) gains +0.02 to +0.06 (C2). Experiment 60 found D0M at 0.44 to 0.47, so PL is expected at
  0.48 to 0.54.
decision_rule: >-
  Three planned contrasts, read at 98.33% (Bonferroni across C1 to C3) from 20,000 resamples of
  whole calendar months. C1 is the paired difference PL minus D0M in MW-weighted recall, averaged
  with equal weight over S2, S3 and S4, over experiment 58's 620 headline-band transfers that start
  in even months. C2 is the same difference for PF minus D0M. C3 is the paired count of labelled
  real transfers over 1 day found by PL (with a cross-fitted prior, below) against D0M, reported
  without changing the verdict. `supported` when the C1 interval lies wholly above zero, the C1
  point value is at least +0.02, the C2 interval lies wholly above zero, the even-month
  unexplained-alarm count of PL is no more than 10% above D0M's, and the same holds for PF.
  `partly supported` when the C1 conditions hold and the C2 interval does not lie wholly above
  zero (the gain then depends on the duration prior, which is optimistic on injected transfers,
  see Limits). `refuted` when the upper end of the C1 interval lies below +0.02. `inconclusive`
  otherwise. The threshold is +0.02, higher than the +0.01 of experiments 55, 58 and 60, because
  the paired search adds a pipeline step (a search for the return stamp) beside the step statistic.
benchmark: >-
  D0M, which is D0 at a 24-hour window and experiment 43's threshold of 4.0 noise units, whose
  odd-month unexplained-alarm count (80) the thresholds of PL and PF are matched to.
row_set_id: >-
  none (this experiment scores detector alarms against injected transfers, as experiments 43, 55,
  58 and 60 do, not forecasts)
metric: >-
  MW-weighted recall (headline): the share of the MW moved by injected transfers that sits in
  transfers the detector finds, where a transfer is found when an alarm interval starts within 2
  hours and ends within the larger of 6 hours and 10% of the duration. Unweighted recall is
  secondary.
code: ml-experiments/00061-sw2-6-paired-step-intervals/
---

# Experiment 61: does pairing an onset step with its return step find more of the moved load?

*Planned. The sections below are the plan; the write-up replaces them when the experiment has run.*

## Question

**Experiment 60 found that the shape of the step does not limit the 24-hour detector, and that the
end of the alarm's interval does.** A load transfer is a switching operation that moves part of one
Primary substation's demand onto neighbouring substations for some hours or days. D0M raises an
alarm when the 24-hour average of a substation's residual (measured power minus the power the
lagless XGBoost weather model predicted) steps away from the average of the previous 24 hours. D0M
then follows the level of the residual for as long as it keeps at least half its starting size, and
calls that the interval. Among transfers with a donor step of 6 noise units or more, about a third
have an alarm at the right start and a wrong end. A noise unit is the typical size of the step on a
quiet day. Only 2 of the 35 labelled real transfers of more than 1 day are found.

## Plan

**The paired-step detector scores an onset and a return step together.** At a candidate onset `s`
with step `z(s)` in noise units, and a candidate return stamp `e`, at least 24 hours later and
within 60 days, the pair score is `(|z(s)| + r(e)) / sqrt(2)`. Here `z` is D0's 24-hour level shift
(the mean of the next 24 hours minus the mean of the previous 24 hours, at each stamp) divided by
the series' odd-month noise scale, and `r(e) = -sign(z(s)) * z(e)` is the return step, positive when
the level steps back. An onset of +5 and a return of -4 scores 6.4. The interval's end is `e`, the
first stamp at the old level, the same convention as D0M's `end`; the alarm is known 24 hours later
(`e` + 48 half-hours), so every arm here is a retrospective interval estimator, scored on the
interval alone, as D0M is. The detector
takes every local peak of `|z|` of 2.5 noise units or more as a candidate onset, picks the `e` with
the largest score plus a log prior on the duration `e - s`, and keeps the alarm when that total
reaches a threshold. A candidate with no return step of at least 1.5 noise units keeps D0M's
tracked end and scores `|z(s)| / sqrt(2)` plus the lowest prior value.

- **PE** (the simplest test of the end estimate) keeps D0M's alarms exactly (threshold 4.0, so the
  unexplained alarms are identical by construction) and replaces D0M's tracked end by the stamp `e`
  between 24 hours and 14 days after the onset with the largest return step `r(e)`, when that step
  is 1.5 noise units or more, else D0M's tracked end. PE has no prior. It is reported beside C1 and C2
  as a descriptive contrast (PE minus D0M) with the same interval level, and adds no verdict.
- **PL** takes the prior from a kernel density of the log durations of the labelled real transfers
  (experiment 41's set, 2 hours to 30 days), a Gaussian kernel (Silverman bandwidth) on the log duration in hours, evaluated at each
  half-hour duration searched, as a log ratio to the density's peak, floored at -1 and added to the
  pair score with weight 1. A candidate with no return step takes the floor.
- **PF** has a flat prior, and is the guard against PL's prior leaking information about the injected
  durations (see Limits).
- **Thresholds** of PL and PF are each set on odd months so that the unexplained-alarm count (alarms
  more than 24 hours from any logged closure on the series) equals D0M's 80, as in experiment 60.
  Even months are scored only.
- **Candidates** are local peaks of `|z|` within +/- 24 hours, as D0 uses, skipping peaks within 24
  hours of a monthly refit. The return stamp `e` may be any stamp, not only a peak.
- **A return step does not raise a new alarm:** a candidate on a series, of the opposite sign, within
  24 hours of an accepted pair's end is dropped.
- **Shapes and population** are experiment 60's: 620 headline-band transfers (25% to 40% of the
  receiving load, over 1 day) starting in even months, in shapes S1 (a share of the donor's
  profile), S2 (a constant MW), S3 (a share of each receiver's profile), S4 (a mix fitted to real
  transfers), and S5 (the guard, profile shifted 12 hours). The detector assumes none of the
  shapes, so C1 and C2 pool S2, S3 and S4, as experiment 60 did; S1 and S5 are reported alone.

**The match on alarm count is a match on count, not on composition,** and the maximum over up to
2,800 return stamps inflates the score of noise candidates; the odd-month calibration absorbs the
average inflation. The report therefore also prints, for each arm, the odd-month and even-month
alarm counts, the share of alarms near a logged closure, and the number of candidate onsets above
the 2.5 floor.

**Standard tables, as experiments 55, 58 and 60 print them:** recall by donor step in noise units,
split by metered receiver; MW-weighted recall as the headline with unweighted recall second; and
the share of the moved MW missed, stated in the take-home message. A post hoc table (no predicted values) counts, among
transfers with a donor step of 6 noise units or more, those found, those with the right start and a
wrong end, and those with no alarm at the start.

## Sizing (the five triggers)

- **What gets stored:** only the private tables under `data/ml-experiments/00061-*`; nothing that
  production reads.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes (the score, the prior, the candidate floor), which
  is why the prior is tested in two arms and the other choices are fixed here before any fit.
- **Code whose callers cannot be named without searching:** no; the experiment copies experiment
  60's folder and changes the detector and the report.

That is a medium experiment: a Sonnet plan review, a Sonnet code review, a Sonnet numbers review,
and a Sonnet prose review.

## Earlier advice followed

- Experiment 60: work on the interval end before the step statistic (this experiment); report recall
  by donor step in noise units, split by metered receiver (standard tables above).
- Experiment 60: look at why only 2 of 35 labelled real transfers are found; C3 and the
  real-transfer start counts report it, and this experiment does not claim to explain it.

## Limits

- **The injected durations come from the same labelled set as the prior,** so PL's gain on injected
  transfers is an upper bound; C2 (PF) and C3 (a cross-fitted prior for the real transfers: the
  prior for each half of the 35 comes from the other half) are the honest reads.
- **The 620 transfers were used in experiments 55, 58 and 60.** The shapes share the same events.
