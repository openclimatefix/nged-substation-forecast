---
id: "00065"
slug: sw2-14-combined-single-series-and-neighbour-score
status: done
size: medium
hypothesis: >-
  One alarm score that adds a substation's own 24-hour step evidence to the opposite-step evidence
  from its neighbouring substations, with neither acting as a gate, finds more of the MW moved by
  injected load transfers than the 24-hour single-series detector (D0M), the neighbour-confirmation
  gate (D1), and the neighbour veto (D2, experiment 55), at the same count of unexplained alarms
  on odd months.
prediction: >-
  MW-weighted recall (the share of the moved MW that sits in found transfers), even months, pooled
  over the injection shapes S2, S3, and S4 of experiment 60, with interval ends taken from the
  strongest opposite step in the next 14 days (PE, experiment 61): D0M reaches 0.63 to 0.65. The
  likelihood-ratio score (LR) gains +0.00 to +0.04 over D0M and the logistic-regression score
  (LOGIT) gains +0.01 to +0.05. The gain sits in transfers with a metered receiver whose donor step
  is under 4.0 noise units, which D0M cannot alarm on, and no group of transfers loses more than
  0.02. D1 and D2 are expected at or below D0M, as in experiments 43, 55, and 58.
decision_rule: >-
  Six planned contrasts, read at 99.17% (Bonferroni across six) from 20,000 resamples of whole
  calendar months, each version (LR, LOGIT) minus each benchmark (D0M, D1, D2) in MW-weighted recall,
  PE end rule, equal weight over S2, S3, and S4, over experiment 58's 620 headline-band transfers
  that start in even months. Every arm is held to at most 80 unexplained alarms on odd months, the
  count of D0M. A version is `supported` when its interval against each of D0M, D1, and D2 lies
  wholly above zero, its point value against D0M is at least +0.02, and its even-month
  unexplained-alarm count is no more than 10 above D0M's (a count near 80 varies by about 9 from
  chance alone). A version is `partly supported` when its intervals against D1 and D2 lie wholly
  above zero but its interval against D0M does not. The experiment is `refuted` when the upper end
  of both versions' intervals against D0M lies below +0.02, `inconclusive` otherwise, and takes the
  better version's verdict if the two differ. The threshold of +0.02 is set by the work of porting:
  the combined score adds a neighbour-step computation and a calibration fitted on injected
  transfers to a detector that needs neither. The same contrasts under the tracked end rule are
  reported and do not change the verdict.
benchmark: >-
  D0M (the 24-hour step detector at 4.0 noise units, 80 unexplained odd-month alarms), D1 (the
  neighbour-confirmation gate), and D2 at experiment 55's veto strength of 2.0, each with the PE end
  rule, and each with its alarm threshold matched to 80 unexplained odd-month alarms.
row_set_id: >-
  none (this experiment scores detector alarms against injected transfers, as experiments 43, 55,
  58, 59, and 61 do, not forecasts)
metric: >-
  MW-weighted recall (headline): the share of the MW moved by injected transfers that sits in
  transfers the detector finds, where a transfer is found when an alarm interval starts within 2
  hours and ends within the larger of 6 hours and 10% of the duration. Unweighted recall is
  secondary.
code: ml-experiments/00065-sw2-14-combined-single-series-and-neighbour-score/
verdict: inconclusive
best_arm: >-
  Of the pre-registered arms, LOGIT (logistic regression) finds 0.665 of the moved MW against
  D0M's 0.646, a gain of +0.019 (99.17% interval -0.011 to +0.062). LR (likelihood-ratio sum) gains
  +0.005 (-0.010 to +0.014) and is refuted. The post hoc arm LOGIT_SAME, which adds the count of
  same-sign neighbour steps, finds 0.681 (+0.034, +0.013 to +0.091).
lessons: >-
  Adding the neighbours' opposite-sign steps to a substation's own step did not find more of the
  moved MW than the own step alone. The neighbour gate D1 finds about 0.1 less. What did help, in a
  post hoc arm, was the count of neighbours that stepped the same way, which is the evidence the
  veto uses: it raised recall by +0.034 over D0M and by +0.023 over the veto. The gain sits in
  transfers whose donor step is 2.5 to 6 noise units. Every candidate has an MVA-only neighbour,
  so the MVA flag carried no information.
advice:
  - >-
    Do not build the likelihood-ratio sum of own step and neighbour opposite step; it gained +0.005
    over D0M with an upper bound under +0.02 (C1, figure 1).
  - >-
    Pre-register a score that uses both the neighbours' opposite-sign steps and the count of
    same-sign neighbour steps before porting any neighbour evidence, because only the same-sign
    count raised recall, by +0.034 (+0.013 to +0.091) over D0M (post hoc arm, figure 1).
  - >-
    Use D0M with the PE end rule, not D1, as the benchmark for neighbour evidence: D1 at 80
    unexplained alarms finds about 0.1 less than D0M (figure 1).
  - >-
    Do not use an MVA-neighbour flag as a feature on the 16 trial Primaries: all 16 have an
    MVA-only neighbour, so the flag is constant (fitted scores section).
production_status: not-for-production
adopted_in: none
---

# Experiment 65: does one score from a substation's own step and its neighbours' opposite steps find more of the moved load?

**Adding the neighbouring substations' opposite-sign steps to a substation's own step did not find
clearly more of the moved load than the own step alone.** In a post hoc variant, adding the count of
neighbours that stepped the same way raised the share found. The neighbour veto of experiment 55
already uses that count. Whether one score that combines the own step, the opposite neighbour step,
and the same-sign count beats the own step alone is untested.

We asked whether load transfers are found more often when one alarm score adds a substation's own
step to the largest opposite step among its neighbours. A load transfer moves power off a donor
Primary substation and onto receiver substations. The step of a substation is the change in the
average of its residual (measured minus predicted power) from the 24 hours before a half-hour to
the 24 hours after it, in noise units (multiples of the usual size of that step on a quiet day). We
injected 620 known load transfers into the measured power of 16 Primary substations and counted
the share of the moved megawatts (MW) found. A transfer counts as found when an alarm starts within
2 hours of its start and ends within the larger of 6 hours and 10% of its duration. Every detector
was held to 80 alarms on odd months that match no closure in the NGED switching log. The
single-series detector D0M finds 0.646 of the moved MW (0.354 missed). The logistic-regression
score (LOGIT) finds 0.665, which is +0.019 more (99.17% interval, from resampling whole calendar
months, -0.011 to +0.062). The likelihood-ratio score (LR) finds 0.651, which is +0.005 more
(-0.010 to +0.014).

![Gain in MW-weighted recall over D0M](figure_1_gain_over_d0m.svg)

## Verdict

**The experiment is inconclusive, because the better score's interval against D0M includes zero.**
The likelihood-ratio score (LR) is refuted, because the upper end of its interval
against D0M is below +0.02. The logistic-regression score (LOGIT) is inconclusive, because its interval
against D0M includes zero and its upper end is above +0.02. The experiment takes the more favourable
verdict of the two scores, and that choice is not covered by the correction for six planned
contrasts (a Bonferroni correction). All intervals are
99.17%, from 20,000 resamples of whole calendar months, over the 620
injected transfers that move 25% to 40% of the receiving load for over 1 day and start in even
months, pooled with equal weight over the constant-MW, receiver-profile, and fitted-mix shapes
(none of which the scores assume). Interval ends come from the strongest opposite step in the next
14 days, the end rule of [experiment 61](../00061-sw2-6-paired-step-intervals/result.md). That rule
needs up to 14 days of hindsight, so detectors built on it serve training-time masks and the NGED
event table, not live forecast inputs.

| Contrast (gain in MW-weighted recall) | LR | LOGIT |
|---|---|---|
| Against D0M, the 24-hour single-series detector | +0.005 (-0.010 to +0.014) | +0.019 (-0.011 to +0.062) |
| Against D1, the neighbour gate | +0.101 (+0.043 to +0.176) | +0.116 (+0.050 to +0.231) |
| Against D2, the neighbour veto of [experiment 55](../00055-sw2-8-neighbour-veto-single-series-detector/result.md) | -0.007 (-0.036 to +0.008) | +0.008 (-0.028 to +0.042) |

**Both scores beat the neighbour gate D1 clearly, but D1 is the weakest detector.** At 80
unexplained alarms D1 finds 0.550 of the moved MW against D0M's 0.646. D2 finds 0.658. The
pre-registered rule needed an interval above zero against D0M as well, and neither score has one.

## Recall by arm

**The best arms still miss about a third of the moved MW.** The table gives MW-weighted recall under
each end rule, pooled over the three shapes.

| Arm | Tracked end | Strongest-opposite-step end (PE) | MW missed under PE |
|---|---|---|---|
| D0M | 0.455 | 0.646 | 0.354 |
| D1 | 0.389 | 0.550 | 0.450 |
| D2 | 0.459 | 0.658 | 0.342 |
| LR | 0.460 | 0.651 | 0.349 |
| LOGIT | 0.466 | 0.665 | 0.335 |
| LOGIT + same-sign count (post hoc) | 0.479 | 0.681 | 0.319 |
| D2 at veto 1.0 (post hoc) | 0.466 | 0.666 | 0.334 |

The same contrasts under the tracked end have the same signs and sizes (LR +0.005, LOGIT +0.011).
Even-month unexplained-alarm counts on the real residual were 80 for D0M, 78 for LR, 54 for LOGIT,
72 for D2, and 49 for D1, so no combined score bought its recall with extra alarms. The odd-month
counts are all 80 or 79.

## Where the gain sits

**The scores gain where the donor step is 2.5 to 6 noise units, mostly for transfers with a metered
receiver.** A noise unit is the usual size of the 24-hour step statistic on a quiet day. With a
metered receiver (a receiver whose own power is measured) the share of transfers found at a donor
step of 2.5 to 4 rises from 0.58 for D0M to 0.71 for LOGIT with the same-sign count. Without a
metered receiver the gain spans 2.5 to 6 noise units and is larger at 4 to 6 (0.37 for D0M, 0.55
for that arm). Donor steps below 2.5 noise units are found for 5 of 40 transfers at most. A donor
step above 6 gains nothing, because D0M already finds those transfers.

![Share found by donor step](figure_2_share_found_by_step.svg)

## Labelled real transfers

**On the 35 labelled real transfers of more than 1 day, the scores find 6 to 8 against D0M's 7
under PE, so the real data cannot separate them.** With the tracked end the scores find 5 and D0M
finds 2. Alarms at the right start are found for 14 (LR), 12 (LOGIT), and 11 (D0M) of the 35. The
count is too small to rank the scores. Of the 35, 20 start in even months and 15 in odd months.

## Fitted scores

**The fitted LR lookup tables rise with both statistics, and LOGIT weights the neighbour step about as
much as the own step.** The standardised logistic coefficients are 2.59 on the own step, 2.95 on the
largest opposite neighbour step, and -1.11 on the number of neighbours. The coefficient on the
MVA-neighbour flag is zero, because every one of the 16 series has an MVA-only neighbour, so the flag
is constant. The post hoc same-sign count has a coefficient of -0.85, which means a same-sign
neighbour step lowers the score, as the veto does.

## Limits

- **Training data come from the same injection generator as the test transfers.** Training uses odd
  months and scoring uses even months, but both draw donors and receivers the same way, so the
  scores may do worse on real transfers.
- **The candidate floor of 2.5 noise units caps recall.** The 507 negative examples are alarm
  candidates from the real odd-month residual.
- **Weights and shrinkage.** Every shape weights a transfer by its MW moved under the donor-profile
  shape, so the three shapes share one set of weights. The logistic regression's penalty of 1.0 is
  weak against about 2,700 training candidates, so it shrinks the coefficients very little.
- **Alarm counts.** LR and both logistic scores land on exactly 80 unexplained odd-month alarms.
  D0M, D1, and D2 reach at most 80 on a 0.05 grid of thresholds (D0M 80, D1 80, D2 80, veto 1.0 at
  79).
- **The post hoc arms** were chosen after reading the pre-registered result, so their intervals
  overstate the evidence. Their contrasts use the same level as the planned ones.

## What the next experiment should answer

**Does one pre-registered score that uses the largest opposite neighbour step, the count of
same-sign neighbour steps, and the own step together beat D0M by at least +0.02?** The post hoc arm
suggests it does (+0.034, +0.013 to +0.091). A pre-registered run would also say whether the gain
holds on the labelled real transfers, where this experiment could not tell the arms apart.
